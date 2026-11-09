"""Process broker for CPU environment workers, independent of Miles GPU actors.

The ROUTER/DEALER and parent-death-pipe design derives from PrimeRL (see NOTICE).
Each worker must finish env.serving() startup before routing. Crashed workers fail
owned requests explicitly and may restart; stateful episodes are never replayed.
"""

import asyncio
import contextlib
import logging
import multiprocessing as mp
import os
import signal
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zmq
import zmq.asyncio
from verifiers.v1.runtimes import set_base_sandbox_labels
from verifiers.v1.utils.loaders import resolve_env_config

from miles.env_server.config import ElasticPoolConfig, EnvServerConfig, config_fingerprint
from miles.env_server.delta import pack, unpack
from miles.env_server.server import LATE_CANCEL_SECONDS, EnvServer
from miles.env_server.types import BaseResponse, CancelResponse, HealthResponse


@dataclass
class Worker:
    process: Any
    pipe: Any
    dealer: Any
    address: str
    started: float
    ready: bool = False
    active: int = 0
    restarts: int = 0


@dataclass
class Pending:
    client_id: bytes
    request_id: bytes
    worker: Worker
    method: bytes


def _arm_teardown(death_pipe=None):
    def terminate(*_):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    if death_pipe is not None:

        def watch_parent():
            with contextlib.suppress(Exception):
                death_pipe.recv()
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=watch_parent, daemon=True).start()


def serve_worker(*, address: str, config_data: dict, death_pipe):
    _arm_teardown(death_pipe)
    config = EnvServerConfig.model_validate(config_data)
    logging.basicConfig(level=config.log_level)
    set_base_sandbox_labels(config.sandbox_labels)
    server = EnvServer(
        resolve_env_config(config.env), address, config.serve.max_concurrent, payload=config.serve.payload
    )
    try:
        asyncio.run(server.run(on_ready=lambda health: death_pipe.send(health)))
    except KeyboardInterrupt:
        pass


class EnvServerPool:
    def __init__(self, config: EnvServerConfig):
        self.config = config
        self.env_config = resolve_env_config(config.env).model_dump(mode="json", serialize_as_any=True)
        self._ipc = tempfile.TemporaryDirectory(prefix="miles-env-")
        self._mpctx = mp.get_context("spawn")
        self.ctx = zmq.asyncio.Context()
        self.frontend = self.ctx.socket(zmq.ROUTER)
        self.frontend.setsockopt(zmq.LINGER, 0)
        self.frontend.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.frontend.bind(config.serve.address)
        self.address = self.frontend.getsockopt_string(zmq.LAST_ENDPOINT)
        self._poller = zmq.asyncio.Poller()
        self._poller.register(self.frontend, zmq.POLLIN)
        self.workers: list[Worker] = []
        self.pending: dict[bytes, Pending] = {}
        self.routes: dict[tuple[bytes, bytes], bytes] = {}
        self.replied: dict[tuple[bytes, bytes], tuple[float, Worker, bytes]] = {}
        self._lost: dict[tuple[bytes, bytes], tuple[float, str]] = {}
        self._ready_reported = False

    def _spawn(self, restarts: int = 0):
        address = f"ipc://{self._ipc.name}/{uuid.uuid4().hex}"
        parent, child = self._mpctx.Pipe()
        process = self._mpctx.Process(
            target=serve_worker,
            kwargs={"address": address, "config_data": self.config.model_dump(mode="json"), "death_pipe": child},
        )
        process.start()
        child.close()
        dealer = self.ctx.socket(zmq.DEALER)
        dealer.setsockopt(zmq.LINGER, 0)
        dealer.connect(address)
        worker = Worker(process, parent, dealer, address, time.monotonic(), restarts=restarts)
        self.workers.append(worker)
        self._poller.register(dealer, zmq.POLLIN)

    def health_response(self):
        ready = sum(worker.ready and worker.process.is_alive() for worker in self.workers)
        pool = self.config.serve.pool
        required = 1 if isinstance(pool, ElasticPoolConfig) else pool.num_workers
        return HealthResponse(
            success=ready >= required,
            error=None if ready >= required else "Environment workers are starting or unavailable",
            ready_workers=ready,
            total_workers=len(self.workers),
            env_config=self.env_config,
            config_fingerprint=config_fingerprint(self.env_config),
        )

    async def _reply(self, client_id, request_id, response):
        with contextlib.suppress(zmq.ZMQError):
            await self.frontend.send_multipart([client_id, request_id, b"reply", pack(response)])

    async def _forward(self, client_id, request_id, method, payload, worker):
        internal_id = uuid.uuid4().hex.encode()
        self.pending[internal_id] = Pending(client_id, request_id, worker, method)
        self.routes[(client_id, request_id)] = internal_id
        worker.active += int(method == b"run")
        await worker.dealer.send_multipart([internal_id, method, payload])

    async def _route_cancel(self, client_id, request_id, payload):
        try:
            target = (client_id, unpack(payload)["request_id"].encode())
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            await self._reply(client_id, request_id, BaseResponse(success=False, error=f"Invalid cancel: {error}"))
            return
        if target in self._lost:
            await self._reply(
                client_id,
                request_id,
                CancelResponse(
                    success=False, cancelled=True, drained=False, error=self._lost[target][1], error_code="worker_lost"
                ),
            )
            return
        internal = self.routes.get(target)
        active = self.pending.get(internal)
        late = self.replied.get(target)
        if active is not None:
            worker = active.worker
        elif late is not None:
            _, worker, internal = late
        else:
            await self._reply(client_id, request_id, CancelResponse(cancelled=False))
            return
        await self._forward(client_id, request_id, b"cancel", pack({"request_id": internal.decode()}), worker)

    async def _route_request(self):
        frames = await self.frontend.recv_multipart()
        if len(frames) != 4:
            return
        client_id, request_id, method, payload = frames
        if (client_id, request_id) in self.routes:
            await self._reply(client_id, request_id, BaseResponse(success=False, error="Duplicate request ID"))
        elif method == b"health":
            await self._reply(client_id, request_id, self.health_response())
        elif method == b"cancel":
            await self._route_cancel(client_id, request_id, payload)
        elif method == b"run":
            workers = [worker for worker in self.workers if worker.ready and worker.process.is_alive()]
            if not workers:
                await self._reply(
                    client_id, request_id, BaseResponse(success=False, error="No ready environment worker")
                )
                return
            await self._forward(client_id, request_id, method, payload, min(workers, key=lambda worker: worker.active))
            pool = self.config.serve.pool
            if isinstance(pool, ElasticPoolConfig) and len(self.workers) < pool.max_workers:
                active = sum(worker.active for worker in self.workers)
                if active >= 0.9 * len(self.workers) * pool.multiplex:
                    self._spawn()
        else:
            await self._reply(client_id, request_id, BaseResponse(success=False, error=f"Unknown method {method!r}"))

    async def _relay(self, worker):
        frames = await worker.dealer.recv_multipart()
        if len(frames) != 3:
            return
        internal_id, kind, payload = frames
        request = self.pending.get(internal_id)
        if request is None or request.worker is not worker:
            return
        if kind == b"reply":
            self.pending.pop(internal_id)
            self.routes.pop((request.client_id, request.request_id), None)
            if request.method == b"run":
                worker.active -= 1
                self.replied[(request.client_id, request.request_id)] = (time.monotonic(), worker, internal_id)
        try:
            await self.frontend.send_multipart([request.client_id, request.request_id, kind, payload])
        except zmq.ZMQError:
            # The client disappeared; a detached reply must not retain remote work.
            await worker.dealer.send_multipart([b"", b"cancel", pack({"request_id": internal_id.decode()})])

    async def _check_workers(self):
        for worker in list(self.workers):
            if not worker.ready and worker.pipe.poll():
                try:
                    health = HealthResponse.model_validate(worker.pipe.recv())
                    if health.config_fingerprint != config_fingerprint(self.env_config):
                        raise RuntimeError("Worker resolved a different environment configuration")
                    worker.ready = health.success
                except (EOFError, OSError):
                    pass
            expired = not worker.ready and time.monotonic() - worker.started > self.config.serve.startup_timeout
            if worker.process.is_alive() and not expired:
                continue
            await self._worker_failed(worker, "startup timeout" if expired else "process exited")
        now = time.monotonic()
        for key, (completed, _, _) in list(self.replied.items()):
            if now - completed >= LATE_CANCEL_SECONDS:
                self.replied.pop(key, None)
        for key, (completed, _) in list(self._lost.items()):
            if now - completed >= LATE_CANCEL_SECONDS:
                self._lost.pop(key, None)

    async def _worker_failed(self, worker, reason):
        for internal, pending in list(self.pending.items()):
            if pending.worker is worker:
                self.pending.pop(internal)
                self.routes.pop((pending.client_id, pending.request_id), None)
                message = f"Environment worker lost ({reason}); remote inference cleanup is unknown; episode was not replayed"
                if pending.method == b"run":
                    self._lost[(pending.client_id, pending.request_id)] = (time.monotonic(), message)
                await self._reply(
                    pending.client_id,
                    pending.request_id,
                    BaseResponse(success=False, error=message, error_code="worker_lost"),
                )
        self.replied = {key: value for key, value in self.replied.items() if value[1] is not worker}
        self._poller.unregister(worker.dealer)
        worker.dealer.close()
        worker.pipe.close()
        if worker.process.is_alive():
            worker.process.terminate()
        await asyncio.to_thread(worker.process.join, self.config.serve.shutdown_timeout)
        if worker.process.is_alive():
            worker.process.kill()
            await asyncio.to_thread(worker.process.join)
        self.workers.remove(worker)
        if self.config.serve.restart_workers and worker.restarts < self.config.serve.max_worker_restarts:
            self._spawn(restarts=worker.restarts + 1)

    async def run(self, on_ready=None):
        pool = self.config.serve.pool
        for _ in range(1 if isinstance(pool, ElasticPoolConfig) else pool.num_workers):
            self._spawn()
        try:
            while True:
                await self._check_workers()
                if not self._ready_reported and self.health_response().success:
                    self._ready_reported = True
                    if on_ready is not None:
                        on_ready(self.address)
                if not self.workers:
                    raise RuntimeError("All environment workers failed and exhausted restart limits")
                events = dict(await self._poller.poll(timeout=100))
                if self.frontend in events:
                    await self._route_request()
                for worker in list(self.workers):
                    if worker.dealer in events:
                        await self._relay(worker)
        finally:
            await self._shutdown()

    async def _shutdown(self):
        for worker in self.workers:
            worker.pipe.close()
            if worker.process.is_alive():
                worker.process.terminate()
        # Keep relaying terminal responses while workers drain. Otherwise an engine
        # abort failure could be replaced with a generic successful shutdown ACK.
        deadline = time.monotonic() + self.config.serve.shutdown_timeout
        while time.monotonic() < deadline:
            events = dict(await self._poller.poll(timeout=100))
            for worker in self.workers:
                if worker.dealer in events:
                    await self._relay(worker)
            if not events and all(not worker.process.is_alive() for worker in self.workers):
                break
        for worker in self.workers:
            if worker.process.is_alive():
                worker.process.kill()
            await asyncio.to_thread(worker.process.join)
            worker.dealer.close()
        for pending in self.pending.values():
            await self._reply(
                pending.client_id,
                pending.request_id,
                BaseResponse(
                    success=False,
                    error="Server shut down before remote inference cleanup could be confirmed",
                    error_code="worker_lost",
                ),
            )
        self.frontend.close()
        self.ctx.term()
        self._ipc.cleanup()


def serve_env(config: EnvServerConfig, address_queue=None):
    """Spawn a pool, publishing its bound address only after worker readiness."""
    _arm_teardown()
    os.environ.setdefault("VF_RUN_ID", config.run_id or uuid.uuid4().hex)
    server = EnvServerPool(config)

    def ready(address):
        if config.address_file is not None:
            path = Path(config.address_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(address + "\n")
            temporary.replace(path)
        if address_queue is not None:
            address_queue.put(address)

    try:
        asyncio.run(server.run(on_ready=ready))
    except KeyboardInterrupt:
        pass
