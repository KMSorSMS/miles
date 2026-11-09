"""One CPU worker executes complete verifiers Episodes and streams trace deltas.

Adapted from PrimeRL's env/server.py; see NOTICE. A cancelled run replies only
once env cleanup and any off-loop payload keep/discard have completed.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable

import zmq
import zmq.asyncio
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.episode import Episode
from verifiers.v1.errors import find_inference_cleanup_error
from verifiers.v1.task import Task
from verifiers.v1.utils.loaders import load_environment

from miles.env_server.config import PayloadConfig, config_fingerprint
from miles.env_server.delta import DeltaStreamer, TraceSummary, dump, pack, unpack
from miles.env_server.payload import PayloadHook
from miles.env_server.types import BaseResponse, CancelRequest, CancelResponse, HealthResponse, RunRequest, RunResponse

logger = logging.getLogger(__name__)
LATE_CANCEL_SECONDS = 60.0


class EnvServer:
    def __init__(self, config: EnvConfig, address: str, max_concurrent: int, *, payload: PayloadConfig | None = None):
        self.env = load_environment(config)
        self.env_config = config.model_dump(mode="json", serialize_as_any=True)
        self.task_cls = type(self.env.taskset).task_type()
        self.data_cls = self.task_cls.data_type()
        excluded = [name for name, field in self.data_cls.model_fields.items() if field.exclude]
        if excluded:
            raise ValueError(f"Served TaskData excludes serialized fields: {excluded}")
        self._gate = asyncio.Semaphore(max_concurrent)
        self._running: dict[tuple[bytes, bytes], asyncio.Task] = {}
        self._delivered: dict[tuple[bytes, bytes], tuple[float, list[str]]] = {}
        self._drain_failures: dict[tuple[bytes, bytes], tuple[float, str]] = {}
        self._hook = PayloadHook(payload or PayloadConfig())
        self.ctx = zmq.asyncio.Context()
        self.frontend = self.ctx.socket(zmq.ROUTER)
        self.frontend.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.frontend.setsockopt(zmq.LINGER, 0)
        self.frontend.bind(address)
        self.address = self.frontend.getsockopt_string(zmq.LAST_ENDPOINT)

    def _build_task(self, data: dict) -> Task:
        return self.task_cls(self.data_cls.model_validate(data), self.env.config.taskset.task)

    async def _keep(self, episode: Episode, sampling):
        task = asyncio.create_task(asyncio.to_thread(self._hook.keep, episode, sampling))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # A thread cannot be cancelled. Never acknowledge cancellation while it
            # can still publish arrays; join it and then release its ownership.
            while not task.done():
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.shield(task)
            if not task.cancelled():
                task.exception()
            self._hook.discard(episode)
            raise

    async def _run(self, request: RunRequest, client_id: bytes, request_id: bytes) -> RunResponse:
        ctx = ModelContext(client=request.client, model=request.model, sampling=request.sampling)
        (slot,) = self.env.slots(self._build_task(request.task_data))

        async def send_delta(delta: dict):
            await self.frontend.send_multipart([client_id, request_id, b"delta", pack(delta)])

        episode = None
        try:
            async with DeltaStreamer(lambda: slot.traces, send_delta) as streamer:
                episode = await self.env.run_slot(slot, ctx, self._gate, on_trace=streamer.watch)
                await self._keep(episode, ctx.sampling)
            response = RunResponse(
                head=dump(episode, exclude={"traces"}),
                traces=[TraceSummary(id=t.id, nodes=len(t.nodes), calls=len(t.calls)) for t in episode.traces],
            )
            self._delivered[(client_id, request_id)] = (time.monotonic(), self._hook.kept_ids(episode))
            return response
        except BaseException:
            if episode is not None:
                self._hook.discard(episode)
            raise

    def _expire_delivered(self):
        now = time.monotonic()
        for history in (self._delivered, self._drain_failures):
            for key, (completed, _) in list(history.items()):
                if now - completed >= LATE_CANCEL_SECONDS:
                    history.pop(key, None)

    async def _cancel(self, client_id: bytes, request: CancelRequest) -> CancelResponse:
        key = (client_id, request.request_id.encode())
        task = self._running.get(key)
        if task is not None:
            if not task.cancelling():
                task.cancel()
            while not task.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(task)
            if not task.cancelled():
                task.exception()
        delivered = self._delivered.pop(key, None)
        if delivered is not None:
            self._hook.discard_ids(delivered[1])
        failed = self._drain_failures.get(key)
        if failed is not None:
            return CancelResponse(
                success=False, error=failed[1], error_code="inference_cleanup", cancelled=True, drained=False
            )
        return CancelResponse(cancelled=task is not None or delivered is not None)

    def health_response(self) -> HealthResponse:
        return HealthResponse(env_config=self.env_config, config_fingerprint=config_fingerprint(self.env_config))

    async def _handle(self, client_id: bytes, request_id: bytes, method: bytes, payload: bytes):
        key = (client_id, request_id)
        try:
            raw = unpack(payload)
            if method == b"health":
                response = self.health_response()
            elif method == b"run":
                response = await self._run(RunRequest.model_validate(raw), client_id, request_id)
            elif method == b"cancel":
                response = await self._cancel(client_id, CancelRequest.model_validate(raw))
            else:
                response = BaseResponse(success=False, error=f"Unknown method: {method!r}")
        except BaseException as error:
            critical = find_inference_cleanup_error(error)
            if critical is not None:
                message = f"Inference cleanup was not confirmed: {critical}"
                if method == b"run":
                    self._drain_failures[key] = (time.monotonic(), message)
                response = BaseResponse(success=False, error=message, error_code="inference_cleanup")
            elif isinstance(error, asyncio.CancelledError):
                response = BaseResponse(success=False, error="Cancelled: episode execution and cleanup finished")
            elif isinstance(error, Exception):
                logger.exception("Environment request failed")
                response = BaseResponse(success=False, error=f"{type(error).__name__}: {error}")
            else:
                raise
        try:
            try:
                encoded = pack(response)
            except Exception as error:
                encoded = pack(BaseResponse(success=False, error=f"Response encoding failed: {error}"))
                response = BaseResponse(success=False)
            await self.frontend.send_multipart([client_id, request_id, b"reply", encoded])
        except (zmq.ZMQError, asyncio.CancelledError):
            response = BaseResponse(success=False)
        finally:
            if method == b"run":
                self._running.pop(key, None)
            if not response.success:
                delivered = self._delivered.pop(key, None)
                if delivered is not None:
                    self._hook.discard_ids(delivered[1])

    async def _serve_requests(self, tasks):
        poller = zmq.asyncio.Poller()
        poller.register(self.frontend, zmq.POLLIN)
        while True:
            self._expire_delivered()
            if self.frontend not in dict(await poller.poll(timeout=1000)):
                continue
            frames = await self.frontend.recv_multipart()
            if len(frames) != 4:
                logger.warning("Discarding malformed environment request")
                continue
            client_id, request_id, method, payload = frames
            key = (client_id, request_id)
            if method == b"run" and key in self._running:
                await self.frontend.send_multipart(
                    [client_id, request_id, b"reply", pack(BaseResponse(success=False, error="Duplicate request ID"))]
                )
                continue
            task = asyncio.create_task(self._handle(client_id, request_id, method, payload))
            if method == b"run":
                self._running[key] = task
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            # Give the handler its first turn before queued cancellation. Cancelling
            # an unstarted coroutine would skip its cleanup/reply entirely.
            await asyncio.sleep(0)

    async def run(self, on_ready: Callable[[dict], None] | None = None):
        tasks: set[asyncio.Task] = set()
        try:
            async with self.env.serving():
                try:
                    if on_ready is not None:
                        on_ready(self.health_response().model_dump())
                    await self._serve_requests(tasks)
                finally:
                    for task in list(tasks):
                        if not task.cancelling():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await asyncio.to_thread(self._hook.close)
            self.frontend.close()
            self.ctx.term()
