"""Multiplexed remote Episode client, with acknowledged cancellation and draining.

The delta protocol is adapted from PrimeRL (see NOTICE). Unlike fire-and-forget
cancellation, terminal futures stay registered until server cleanup is confirmed.
"""

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import zmq
import zmq.asyncio
from verifiers.v1.episode import WireEpisode

from miles.env_server.delta import EpisodeAssembly, pack, unpack
from miles.env_server.types import (
    PROTOCOL_VERSION,
    BaseRequest,
    CancelRequest,
    CancelResponse,
    HealthRequest,
    HealthResponse,
    RunRequest,
    RunResponse,
)

logger = logging.getLogger(__name__)


class EnvDrainError(RuntimeError):
    """The remote worker could not confirm termination within the cleanup deadline."""


async def _settle_cleanup(task):
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


@dataclass
class _Pending:
    future: asyncio.Future
    method: str
    on_delta: Callable[[dict], None] | None = None


class EnvClient:
    def __init__(self, address: str, *, timeout: float = 3600, cancel_timeout: float = 30):
        if timeout <= 0 or cancel_timeout <= 0:
            raise ValueError("Request and cancellation timeouts must be positive")
        self.address = address
        self.timeout = timeout
        self.cancel_timeout = cancel_timeout
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(address)
        self._pending: dict[str, _Pending] = {}
        self._receiver: asyncio.Task | None = None
        self._closing = False
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    def _start_receiver(self):
        if self._receiver is None:
            self._receiver = asyncio.create_task(self._receive())

    async def _receive(self):
        try:
            while True:
                frames = await self.socket.recv_multipart()
                self._handle_frame(frames)
                # Drain a bounded batch; an unrelated run must not wait for a
                # thread roundtrip per tiny delta, nor monopolize the event loop.
                for _ in range(511):
                    try:
                        ready = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                        self._handle_frame(ready.result())
                    except zmq.Again:
                        break
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            for pending in self._pending.values():
                if not pending.future.done():
                    pending.future.set_exception(ConnectionError(f"Environment receive loop failed: {error}"))

    def _handle_frame(self, frames):
        if len(frames) != 3:
            logger.warning("Malformed environment server response")
            return
        request_id, kind, body = frames
        pending = self._pending.get(request_id.decode())
        if pending is None or pending.future.done():
            return
        try:
            if kind == b"delta":
                if pending.on_delta is None:
                    raise ValueError("Unexpected delta for non-run request")
                pending.on_delta(unpack(body))
            elif kind == b"reply":
                pending.future.set_result(body)
            else:
                raise ValueError(f"Unknown response frame type {kind!r}")
        except Exception as error:
            # A lost reward scalar need not change node counts. Never silently
            # skip a malformed delta and return a possibly corrupted episode.
            pending.future.set_exception(error)

    async def _request(self, request: BaseRequest, response_type, *, timeout, on_delta=None, request_id=None):
        if self._closed or (self._closing and request.method != "cancel"):
            raise RuntimeError("Environment client is closing or closed")
        self._start_receiver()
        request_id = request_id or uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = _Pending(future, request.method, on_delta)
        sent = False
        terminal_error = False
        try:
            payload = pack(request.model_dump(mode="json", serialize_as_any=True))
            async with asyncio.timeout(timeout):
                # send() may queue a frame before its await gets cancelled. Once
                # attempted, always issue a cancel on an abandoned run.
                sent = True
                await self.socket.send_multipart([request_id.encode(), request.method.encode(), payload])
                body = await asyncio.shield(future)
            response = response_type.model_validate(unpack(body))
            if not response.success and request.method != "health":
                terminal_error = True
                if response.error_code in {"inference_cleanup", "worker_lost"}:
                    raise EnvDrainError(response.error or "Remote inference termination was not confirmed")
                raise RuntimeError(response.error or "Environment request failed")
            return response
        except BaseException:
            if sent and request.method == "run" and not terminal_error:
                await self.cancel_and_wait(request_id)
            raise
        finally:
            self._pending.pop(request_id, None)
            # Retrieve an exception delivered while the caller was draining.
            if future.done() and not future.cancelled():
                future.exception()

    async def cancel_and_wait(self, request_id: str):
        """Cancel one wire request and wait for worker execution AND payload cleanup."""
        cleanup = asyncio.create_task(
            self._request(CancelRequest(request_id=request_id), CancelResponse, timeout=self.cancel_timeout)
        )
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # A second caller cancellation must not detach the cleanup task.
            await _settle_cleanup(cleanup)
            raise
        except Exception as error:
            raise EnvDrainError(f"Could not confirm cancellation at {self.address}: {request_id}") from error

    async def describe(self, timeout: float = 2) -> HealthResponse:
        response = await self._request(HealthRequest(), HealthResponse, timeout=timeout)
        if response.protocol_version != PROTOCOL_VERSION:
            raise RuntimeError(f"Incompatible env-server protocol: {response.protocol_version}")
        return response

    async def health(self, timeout: float = 2) -> bool:
        try:
            response = await self.describe(timeout)
            return response.success and response.ready_workers > 0
        except (TimeoutError, ConnectionError):
            return False

    async def wait_for_server_startup(self, timeout: float = 120, interval: float = 0.2):
        deadline = time.monotonic() + timeout
        while (remaining := deadline - time.monotonic()) > 0:
            if await self.health(timeout=min(remaining, 2)):
                return
            await asyncio.sleep(min(interval, max(0, deadline - time.monotonic())))
        raise TimeoutError(f"Environment server did not become ready: {self.address}")

    async def run(self, client, model: str, sampling, task_data: dict, on_delta=None, *, timeout: float | None = None):
        assembly = EpisodeAssembly()

        def apply(delta):
            assembly.apply(delta)
            if on_delta is not None:
                on_delta(delta)

        request_id = uuid.uuid4().hex
        response = await self._request(
            RunRequest(client=client, model=model, sampling=sampling, task_data=task_data),
            RunResponse,
            timeout=self.timeout if timeout is None else timeout,
            on_delta=apply,
            request_id=request_id,
        )
        try:
            if response.head is None:
                raise ValueError("Environment server returned no Episode head")
            # Validation happens once at the boundary; the wire task is deliberately
            # non-strict so env-specific data survives without loading its plugin.
            return WireEpisode.model_validate(assembly.finish(response.head, response.traces))
        except BaseException:
            await self.cancel_and_wait(request_id)
            raise

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _settle_cleanup(self._close_task)
            raise

    async def _close(self):
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            runs = [key for key, pending in self._pending.items() if pending.method == "run"]
            failures = await asyncio.gather(*(self.cancel_and_wait(key) for key in runs), return_exceptions=True)
            if self._receiver is not None:
                self._receiver.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._receiver
            for pending in self._pending.values():
                if not pending.future.done():
                    pending.future.set_exception(ConnectionError("Environment client closed"))
            self.socket.close()
            self.ctx.term()
            self._closed = True
            for failure in failures:
                if isinstance(failure, BaseException):
                    raise EnvDrainError("One or more remote runs could not be drained during close") from failure
