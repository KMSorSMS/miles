"""Bounded HTTP storage for native verifiers node arrays, materialized by Miles.

Adapted from PrimeRL's HTTP payload ownership pattern (see NOTICE). We transport
node fields directly, without its trainer's branch packing, padding, or fallback
arrays. Missing/corrupt data is an error, never fabricated training evidence.
"""

import asyncio
import contextlib
import hashlib
import logging
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
from verifiers.v1.episode import Episode, WireEpisode
from verifiers.v1.graph import MessageNode

from miles.env_server.config import PayloadConfig
from miles.env_server.delta import TRAINING_ARRAY_FIELDS, dump, pack, unpack

PAYLOAD_INFO_KEY = "miles_training_payload"
logger = logging.getLogger(__name__)


class _HTTPServer(ThreadingHTTPServer):
    request_queue_size = 1024
    daemon_threads = True


class PayloadServer:
    """GET is repeatable, DELETE is idempotent; TTL bounds abandoned clients' blobs."""

    def __init__(self, config: PayloadConfig):
        self.config = config
        self._blobs: dict[str, tuple[float, bytes]] = {}
        self._bytes = 0
        self._lock = threading.Lock()
        self._closed = threading.Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                blob = owner.get(self.path.removeprefix("/"))
                self.send_response(200 if blob is not None else 404)
                self.send_header("Content-Length", str(len(blob or b"")))
                self.send_header("Content-Type", "application/msgpack")
                self.end_headers()
                if blob is not None:
                    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                        self.wfile.write(blob)

            def do_DELETE(self):
                owner.delete(self.path.removeprefix("/"))
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        self.http = _HTTPServer((config.bind_host, config.port), Handler)
        host = config.advertised_host or config.bind_host
        self.url = f"http://{host}:{self.http.server_port}"
        self._http_thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self._reaper_thread = threading.Thread(target=self._reap, daemon=True)
        self._http_thread.start()
        self._reaper_thread.start()

    def _expire(self):
        now = time.monotonic()
        expired = []
        for key, (created, _) in self._blobs.items():
            if now - created < self.config.ttl_seconds:
                break
            expired.append(key)
        for key in expired:
            self._bytes -= len(self._blobs.pop(key)[1])

    def _reap(self):
        while not self._closed.wait(min(30, self.config.ttl_seconds)):
            with self._lock:
                self._expire()

    def put(self, blob: bytes) -> str:
        with self._lock:
            self._expire()
            if self._closed.is_set():
                raise RuntimeError("Payload server is closed")
            if self._bytes + len(blob) > self.config.max_bytes:
                raise BufferError("Environment payload store is full; consume or discard finished episodes")
            key = uuid.uuid4().hex
            self._blobs[key] = (time.monotonic(), blob)
            self._bytes += len(blob)
            return key

    def get(self, key: str) -> bytes | None:
        with self._lock:
            self._expire()
            item = self._blobs.get(key)
            return item[1] if item is not None else None

    def delete(self, key: str):
        with self._lock:
            item = self._blobs.pop(key, None)
            if item is not None:
                self._bytes -= len(item[1])

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        self.http.shutdown()
        self.http.server_close()
        self._http_thread.join()
        self._reaper_thread.join()
        with self._lock:
            self._blobs.clear()
            self._bytes = 0


class PayloadHook:
    """Move excluded node fields into one independently owned blob per trace."""

    def __init__(self, config: PayloadConfig):
        self.config = config
        self._server: PayloadServer | None = None
        self._server_lock = threading.Lock()

    @property
    def server(self):
        with self._server_lock:
            if self._server is None:
                self._server = PayloadServer(self.config)
            return self._server

    def keep(self, episode: Episode, sampling: Any = None):
        stored = []
        try:
            for trace in episode.traces:
                nodes = [dump(node, include=TRAINING_ARRAY_FIELDS) for node in trace.nodes]
                if not any(value is not None for node in nodes for value in node.values()):
                    continue
                blob = pack({"version": 1, "trace": trace.id, "nodes": nodes})
                key = self.server.put(blob)
                stored.append(key)
                trace.info[PAYLOAD_INFO_KEY] = {
                    "url": f"{self.server.url}/{key}",
                    "id": key,
                    "size": len(blob),
                    "sha256": hashlib.sha256(blob).hexdigest(),
                }
                # The final delta carries the reference. Fields never stream inline.
                for node in trace.nodes:
                    for field in TRAINING_ARRAY_FIELDS:
                        if hasattr(node, field):  # config-access-exempt: traverse VF's shared training-array schema
                            setattr(node, field, None)
        except BaseException:
            for key in stored:
                self.server.delete(key)
            raise

    @staticmethod
    def kept_ids(episode: Episode) -> list[str]:
        return [trace.info[PAYLOAD_INFO_KEY]["id"] for trace in episode.traces if PAYLOAD_INFO_KEY in trace.info]

    def discard_ids(self, keys: list[str]):
        if self._server is not None:
            for key in keys:
                self._server.delete(key)

    def discard(self, episode: Episode):
        self.discard_ids(self.kept_ids(episode))

    def close(self):
        if self._server is not None:
            self._server.close()


async def _fetch(client: httpx.AsyncClient, reference: dict) -> bytes:
    for attempt in range(3):
        try:
            response = await client.get(reference["url"])
            response.raise_for_status()
            blob = response.content
            if len(blob) != reference["size"] or hashlib.sha256(blob).hexdigest() != reference["sha256"]:
                raise ValueError("Environment training payload failed length/checksum validation")
            return blob
        except (httpx.TransportError, httpx.HTTPStatusError) as error:
            if isinstance(error, httpx.HTTPStatusError) and error.response.status_code < 500:
                raise
            if attempt == 2:
                raise
            await asyncio.sleep(0.2 * 2**attempt)
    raise AssertionError("unreachable")


async def _delete(client: httpx.AsyncClient, reference: dict):
    for attempt in range(3):
        try:
            response = await client.delete(reference["url"])
            response.raise_for_status()
            return
        except httpx.HTTPError:
            if attempt == 2:
                logger.warning("Payload DELETE failed; server TTL will reclaim %s", reference["id"], exc_info=True)
                return
            await asyncio.sleep(0.2 * 2**attempt)


def _restore(trace, blob: bytes):
    payload = unpack(blob)
    if payload.get("version") != 1 or payload.get("trace") != trace.id:
        raise ValueError("Environment payload identity/version mismatch")
    nodes = payload["nodes"]
    if len(nodes) != len(trace.nodes):
        raise ValueError("Environment payload node count does not match trace")
    restored = []
    for node, arrays in zip(trace.nodes, nodes, strict=True):
        if set(arrays) - TRAINING_ARRAY_FIELDS:
            raise ValueError("Unexpected field in environment training payload")
        restored.append(MessageNode.model_validate({**dump(node, context={"token_arrays": True}), **arrays}))
    trace.nodes = restored
    trace.info.pop(PAYLOAD_INFO_KEY, None)


async def _settle_cleanup(task):
    # Cleanup has its own finite HTTP deadlines. Repeated task cancellation must
    # not close its HTTP client while DELETE operations are still using it.
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def _cleanup_payloads(client, downloads, references):
    for task in downloads:
        if not task.done():
            task.cancel()
    await asyncio.gather(*downloads, return_exceptions=True)
    await asyncio.gather(*(_delete(client, reference) for reference in references))


async def materialize_episode(episode: WireEpisode, *, timeout: float = 60) -> WireEpisode:
    """Fetch, validate, restore all training arrays; delete every referenced blob.

    On failure/cancellation every sibling download settles before DELETE, and cleanup
    is shielded. Returned traces can use native verifiers branch array accessors.
    """
    traces = [trace for trace in episode.traces if PAYLOAD_INFO_KEY in trace.info]
    references = [trace.info[PAYLOAD_INFO_KEY] for trace in traces]
    if not references:
        return episode
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        downloads = [asyncio.create_task(_fetch(client, reference)) for reference in references]
        try:
            blobs = await asyncio.gather(*downloads)
            for trace, blob in zip(traces, blobs, strict=True):
                _restore(trace, blob)
        finally:
            cleanup = asyncio.create_task(_cleanup_payloads(client, downloads, references))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await _settle_cleanup(cleanup)
                raise
    return episode


async def discard_episode(episode: WireEpisode, *, timeout: float = 10):
    """Release an unused returned episode without downloading its training arrays."""
    references = [trace.info[PAYLOAD_INFO_KEY] for trace in episode.traces if PAYLOAD_INFO_KEY in trace.info]
    if not references:
        return
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        cleanup = asyncio.ensure_future(asyncio.gather(*(_delete(client, ref) for ref in references)))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await _settle_cleanup(cleanup)
            raise
    for trace in episode.traces:
        trace.info.pop(PAYLOAD_INFO_KEY, None)
