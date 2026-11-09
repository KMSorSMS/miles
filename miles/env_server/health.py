"""Small readiness probe: no verifiers, renderer, trainer, or model imports."""

import asyncio
import uuid

import msgpack
import zmq
import zmq.asyncio

from miles.env_server.protocol import PROTOCOL_VERSION


async def ready(address: str, *, timeout: float = 2) -> bool:
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(address)
    request_id = uuid.uuid4().hex.encode()
    try:
        async with asyncio.timeout(timeout):
            await socket.send_multipart([request_id, b"health", msgpack.packb({})])
            frames = await socket.recv_multipart()
        if len(frames) != 3 or frames[:2] != [request_id, b"reply"]:
            return False
        response = msgpack.unpackb(frames[2], raw=False)
        return (
            isinstance(response, dict)
            and response.get("protocol_version") == PROTOCOL_VERSION
            and response.get("success") is True
            and response.get("ready_workers", 0) > 0
        )
    except (TimeoutError, zmq.ZMQError, ValueError, TypeError):
        return False
    finally:
        socket.close()
        context.term()
