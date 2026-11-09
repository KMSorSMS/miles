"""Least-in-flight routing across independently deployed CPU env servers."""

import asyncio
import time

from miles.env_server.client import EnvClient


class EnvClientPool:
    def __init__(self, addresses: list[str], *, timeout: float = 3600, cancel_timeout: float = 30):
        if not addresses or len(set(addresses)) != len(addresses):
            raise ValueError("Provide at least one distinct environment server address")
        self.clients = [EnvClient(address, timeout=timeout, cancel_timeout=cancel_timeout) for address in addresses]
        self._in_flight = [0] * len(self.clients)
        self._available = [True] * len(self.clients)
        self._cursor = 0
        self._closing = False
        self._fingerprint: str | None = None
        self._probe_lock = asyncio.Lock()
        self._last_probe = 0.0

    async def describe(self, timeout: float = 2):
        descriptions = await asyncio.gather(*(client.describe(timeout) for client in self.clients))
        fingerprints = {item.config_fingerprint for item in descriptions}
        if len(fingerprints) != 1:
            raise ValueError("Environment replicas have different resolved configurations")
        self._fingerprint = descriptions[0].config_fingerprint
        self._last_probe = time.monotonic()
        self._available = [item.success and item.ready_workers > 0 for item in descriptions]
        return descriptions[0].model_copy(
            update={
                "success": all(self._available),
                "ready_workers": sum(item.ready_workers for item in descriptions),
                "total_workers": sum(item.total_workers for item in descriptions),
            }
        )

    async def health(self, timeout: float = 2):
        async with self._probe_lock:
            descriptions = await asyncio.gather(
                *(client.describe(timeout) for client in self.clients), return_exceptions=True
            )
            self._last_probe = time.monotonic()
            fingerprints = {item.config_fingerprint for item in descriptions if not isinstance(item, BaseException)}
            if self._fingerprint is not None:
                fingerprints.add(self._fingerprint)
            if len(fingerprints) > 1:
                raise ValueError("Environment replicas have different resolved configurations")
            if fingerprints:
                self._fingerprint = next(iter(fingerprints))
            self._available = [
                not isinstance(item, BaseException) and item.success and item.ready_workers > 0
                for item in descriptions
            ]
            return all(self._available)

    async def wait_for_server_startup(self, timeout: float = 120, interval: float = 0.2):
        await asyncio.gather(*(client.wait_for_server_startup(timeout, interval) for client in self.clients))
        await self.describe()

    async def run(self, client, model, sampling, task_data, on_delta=None, *, timeout=None):
        if self._closing:
            raise RuntimeError("Environment client pool is closing")
        if not all(self._available) and time.monotonic() - self._last_probe >= 2:
            await self.health()
        candidates = [i for i, available in enumerate(self._available) if available]
        if not candidates:
            # Probe again so a previously unavailable replica can recover.
            await self.health()
            candidates = [i for i, available in enumerate(self._available) if available]
        if not candidates:
            raise ConnectionError("No healthy environment replicas")
        index = min(candidates, key=lambda i: (self._in_flight[i], (i - self._cursor) % len(self.clients)))
        self._cursor = (index + 1) % len(self.clients)
        self._in_flight[index] += 1
        try:
            return await self.clients[index].run(client, model, sampling, task_data, on_delta, timeout=timeout)
        except (TimeoutError, ConnectionError):
            self._available[index] = False
            # Never replay automatically: task tools/rewards may have side effects.
            raise
        finally:
            self._in_flight[index] -= 1

    async def close(self):
        self._closing = True
        results = await asyncio.gather(*(client.close() for client in self.clients), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
