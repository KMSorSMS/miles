"""Client-side configuration and owned CPU-service lifecycle."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import tomllib
from pydantic import BaseModel, ConfigDict, Field, model_validator
from verifiers.v1.configs.client import TrainClientConfig
from verifiers.v1.configs.select import SelectConfig
from verifiers.v1.utils.loaders import resolve_env_config


class RemoteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    addresses: list[str] = Field(default_factory=list)
    server_config: Path | None = None
    """Optionally launch one Miles-owned local pool; external pools use addresses."""
    request_timeout: float = Field(3600, gt=0)
    cancel_timeout: float = Field(60, gt=0)
    startup_timeout: float = Field(180, gt=0)
    episodes_per_prompt: int | None = Field(None, ge=1)
    eval_episodes_per_prompt: int | None = Field(None, ge=1)
    """Unset: one independent episode per sample. Set 1 for an env that generates
    the complete n_samples_per_prompt group internally (e.g. CodeGolf attempts)."""
    max_task_attempts: int = Field(1000, ge=1)
    """Bound failed/filtered groups per training batch, avoiding infinite retry."""

    @model_validator(mode="after")
    def check_addresses(self):
        if not self.addresses and self.server_config is None:
            raise ValueError("Set remote.addresses or remote.server_config")
        if len(set(self.addresses)) != len(self.addresses):
            raise ValueError("Duplicate remote.addresses would distort server load accounting")
        return self


class AdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env: dict
    select: SelectConfig = Field(default_factory=SelectConfig)
    client: TrainClientConfig = Field(
        default_factory=lambda: TrainClientConfig(backend="sglang", api_key_var="SGLANG_API_KEY")
    )
    remote: RemoteConfig

    @model_validator(mode="after")
    def check_client(self):
        if self.client.backend != "sglang":
            raise ValueError("The Miles adapter requires client.backend='sglang'")
        return self


def load_config_data(path: str) -> dict:
    config_path = Path(path)
    if config_path.suffix.lower() != ".toml":
        raise ValueError("VERIFIERS_CONFIG must point to a TOML file")
    return tomllib.loads(config_path.read_text())


def load_adapter_config(path: str):
    config = AdapterConfig.model_validate(load_config_data(path))
    if config.remote.server_config is not None:
        config.remote.server_config = (Path(path).resolve().parent / config.remote.server_config).resolve()
    return config, resolve_env_config(config.env)


class ManagedServer:
    """An optional local CPU pool, with process-group cleanup owned by this client."""

    def __init__(self, config: RemoteConfig):
        self.config = config
        self.process: asyncio.subprocess.Process | None = None
        self._temporary: tempfile.TemporaryDirectory | None = None
        self._closing: asyncio.Task | None = None

    async def start(self) -> str:
        if self.config.server_config is None:
            raise ValueError("No managed server config")
        from miles.env_server.config import EnvServerConfig

        path = self.config.server_config
        raw = json.loads(path.read_text()) if path.suffix == ".json" else tomllib.loads(path.read_text())
        server = EnvServerConfig.model_validate(raw)
        self._temporary = tempfile.TemporaryDirectory(prefix="miles-env-")
        directory = Path(self._temporary.name)
        server.address_file = directory / "address.json"
        launch_config = directory / "server.json"
        launch_config.write_text(server.model_dump_json())
        try:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "miles.env_server",
                "--config",
                str(launch_config),
                start_new_session=True,
            )
            async with asyncio.timeout(self.config.startup_timeout):
                while not server.address_file.exists():
                    if self.process.returncode is not None:
                        raise RuntimeError(f"Environment server exited with code {self.process.returncode}")
                    await asyncio.sleep(0.05)
                content = server.address_file.read_text().strip()
                # The server address file is text; JSON is accepted for older
                # independently deployed launchers that publish an address object.
                return json.loads(content)["address"] if content.startswith("{") else content
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        # A second cancellation must not orphan the process group or remove its
        # configuration while workers are still shutting down.
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        closing = self._closing
        cancelled = False
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
        try:
            closing.result()
        finally:
            if self._closing is closing:
                self._closing = None
        if cancelled:
            raise asyncio.CancelledError

    async def _close(self) -> None:
        import contextlib
        import signal

        try:
            if self.process is not None and self.process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=self.config.cancel_timeout)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(self.process.pid, signal.SIGKILL)
                    await self.process.wait()
        finally:
            self.process = None
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None
