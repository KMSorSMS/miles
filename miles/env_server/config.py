"""CPU environment serving configuration; independent of the training runtime."""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StaticPoolConfig(Config):
    type: Literal["static"] = "static"
    num_workers: int = Field(4, ge=1)


class ElasticPoolConfig(Config):
    """Scale up at 90% of capacity, with an explicit finite process limit."""

    type: Literal["elastic"] = "elastic"
    max_workers: int = Field(16, ge=1)
    multiplex: int = Field(128, ge=1)


PoolConfig = Annotated[StaticPoolConfig | ElasticPoolConfig, Field(discriminator="type")]


class PayloadConfig(Config):
    bind_host: str = "127.0.0.1"
    advertised_host: str | None = None
    port: int = Field(0, ge=0, le=65535)
    ttl_seconds: float = Field(3600, gt=0)
    max_bytes: int = Field(4 << 30, ge=1)

    @model_validator(mode="after")
    def validate_host(self):
        if self.bind_host in {"0.0.0.0", "::", "*"} and not self.advertised_host:
            raise ValueError("A wildcard payload bind requires advertised_host reachable by Miles clients")
        return self


class ServeConfig(Config):
    address: str = "tcp://127.0.0.1:0"
    pool: PoolConfig = Field(default_factory=StaticPoolConfig)
    max_concurrent: int = Field(128, ge=1)
    startup_timeout: float = Field(120, gt=0)
    shutdown_timeout: float = Field(30, gt=0)
    restart_workers: bool = True
    max_worker_restarts: int = Field(3, ge=0)
    payload: PayloadConfig = Field(default_factory=PayloadConfig)

    @model_validator(mode="after")
    def validate_payload_port(self):
        count = self.pool.num_workers if isinstance(self.pool, StaticPoolConfig) else self.pool.max_workers
        if self.payload.port and count > 1:
            raise ValueError("Multiple workers require payload.port=0 (each worker binds its own port)")
        return self


class EnvServerConfig(Config):
    """JSON/TOML: [env] is a verifiers EnvConfig; [serve] configures CPU workers."""

    env: dict
    serve: ServeConfig = Field(default_factory=ServeConfig)
    address_file: Path | None = None
    log_level: str = "INFO"
    run_id: str | None = None
    sandbox_labels: list[str] = Field(default_factory=list)


def _canonical_numbers(value):
    # Pydantic may leave a default float as an int until a JSON roundtrip. Those
    # configurations are semantically identical; booleans remain distinct.
    if type(value) is float and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _canonical_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical_numbers(item) for item in value]
    return value


def config_fingerprint(config: dict) -> str:
    canonical = _canonical_numbers(config)
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
