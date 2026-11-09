"""Typed messages for the Miles env-server wire (DEALER/ROUTER + msgpack)."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field
from verifiers.v1.configs.client import ClientConfig
from verifiers.v1.configs.sampling import SamplingConfig

from miles.env_server.delta import TraceSummary
from miles.env_server.protocol import PROTOCOL_VERSION


class BaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: ClassVar[str]


class BaseResponse(BaseModel):
    success: bool = True
    error: str | None = None
    error_code: str | None = None


class HealthRequest(BaseRequest):
    method: ClassVar[str] = "health"


class HealthResponse(BaseResponse):
    protocol_version: int = PROTOCOL_VERSION
    ready_workers: int = 1
    total_workers: int = 1
    env_config: dict = Field(default_factory=dict)
    config_fingerprint: str = ""


class CancelRequest(BaseRequest):
    """The reply acknowledges terminal cleanup, not merely task.cancel()."""

    method: ClassVar[str] = "cancel"
    request_id: str


class CancelResponse(BaseResponse):
    cancelled: bool = False
    drained: bool = True


class RunRequest(BaseRequest):
    method: ClassVar[str] = "run"
    task_data: dict
    client: ClientConfig
    model: str
    sampling: SamplingConfig


class RunResponse(BaseResponse):
    head: dict | None = None
    traces: list[TraceSummary] = Field(default_factory=list)
