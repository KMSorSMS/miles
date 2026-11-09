"""The adapter reads data/config without constructing a local Environment."""

import pytest

vf = pytest.importorskip("verifiers.v1")
if not hasattr(vf, "TrainClientConfig") or "backend" not in vf.TrainClientConfig.model_fields:
    pytest.skip("Requires the native-SGLang Verifiers SDK", allow_module_level=True)

from examples.experimental.verifiers.verifiers_config import load_adapter_config
from examples.experimental.verifiers.verifiers_rollout import _import_verifiers


def test_backend_configuration_roundtrips_and_task_config_narrows(tmp_path):
    path = tmp_path / "adapter.toml"
    path.write_text(
        """[env]
attempts = 3
[env.taskset]
id = "code-golf"
[client]
backend = "sglang"
renderer_model_name = "Qwen/Qwen3-0.6B"
[remote]
addresses = ["tcp://cpu-a:5555", "tcp://cpu-b:5555"]
episodes_per_prompt = 1
"""
    )
    config, env = load_adapter_config(str(path))
    assert env.attempts == 3
    assert config.remote.episodes_per_prompt == 1
    vf = _import_verifiers()
    restored = vf.TrainClientConfig.model_validate(config.client.model_dump())
    assert restored.backend == "sglang"


def test_remote_config_requires_a_destination_and_rejects_unknown_fields(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[env]\n[remote]\n")
    with pytest.raises(ValueError, match="remote.addresses"):
        load_adapter_config(str(path))
    path.write_text('[env]\n[remote]\naddresses=["tcp://cpu:55"]\nignored=1\n')
    with pytest.raises(ValueError, match="ignored"):
        load_adapter_config(str(path))
