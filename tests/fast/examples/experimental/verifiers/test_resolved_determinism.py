import asyncio
from types import SimpleNamespace

import pytest

vf = pytest.importorskip("verifiers.v1")
if not hasattr(vf, "TrainClientConfig") or "backend" not in vf.TrainClientConfig.model_fields:
    pytest.skip("Requires the native-SGLang Verifiers SDK", allow_module_level=True)

from examples.experimental.verifiers.verifiers_rollout import VerifiersRolloutFn


@pytest.mark.asyncio
@pytest.mark.parametrize("deterministic", [True, False])
async def test_remote_sampling_keeps_separate_episode_seed_ranges(monkeypatch, deterministic):
    adapter = object.__new__(VerifiersRolloutFn)
    adapter.args = SimpleNamespace(sglang=SimpleNamespace(common_value=lambda _: deterministic))
    adapter.eval_args = adapter.args
    adapter.model = "base"
    adapter.config = SimpleNamespace(remote=SimpleNamespace(episodes_per_prompt=2, eval_episodes_per_prompt=1))
    adapter.sampling = vf.SamplingConfig(temperature=0.8)
    adapter.eval_sampling = vf.SamplingConfig(temperature=0.2)
    sent = []

    async def ensure():
        pass

    async def run(**kwargs):
        sent.append(kwargs)
        return vf.Episode(task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="q")), ok=True)

    adapter._ensure_remote = ensure
    adapter._client_config = lambda _: vf.TrainClientConfig(backend="sglang")
    adapter._remote = SimpleNamespace(run=run)
    task = vf.Task(vf.TaskData(prompt="complete task", idx=37))
    episodes = await adapter._run_task_group(task, 3, asyncio.Semaphore(2), 100)
    assert len(episodes) == 2
    assert [request["task_data"]["prompt"] for request in sent] == ["complete task"] * 2
    assert [request["sampling"].model_dump().get("seed") for request in sent] == (
        [100, 103] if deterministic else [None, None]
    )
    assert adapter.sampling.model_dump().get("seed") is None
