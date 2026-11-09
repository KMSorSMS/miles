"""CPU contracts for the complete adapter, including real env-server RPC.

Only GPU-only dumper/router hooks and post-generation prefill are substituted;
the dataset selection, config serialization, scheduling, wire transport, trace
conversion, evaluation, and checkpoints use their production implementations.
"""

import asyncio
import contextlib
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

vf = pytest.importorskip("verifiers.v1")
if "backend" not in vf.TrainClientConfig.model_fields:
    pytest.skip("Requires the native SGLang SDK integration", allow_module_level=True)
pytest.importorskip("renderers.backends")

from examples.experimental.verifiers.verifiers_config import ManagedServer, RemoteConfig
from examples.experimental.verifiers.verifiers_rollout import VerifiersRolloutFn

from miles.env_server.client import EnvDrainError
from miles.env_server.config import EnvServerConfig
from miles.env_server.pool import EnvServerPool
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnTrainInput


def make_args(**changes):
    values = dict(
        rollout_global_dataset=False,
        partial_rollout=False,
        chat_template_path=None,
        use_opd=False,
        use_output_store=False,
        hf_checkpoint="base",
        rollout_temperature=0.8,
        rollout_top_p=0.9,
        rollout_top_k=20,
        rollout_max_response_len=16,
        rollout_max_context_len=32,
        rollout_max_prompt_len=16,
        apply_chat_template_kwargs={},
        rollout_top_logprobs_num=0,
        rollout_skip_special_tokens=False,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        sglang_server_concurrency=2,
        inference_runtime_mut_state=SimpleNamespace(engine_count=1),
        dynamic_sampling_filter_path=None,
        rollout_shuffle=False,
        rollout_seed=7,
        eval_temperature=0.2,
        eval_top_p=None,
        eval_top_k=None,
        eval_max_response_len=8,
        eval_max_context_len=None,
        eval_max_prompt_len=12,
        eval_reward_key=None,
        reward_key=None,
        eval_min_new_tokens=0,
        use_sampling_support_replay=False,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        sglang_model_routers=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        lora_rank=0,
        lora_adapter_path=None,
        lora_train_only=False,
        namespaced_radix_cache=True,
        group_rm=False,
        custom_rm_path=None,
        rm_type=None,
        rollout_batch_size=1,
        n_samples_per_prompt=2,
        n_samples_per_eval_prompt=2,
        over_sampling_batch_size=1,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
        sglang=SimpleNamespace(
            models=[],
            common_value=lambda key: {
                "enable_deterministic_inference": True,
                "tokenizer_path": "base-tokenizer",
            }[key],
        ),
    )
    return Namespace(**(values | changes))


@pytest.fixture
def adapter_factory(tmp_path, monkeypatch):
    fixture_dir = Path(__file__).resolve().parents[3] / "env_server"
    monkeypatch.syspath_prepend(str(fixture_dir))
    import miles_env_fixture as fixture

    # The parent loads the dataset. Fresh spawned workers retain the fixture's
    # assertion that dataset loading must never happen on the server.
    monkeypatch.setattr(
        fixture.Taskset,
        "load",
        lambda self: (fixture.Task(fixture.Data(prompt=f"question-{i}", value=10 + i)) for i in range(3)),
    )
    backend = SimpleNamespace(features=SimpleNamespace(routing_key_header=None), routes_by_key=lambda _: False)
    router = ModuleType("miles.router.backend")
    router.make_router_backend = lambda _: backend
    monkeypatch.setitem(sys.modules, "miles.router.backend", router)
    dumper = ModuleType("miles.utils.dumper_utils")

    async def configure(_):
        pass

    dumper.configure_sglang = configure
    monkeypatch.setitem(sys.modules, "miles.utils.dumper_utils", dumper)
    prompt = tmp_path / "system.txt"
    prompt.write_text("Taskset-level system prompt")

    def create(address="tcp://127.0.0.1:65530", **args):
        config = tmp_path / "adapter.toml"
        config.write_text(
            f"""[env]
attempts = 2
[env.taskset]
id = "miles_env_fixture"
system_prompt = "{prompt}"
[select.include]
idx = ["1:3"]
[remote]
addresses = ["{address}"]
episodes_per_prompt = 1
eval_episodes_per_prompt = 1
startup_timeout = 15
cancel_timeout = 5
"""
        )
        monkeypatch.setenv("VERIFIERS_CONFIG", str(config))
        adapter = VerifiersRolloutFn(RolloutFnConstructorInput(args=make_args(**args), data_source=None))

        async def postprocess(data, all_data):
            for group in data:
                for sample in group:
                    sample.validate()

        monkeypatch.setattr(adapter, "_postprocess_train_samples", postprocess)
        return adapter

    return create


def test_constructor_selection_client_config_and_checkpoint(adapter_factory, tmp_path):
    adapter = adapter_factory(use_rollout_indexer_replay=True)
    assert [task.data.idx for task in adapter._tasks] == [1, 2]
    assert [task.data.system_prompt for task in adapter._tasks] == ["Taskset-level system prompt"] * 2
    adapter.config.client.backend_options["indexer_topk_shape"] = [1, 3]
    client = adapter._client_config(adapter.args)
    restored = vf.TrainClientConfig.model_validate_json(client.model_dump_json())
    assert restored.backend_options["indexer_topk_shape"] == [1, 3]
    assert restored.backend_options["abort_strategy"] == "workers"
    assert restored.renderer_model_name == "base-tokenizer"
    assert restored.base_url == "http://127.0.0.1:30000/v1"
    assert restored.backend_options["sampling_constraints"]["top_k"] == 20
    evaluation = adapter._client_config(adapter.eval_args)
    assert not evaluation.backend_options["return_indexer_topk"]
    assert evaluation.backend_options["sampling_constraints"] == {}
    assert evaluation.backend_options["max_response_tokens"] == 8
    adapter._next_train_task_idx, adapter._next_sample_index, adapter._next_seed = 1, 2, 9
    checkpoint = tmp_path / "checkpoint"
    adapter.save(checkpoint)
    resumed = adapter_factory()
    resumed.load(checkpoint)
    assert (resumed._next_train_task_idx, resumed._next_sample_index, resumed._next_seed) == (1, 2, 9)
    resumed._task_fingerprint = "different tasks"
    with pytest.raises(ValueError, match="different taskset"):
        resumed.load(checkpoint)


def test_explicit_router_url_keeps_worker_abort_unless_direct_engine_is_selected(adapter_factory):
    adapter = adapter_factory()
    adapter.config.client = adapter.config.client.model_copy(update={"base_url": "http://cpu-visible-router/v1"})
    client = adapter._client_config(adapter.args)
    assert client.base_url == "http://cpu-visible-router/v1"
    assert client.backend_options["abort_strategy"] == "workers"
    adapter.config.client.backend_options["abort_strategy"] = "endpoint"
    assert adapter._client_config(adapter.args).backend_options["abort_strategy"] == "endpoint"


@pytest.mark.asyncio
async def test_environment_rewards_do_not_import_optional_miles_graders(adapter_factory, monkeypatch):
    adapter = adapter_factory()
    monkeypatch.setitem(sys.modules, "miles.rollout.rm_hub", None)
    sample = SimpleNamespace(reward=0.5)
    await adapter._apply_miles_rewards([sample])
    assert sample.reward == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize("group_rm", [False, True])
async def test_miles_rewards_score_each_logical_rollout_once(adapter_factory, monkeypatch, group_rm):
    adapter = adapter_factory(group_rm=group_rm, rm_type=None if group_rm else "test")
    scored = []
    graders = ModuleType("miles.rollout.rm_hub")

    async def one(args, sample):
        scored.append(sample.index)
        return sample.rollout_id / 100

    async def batch(args, samples):
        return [await one(args, sample) for sample in samples]

    graders.async_rm, graders.batched_async_rm = one, batch
    monkeypatch.setitem(sys.modules, "miles.rollout.rm_hub", graders)
    samples = [
        SimpleNamespace(index=0, rollout_id=17, reward=None),
        SimpleNamespace(index=1, rollout_id=17, reward=None),
        SimpleNamespace(index=2, rollout_id=23, reward=None),
    ]
    await adapter._apply_miles_rewards([samples[:2], samples[2]])
    assert scored == [0, 2]
    assert [sample.reward for sample in samples] == [0.17, 0.17, 0.23]


@pytest.mark.asyncio
async def test_adapter_training_and_eval_through_real_cpu_workers(adapter_factory, tmp_path):
    adapter = adapter_factory()
    server = EnvServerPool(
        EnvServerConfig(
            env=adapter.env_config.model_dump(mode="json", serialize_as_any=True),
            serve={"pool": {"type": "static", "num_workers": 2}, "shutdown_timeout": 3},
        )
    )
    assert server.env_config == adapter.env_config.model_dump(mode="json", serialize_as_any=True)
    adapter.config.remote.addresses = [server.address]
    serving = asyncio.create_task(server.run())
    try:
        result = await adapter(RolloutFnTrainInput(rollout_id=0))
        assert len(result.samples) == 1 and len(result.samples[0]) == 2
        assert [sample.tokens for sample in result.samples[0]] == [[11, 11], [11, 11]]
        assert [sample.reward for sample in result.samples[0]] == [0.5, 0.5]
        assert [sample.metadata["verifiers"]["task_index"] for sample in result.samples[0]] == [1, 1]
        assert not adapter._active
        checkpoint = tmp_path / "checkpoint"
        adapter.save(checkpoint)
        resumed = adapter_factory(server.address)
        resumed.load(checkpoint)
        try:
            second = await resumed(RolloutFnTrainInput(rollout_id=1))
            assert [sample.tokens for sample in second.samples[0]] == [[11, 12], [11, 12]]
            assert [sample.index for sample in second.samples[0]] == [2, 3]
            evaluation = await resumed(RolloutFnEvalInput(rollout_id=1))
            data = evaluation.data[adapter.env_config.env_id]
            assert data["rewards"] == [0.5] * 4
            assert data["truncated"] == [False] * 4
            assert len(data["samples"]) == 4
            assert not resumed._active
        finally:
            await resumed.dispose()
    finally:
        await adapter.dispose()
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serving


@pytest.mark.asyncio
async def test_training_drains_other_groups_before_reporting_failure(adapter_factory, tmp_path):
    adapter = adapter_factory(rollout_batch_size=2)
    entered = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_finish = asyncio.Event()
    calls = 0

    async def remote_run(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            await entered.wait()
            raise RuntimeError("remote task failed")
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            cleanup_started.set()
            await cleanup_finish.wait()
            raise EnvDrainError("engine abort unconfirmed") from error

    async def close():
        pass

    adapter._remote = SimpleNamespace(run=remote_run, close=close)
    running = asyncio.create_task(adapter(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.wait_for(cleanup_started.wait(), 3)
    for _ in range(2):
        running.cancel()
        await asyncio.sleep(0)
    assert not running.done()
    cleanup_finish.set()
    with pytest.raises(EnvDrainError, match="engine abort unconfirmed"):
        await asyncio.wait_for(running, 3)
    assert not adapter._active
    adapter.save(tmp_path / "checkpoint")
    await adapter.dispose()


@pytest.mark.asyncio
async def test_cleanup_failure_takes_priority_over_ordinary_rollout_failure(adapter_factory):
    adapter = adapter_factory()
    loop = asyncio.get_running_loop()
    ordinary, unsafe = loop.create_future(), loop.create_future()
    ordinary.set_exception(RuntimeError("ordinary task failure"))
    unsafe.set_exception(EnvDrainError("engine still running"))
    with pytest.raises(EnvDrainError, match="engine still running"):
        await adapter._cancel_pending([ordinary, unsafe])


@pytest.mark.asyncio
async def test_managed_server_close_survives_repeated_cancellation(tmp_path):
    """Neither repeated cancellation nor a concurrent disposer can orphan a
    process or delete its launch directory before graceful shutdown completes."""
    marker, gate = tmp_path / "terminating", tmp_path / "finish"
    script = """import signal, sys, time
from pathlib import Path
marker, gate = map(Path, sys.argv[1:])
signal.signal(signal.SIGTERM, lambda *_: marker.write_text('terminating'))
print('ready', flush=True)
while not gate.exists():
    time.sleep(.01)
"""
    server = ManagedServer(RemoteConfig(addresses=["tcp://unused:1"], cancel_timeout=5))
    server._temporary = tempfile.TemporaryDirectory(prefix="miles-close-test-")
    directory = Path(server._temporary.name)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        str(marker),
        str(gate),
        stdout=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    server.process = process
    try:
        assert await asyncio.wait_for(process.stdout.readline(), 3) == b"ready\n"
        first = asyncio.create_task(server.close())
        async with asyncio.timeout(3):
            while not marker.exists():
                await asyncio.sleep(0.01)
        second = asyncio.create_task(server.close())
        for _ in range(2):
            first.cancel()
            await asyncio.sleep(0)
        assert not first.done() and not second.done()
        assert directory.exists() and server.process is process
        gate.touch()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, 3)
        await asyncio.wait_for(second, 3)
        assert process.returncode == 0
        assert server.process is None and not directory.exists()
    finally:
        gate.touch()
        if process.returncode is None:
            process.kill()
        await process.wait()
        await server.close()
