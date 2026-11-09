"""Remote adapter invariants: exact training evidence and complete episode scoring."""

from argparse import Namespace

import numpy as np
import pytest

vf = pytest.importorskip("verifiers.v1")
if not hasattr(vf, "TrainClientConfig") or "backend" not in vf.TrainClientConfig.model_fields:
    pytest.skip("Requires the native-SGLang Verifiers SDK", allow_module_level=True)

from examples.experimental.verifiers.verifiers_rollout import (
    VerifiersRolloutFn,
    _config_path,
    _load_config_data,
    _raise_for_fatal_episode_errors,
    _trace_eval_reward,
    _validate_args,
)
from examples.experimental.verifiers.verifiers_samples import trace_to_samples
from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import Error
from verifiers.v1.types import SamplingMask, TopLogprobs, WeightVersionSpan

from miles.utils.types import Sample


def sample_args(**overrides):
    return Namespace(
        **(
            dict(
                reward_key=None,
                use_sampling_support_replay=False,
                rollout_top_logprobs_num=0,
                use_rollout_routing_replay=False,
                use_rollout_indexer_replay=False,
            )
            | overrides
        )
    )


def make_trace(*, branches=False, arrays=False):
    prefix = MessageNode(message=vf.UserMessage(content="question"), token_ids=[10], mask=[False])
    answer = MessageNode(
        parent=0,
        message=vf.AssistantMessage(content="A"),
        sampled=True,
        token_ids=[20, 21],
        mask=[False, True],
        logprobs=[-0.125],
    )
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=3, prompt="question")),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[prefix, answer],
        ok=True,
        is_completed=True,
    )
    trace.record_reward("score", 2, 0.5)
    if branches:
        trace.nodes.extend(
            [
                MessageNode(
                    parent=1,
                    message=vf.AssistantMessage(content="B"),
                    sampled=True,
                    token_ids=[22],
                    mask=[True],
                    logprobs=[-0.25],
                ),
                MessageNode(
                    parent=1,
                    message=vf.AssistantMessage(content="C"),
                    sampled=True,
                    token_ids=[23],
                    mask=[True],
                    logprobs=[-0.5],
                ),
            ]
        )
    if arrays:
        for node in trace.nodes:
            size = len(node.token_ids)
            node.routed_experts = np.arange(size * 2, dtype=np.uint16).reshape(size, 1, 2) + 300
            node.indexer_topk = np.arange(size * 2, dtype=np.int32).reshape(size, 1, 2) + 1000
        answer.sampling_mask = SamplingMask(ids=np.array([21, 7], np.int32), counts=np.array([2], np.int32))
        answer.top_logprobs = TopLogprobs(
            ids=np.array([21, 7], np.int32),
            counts=np.array([2], np.int32),
            logprobs=np.array([-0.125, -2], np.float32),
        )
    return trace


def test_wire_trace_preserves_training_arrays():
    trace = make_trace(arrays=True)
    episode = vf.Episode(task=trace.task, traces=[trace], ok=True)
    wire = vf.WireEpisode.model_validate(episode.model_dump())
    args = sample_args(
        use_sampling_support_replay=True,
        rollout_top_logprobs_num=2,
        use_rollout_routing_replay=True,
        use_rollout_indexer_replay=True,
        reward_key="score",
    )
    sample = trace_to_samples(args, wire.traces[0], group_index=4, index_start=9)[0]
    assert sample.tokens == [10, 20, 21]
    assert sample.rollout_log_probs == [-0.125]
    assert sample.loss_mask == [1] and sample.reward == {"score": 1.0, "reward": 1.0}
    assert sample.rollout_routed_experts.tolist() == [[[300, 301]], [[300, 301]]]
    assert sample.rollout_indexer_topk.dtype == np.int32
    assert len(sample.rollout_indexer_topk) == len(sample.tokens) - 1
    assert sample.rollout_topk_token_ids.tolist() == [[21, 7]]
    assert len(sample.rollout_sampling_mask) == 1
    sample.validate()


@pytest.mark.asyncio
@pytest.mark.parametrize("passed", [True, False])
async def test_episode_finalizes_joint_rewards(passed):
    # The real CodeGolf finalizer compares siblings in ONE complete episode.
    from code_golf.taskset import CodeGolfEnv, CodeGolfEnvConfig

    traces = [make_trace(), make_trace()]
    traces[0].nodes[-1].message.content = "print(1)"
    traces[1].nodes[-1].message.content = "print(1 + 0)"
    for i, trace in enumerate(traces):
        trace.metrics.update(passed=float(passed), latency=float(i + 1))
    episode = vf.Episode(task=traces[0].task, traces=traces, ok=True)
    env = object.__new__(CodeGolfEnv)
    env.config = CodeGolfEnvConfig(taskset={"id": "code-golf"}, attempts=2)
    await env.finalize(None, episode)
    restored = vf.WireEpisode.model_validate(episode.model_dump())
    assert [trace.reward for trace in restored.traces] == ([2, 1] if passed else [1, 1])
    # The adapter keeps the jointly finalized reward instead of recomputing it.
    assert [
        trace_to_samples(sample_args(), t, group_index=0, index_start=i)[0].reward
        for i, t in enumerate(restored.traces)
    ] == [t.reward for t in restored.traces]


def test_shared_prefix_trains_once_and_siblings_share_rollout_identity():
    samples = trace_to_samples(sample_args(), make_trace(branches=True), group_index=2, index_start=10, rollout_id=7)
    assert len(samples) == 2
    assert samples[0].tokens == [10, 20, 21, 22]
    assert samples[1].tokens == [10, 20, 21, 23]
    assert samples[0].loss_mask == [1, 1] and samples[1].loss_mask == [1]
    assert samples[1].response_length == 1
    assert [s.rollout_id for s in samples] == [7, 7]
    assert [s.lineage.output_index for s in samples] == [0, 1]
    assert all(s.lineage.output_count == 2 for s in samples)


def test_per_call_prefill_versions_survive_wire_and_sample_conversion():
    trace = make_trace(branches=True)
    node = trace.nodes[1]
    node.weight_versions = [WeightVersionSpan(version="v2", start=0, end=1)]
    node.prefill_weight_versions = [WeightVersionSpan(version="v1", start=0, end=2)]
    node.prompt_tokens = 2
    episode = vf.WireEpisode.model_validate(vf.Episode(task=trace.task, traces=[trace]).model_dump())
    samples = trace_to_samples(sample_args(), episode.traces[0], group_index=0, index_start=0)
    for sample in samples:
        call = sample.weight_versions[0]
        assert call.output_start == 2 and call.prompt_tokens == 2
        assert [(s.version, s.abs_start, s.abs_end) for s in call.spans] == [("v2", 2, 3)]
        assert [(s.version, s.abs_start, s.abs_end) for s in call.prefill_spans] == [("v1", 0, 2)]


def test_adapter_rejects_image_trace_without_token_attribution():
    trace = make_trace()
    trace.nodes[0].message = vf.UserMessage(
        content=[{"type": "image_url", "image_url": {"url": "https://example.test/image.png"}}]
    )
    adapter = object.__new__(VerifiersRolloutFn)
    adapter.args = sample_args()
    adapter._next_sample_index = adapter._next_rollout_index = 0
    adapter._processor = None
    with pytest.raises(ValueError, match="multimodal token attribution"):
        adapter._convert_group([vf.Episode(task=trace.task, traces=[trace])], group_index=0)


def test_bad_logprobs_and_missing_replay_cannot_silently_train():
    trace = make_trace()
    trace.nodes[-1].logprobs = np.array([], np.float32)
    with pytest.raises(ValueError, match="missing sampled-token"):
        trace_to_samples(sample_args(), trace, group_index=0, index_start=0)
    with pytest.raises(ValueError, match="incomplete"):
        trace_to_samples(sample_args(use_rollout_routing_replay=True), make_trace(), group_index=0, index_start=0)


def test_nontrainable_or_dropped_trace_is_not_a_training_sample():
    trace = make_trace()
    trace.agent.trainable = False
    assert trace_to_samples(sample_args(), trace, group_index=0, index_start=0) == []
    trace.agent.trainable, trace.dropped = True, "excluded"
    assert trace_to_samples(sample_args(), trace, group_index=0, index_start=0) == []


def test_failed_trace_and_named_rewards():
    trace = make_trace()
    trace.ok = False
    sample = trace_to_samples(sample_args(), trace, group_index=0, index_start=0)[0]
    assert sample.status is Sample.Status.FAILED
    assert _trace_eval_reward(trace, "missing") is None
    trace.ok = True
    with pytest.raises(KeyError):
        _trace_eval_reward(trace, "missing")


def test_recovered_retry_history_does_not_invalidate_success():
    trace = make_trace()
    trace.errors = [Error(type="ProviderError", message="previous attempt", status_code=400)]
    episode = vf.Episode(task=trace.task, traces=[trace], ok=True, errors=list(trace.errors))
    _raise_for_fatal_episode_errors([episode])
    episode.ok = False
    with pytest.raises(RuntimeError, match="protocol error"):
        _raise_for_fatal_episode_errors([episode])
    episode.errors = [Error(type="ProviderError", message="timeout", status_code=408)]
    _raise_for_fatal_episode_errors([episode])
    episode.ok = True
    episode.errors = [Error(type="InferenceCleanupError", message="engine still running", status_code=503)]
    with pytest.raises(RuntimeError, match="engine still running"):
        _raise_for_fatal_episode_errors([episode])


def test_config_and_validation(tmp_path, monkeypatch):
    monkeypatch.delenv("VERIFIERS_CONFIG", raising=False)
    with pytest.raises(ValueError, match="VERIFIERS_CONFIG"):
        _config_path()
    path = tmp_path / "remote.toml"
    path.write_text('[remote]\naddresses=["tcp://cpu:5000"]\n')
    monkeypatch.setenv("VERIFIERS_CONFIG", str(path))
    assert _load_config_data(_config_path())["remote"]["addresses"] == ["tcp://cpu:5000"]
    args = Namespace(
        rollout_global_dataset=False,
        partial_rollout=False,
        chat_template_path=None,
        use_opd=False,
        use_output_store=False,
    )
    _validate_args(args)
    args.partial_rollout = True
    with pytest.raises(ValueError, match="partial-rollout"):
        _validate_args(args)


def test_checkpoint_resumes_exact_cursor_ids_and_seed(tmp_path):
    adapter = object.__new__(VerifiersRolloutFn)
    adapter._active = set()
    adapter._task_fingerprint = "same data and shuffle"
    fields = ("_next_train_task_idx", "_next_group_index", "_next_sample_index", "_next_rollout_index", "_next_seed")
    for index, name in enumerate(fields):
        setattr(adapter, name, index + 7)
    adapter.save(tmp_path)
    for name in fields:
        setattr(adapter, name, 0)
    adapter.load(tmp_path)
    assert [getattr(adapter, name) for name in fields] == list(range(7, 12))
    adapter._task_fingerprint = "other"
    with pytest.raises(ValueError, match="different taskset"):
        adapter.load(tmp_path)
