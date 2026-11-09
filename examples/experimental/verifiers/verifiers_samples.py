"""Convert complete verifiers traces into Miles samples without re-tokenizing output.

Shared graph prefixes contribute loss once. An independent agent run is one Miles
rollout; its physical branches retain that rollout identity and their lineage.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any

import numpy as np
from pydantic import BaseModel

from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.types import Sample, SampleLineage, WeightVersionSpan, WeightVersionsPerCall


def _serialize_prompt(prompt):
    if isinstance(prompt, list):
        return [
            message.model_dump(mode="json", exclude_none=True) if isinstance(message, BaseModel) else message
            for message in prompt
        ]
    return prompt or ""


def _reward_values(trace) -> dict[str, float]:
    return {name: reward.value for name, reward in trace.rewards.items() if reward is not None}


def _trace_eval_reward(trace, reward_key: str | None):
    if reward_key is None:
        return trace.reward
    rewards = {**_reward_values(trace), "reward": trace.reward}
    return rewards.get(reward_key) if trace.has_error else rewards[reward_key]


def _sample_status(trace) -> Sample.Status:
    if trace.has_error:
        return Sample.Status.FAILED
    return Sample.Status.TRUNCATED if trace.is_truncated else Sample.Status.COMPLETED


def _trainable_branches(trace):
    trained_nodes: set[int] = set()
    if not trace.agent.trainable or trace.is_dropped:
        return
    for branch in trace.branches:
        if not branch.trainable:
            continue
        masks = []
        for node in branch.nodes:
            mask = np.asarray(node.mask, dtype=bool).copy()
            if mask.any():
                if len(node.logprobs) != int(mask.sum()):
                    raise ValueError(f"Trace {trace.id} has missing sampled-token logprobs")
                if id(node) in trained_nodes:
                    mask[:] = False
                else:
                    trained_nodes.add(id(node))
            masks.append(mask)
        mask = np.concatenate(masks) if masks else np.zeros(0, dtype=bool)
        if mask.any():
            yield branch, mask


def _sampling_support(branch, start: int, loss_mask: list[int]) -> RolloutSamplingMask | None:
    support = branch.sampling_mask
    if support is None:
        return None
    counts = np.asarray(support.counts)
    if len(counts) != len(branch.token_ids) or np.any(counts < 0):
        raise ValueError("Sampling support does not align with branch tokens")
    offsets = np.concatenate(([0], np.cumsum(counts)))
    if offsets[-1] != len(support.ids):
        raise ValueError("Sampling support counts do not cover its IDs")
    rows = []
    for index, trainable in enumerate(loss_mask, start):
        row = support.ids[offsets[index] : offsets[index + 1]].tolist()
        token = int(branch.token_ids[index])
        if trainable and (not row or token not in row):
            raise ValueError("A trainable token is missing its realized sampling support")
        # Miles requires a nonempty support on observation tokens too; these
        # positions have zero loss, so the singleton is only structural padding.
        rows.append(row or [token])
    return RolloutSamplingMask.from_mask_list(rows)


def _top_logprobs(branch, start: int, loss_mask: list[int], k: int):
    head = branch.top_logprobs
    if not k:
        return None, None
    if head is None:
        raise ValueError("Requested rollout top-k logprobs are missing from the trace")
    counts = np.asarray(head.counts)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    if len(counts) != len(branch.token_ids) or offsets[-1] != len(head.ids) or len(head.ids) != len(head.logprobs):
        raise ValueError("Top-k logprobs do not align with branch tokens")
    ids = np.full((len(loss_mask), k), -1, dtype=np.int32)
    logprobs = np.full((len(loss_mask), k), -np.inf, dtype=np.float32)
    for row, index in enumerate(range(start, len(counts))):
        count = min(int(counts[index]), k)
        if loss_mask[row] and not count:
            raise ValueError("A trainable token has no top-k probabilities")
        pos = int(offsets[index])
        ids[row, :count] = head.ids[pos : pos + count]
        logprobs[row, :count] = head.logprobs[pos : pos + count]
    return ids, logprobs


def _replay_array(branch, field: str, required: bool):
    array = {"routed_experts": branch.routed_experts, "indexer_topk": branch.indexer_topk}[field]
    if array is None:
        if required:
            raise ValueError(f"Requested {field} are incomplete in the returned branch")
        return None
    array = np.asarray(array)
    if array.ndim != 3 or len(array) != len(branch.token_ids):
        raise ValueError(f"{field} must align with the complete token sequence")
    # VF attributes model input rows to graph tokens; the final output token was
    # never fed back to the engine. Miles replay takes exactly the preceding N-1.
    return array[:-1].astype(np.int32, copy=False)


def _weight_versions(branch):
    calls = []
    offset = 0
    for node in branch.nodes:
        positions = np.flatnonzero(node.mask)
        if node.weight_versions or node.prefill_weight_versions:
            spans = []
            for span in node.weight_versions or []:
                if not 0 <= span.start < span.end <= len(positions):
                    raise ValueError("Weight-version spans extend past sampled tokens")
                # Template tokens between sampled runs must not acquire a version.
                selected = positions[span.start : span.end]
                cuts = np.flatnonzero(np.diff(selected) != 1) + 1
                for run in np.split(selected, cuts):
                    spans.append(WeightVersionSpan(str(span.version), offset + int(run[0]), offset + int(run[-1]) + 1))
            prefill = [
                WeightVersionSpan(str(span.version), span.start, span.end)
                for span in node.prefill_weight_versions or []
            ]
            output_start = offset + (int(positions[0]) if len(positions) else len(node.token_ids))
            if len(positions) and np.any(np.diff(positions) != 1):
                if prefill:
                    raise ValueError(
                        "Cannot preserve per-call prefill versions across non-contiguous sampled positions"
                    )
                calls.extend(WeightVersionsPerCall(spans=[span], output_start=span.abs_start) for span in spans)
            else:
                calls.append(
                    WeightVersionsPerCall(
                        spans=spans, prefill_spans=prefill, output_start=output_start, prompt_tokens=node.prompt_tokens
                    )
                )
        offset += len(node.token_ids)
    return calls


def _branch_to_sample(
    args: Namespace, trace, branch, mask, *, group_index: int, index: int, rollout_id: int
) -> Sample:
    tokens = branch.token_ids.tolist()
    logprobs = branch.logprobs.tolist()
    if len(tokens) != len(mask) or len(tokens) != len(logprobs):
        raise ValueError(f"Trace {trace.id} token/mask/logprob lengths differ")
    first = int(np.flatnonzero(mask)[0])
    loss_mask = mask[first:].astype(int).tolist()
    if not np.isfinite(np.asarray(logprobs)[mask]).all():
        raise ValueError(f"Trace {trace.id} contains non-finite sampled-token logprobs")
    task_data = trace.task.data.model_dump(mode="json")
    rewards = _reward_values(trace)
    sample = Sample(
        group_index=group_index,
        index=index,
        rollout_id=rollout_id,
        prompt=_serialize_prompt(task_data.get("prompt", "")),
        tokens=tokens,
        response=trace.last_reply,
        response_length=len(tokens) - first,
        label=task_data.get("label", task_data.get("answer")),
        reward=trace.reward if args.reward_key is None else {**rewards, "reward": trace.reward},
        loss_mask=loss_mask,
        rollout_log_probs=logprobs[first:],
        status=_sample_status(trace),
        routing_key=trace.id,
        weight_versions=_weight_versions(branch),
        metadata={
            "verifiers": {
                "trace_id": trace.id,
                "agent": trace.agent.name,
                "branch_index": branch.index,
                "task_index": task_data.get("idx"),
                "rewards": rewards,
                "metrics": dict(trace.metrics),
                "stop_condition": trace.stop_condition,
                "errors": [error.model_dump(mode="json") for error in trace.errors],
            }
        },
    )
    sample.rollout_sampling_mask = _sampling_support(branch, first, loss_mask)
    if args.use_sampling_support_replay and sample.rollout_sampling_mask is None:
        raise ValueError("Sampling-support replay requested but the environment returned no supports")
    sample.rollout_topk_token_ids, sample.rollout_topk_log_probs = _top_logprobs(
        branch, first, loss_mask, args.rollout_top_logprobs_num
    )
    sample.rollout_routed_experts = _replay_array(branch, "routed_experts", args.use_rollout_routing_replay)
    sample.rollout_indexer_topk = _replay_array(branch, "indexer_topk", args.use_rollout_indexer_replay)
    sample.validate()
    return sample


def trace_to_samples(
    args: Namespace, trace, *, group_index: int, index_start: int, rollout_id: int | None = None
) -> list[Sample]:
    branches = list(_trainable_branches(trace))
    samples = []
    for output_index, (branch, mask) in enumerate(branches):
        sample = _branch_to_sample(
            args,
            trace,
            branch,
            mask,
            group_index=group_index,
            index=index_start + output_index,
            rollout_id=index_start if rollout_id is None else rollout_id,
        )
        sample.lineage = SampleLineage(index_start, output_index, len(branches))
        samples.append(sample)
    return samples


def trace_to_sample(args: Namespace, trace, *, group_index: int, index: int) -> Sample:
    samples = trace_to_samples(args, trace, group_index=group_index, index_start=index)
    if len(samples) != 1:
        raise ValueError(f"Expected one branch, got {len(samples)}")
    return samples[0]


def has_image_inputs(branch) -> bool:
    return any(
        isinstance(message.get("content"), list)
        and any(part.get("type") == "image_url" for part in message["content"])
        for message in _serialize_prompt(branch.messages)
    )


def add_multimodal_inputs(sample: Sample, branch, processor: Any) -> None:
    """Use Miles' processor for training features; verify its exact token expansion.

    Generated token IDs remain the trace's IDs. A processor/template disagreement
    is an error, never permission to replace the sampled sequence by new tokens.
    """
    messages = _serialize_prompt(branch.messages)
    if not has_image_inputs(branch):
        return
    if branch.mm_token_type_ids is None:
        raise ValueError("Image traces have no multimodal token attribution")
    if processor is None:
        raise ValueError("Multimodal traces require a model processor on the Miles client")
    from miles.utils.processing_utils import call_processor, extract_multimodal_train_inputs, process_vision_info

    media = process_vision_info(messages, processor)
    # Collapse expanded placeholders before applying the training processor. This
    # prevents it expanding each of an image's already expanded tokens again.
    ids = branch.token_ids.tolist()
    types = branch.mm_token_type_ids
    collapsed = [
        token
        for i, token in enumerate(ids)
        if not (i and types[i] and types[i] == types[i - 1] and token == ids[i - 1])
    ]
    text = processor.tokenizer.decode(collapsed, skip_special_tokens=False)
    output = call_processor(processor, text, media)
    actual = output["input_ids"][0]
    actual = np.asarray(actual).tolist()
    if actual != ids:
        raise ValueError("Training processor expansion differs from the sampled multimodal token sequence")
    sample.multimodal_inputs = media
    sample.multimodal_train_inputs = extract_multimodal_train_inputs(output)
    sample.validate()
