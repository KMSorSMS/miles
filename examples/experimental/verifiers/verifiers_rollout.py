"""Miles client for complete episodes on distributed CPU environment servers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
from argparse import Namespace

from examples.experimental.verifiers.verifiers_config import ManagedServer, load_adapter_config, load_config_data
from examples.experimental.verifiers.verifiers_samples import (
    _trace_eval_reward,
    _trainable_branches,
    add_multimodal_inputs,
    has_image_inputs,
    trace_to_samples,
)

from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
    compute_kv_cache_namespace,
)
from miles.rollout.filter_hub.base_types import MetricGatherer
from miles.rollout.filter_hub.common_filters import apply_preput_filters
from miles.utils.lora.utils import LORA_ADAPTER_NAME, lora_rollout_enabled

logger = logging.getLogger(__name__)
_CONFIG_ENV_VAR = "VERIFIERS_CONFIG"
_load_config_data = load_config_data


def _import_verifiers():
    import verifiers.v1 as vf
    from verifiers.v1.configs.client import TrainClientConfig

    if "backend" not in TrainClientConfig.model_fields:
        raise RuntimeError("Install matching verifiers/renderers SGLang branches; see this example's README")
    return vf


def _config_path():
    path = os.environ.get(_CONFIG_ENV_VAR)
    if not path:
        raise ValueError(f"Set {_CONFIG_ENV_VAR} to the distributed adapter TOML")
    return path


def _generate_url(args, endpoint="/generate"):
    routers = args.sglang_model_routers
    ip, port = (
        routers["default"] if routers and "default" in routers else (args.sglang_router_ip, args.sglang_router_port)
    )
    if ip is None or port is None:
        raise RuntimeError("Miles inference router is not ready")
    host = f"[{ip}]" if ":" in ip and not ip.startswith("[") else ip
    return f"http://{host}:{port}{endpoint}"


def _flatten_samples(values):
    return [sample for value in values for sample in (_flatten_samples(value) if isinstance(value, list) else [value])]


def _make_eval_args(args):
    result = Namespace(**(vars(args) if isinstance(args, Namespace) else dict(args)))
    for source, target in (
        ("eval_temperature", "rollout_temperature"),
        ("eval_top_p", "rollout_top_p"),
        ("eval_top_k", "rollout_top_k"),
        ("eval_max_response_len", "rollout_max_response_len"),
        ("eval_max_context_len", "rollout_max_context_len"),
    ):
        if (
            value := getattr(args, source)
        ) is not None:  # config-access-exempt: fixed declared eval-to-rollout field mapping
            setattr(result, target, value)
    result.rollout_max_prompt_len = args.eval_max_prompt_len
    result.reward_key = args.eval_reward_key or args.reward_key
    result.use_sampling_support_replay = False
    result.rollout_top_logprobs_num = 0
    result.use_rollout_routing_replay = False
    result.use_rollout_indexer_replay = False
    return result


def _validate_args(args):
    if args.rollout_global_dataset:
        raise ValueError("Pass --disable-rollout-global-dataset; verifiers loads the taskset")
    if args.partial_rollout:
        raise ValueError("--partial-rollout cannot resume arbitrary live environment state")
    if args.chat_template_path is not None:
        raise ValueError("Set client.renderer instead of --chat-template-path")
    if args.use_opd:
        raise ValueError("--use-opd needs a teacher scoring path; env-server transport does not supply one")
    if args.use_output_store:
        raise ValueError("SGLang output-store is separate from env-server payloads; disable --use-output-store")


def _trace_metrics(traces):
    if not traces:
        return {}
    return {
        "verifiers/reward_mean": sum(t.reward for t in traces) / len(traces),
        "verifiers/error_rate": sum(t.has_error for t in traces) / len(traces),
        "verifiers/truncated_rate": sum(t.is_truncated for t in traces) / len(traces),
        "verifiers/num_turns_mean": sum(t.num_turns for t in traces) / len(traces),
    }


def _raise_for_fatal_episode_errors(episodes):
    for episode in episodes:
        errors = [(error, episode.ok) for error in episode.errors]
        errors.extend((error, trace.ok) for trace in episode.traces for error in trace.errors)
        for error, succeeded in errors:
            if succeeded and error.type not in {"EngineAbortError", "InferenceCleanupError"}:
                # VF retains prior retry errors on a successful final episode/trace.
                continue
            if error.type in {
                "NotImplementedError",
                "ValidationError",
                "ValueError",
                "EngineAbortError",
                "InferenceCleanupError",
            } or (
                error.status_code is not None
                and 400 <= error.status_code < 500
                and error.status_code not in {408, 429}
            ):
                raise RuntimeError(f"Environment configuration/protocol error: {error.type}: {error.message}")


class VerifiersRolloutFn(BaseRolloutFn):
    def __init__(self, input):
        super().__init__(input)
        _import_verifiers()
        from verifiers.v1.configs.sampling import SamplingConfig
        from verifiers.v1.utils.loaders import load_taskset

        from miles.utils.function_registry import load_function

        self.args, self.data_source = input.args, input.data_source
        _validate_args(self.args)
        self.config, self.env_config = load_adapter_config(_config_path())
        self.model = self.args.hf_checkpoint
        self.eval_args = _make_eval_args(self.args)
        self.sampling = self._sampling_config(SamplingConfig, self.args, min_tokens=None)
        self.eval_sampling = self._sampling_config(
            SamplingConfig, self.eval_args, min_tokens=self.args.eval_min_new_tokens
        )
        self.max_concurrent = self.args.sglang_server_concurrency * max(
            self.args.inference_runtime_mut_state.engine_count, 1
        )
        self.dynamic_filter = load_function(self.args.dynamic_sampling_filter_path)
        # Data loading only: the rollout actor constructs no Environment/runtime/interception.
        taskset = load_taskset(self.env_config.taskset).select(self.config.select)
        if not taskset.bounded:
            raise ValueError("Bound infinite tasksets with select.limit or select.include.idx")
        self._tasks = list(taskset)
        if self.args.rollout_shuffle:
            random.Random(self.args.rollout_seed).shuffle(self._tasks)
        if not self._tasks:
            raise ValueError("Verifiers taskset selected zero tasks")
        data = [task.data.model_dump(mode="json") for task in self._tasks]
        self._task_fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        self._next_train_task_idx = self._next_group_index = self._next_sample_index = self._next_rollout_index = 0
        self._next_seed = self.args.rollout_seed
        self._remote = None
        self._managed = ManagedServer(self.config.remote)
        self._start_lock = asyncio.Lock()
        self._active = set()
        self._processor = None

    @staticmethod
    def _sampling_config(SamplingConfig, args, *, min_tokens):
        data = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_tokens=args.rollout_max_response_len,
        )
        if min_tokens is not None:
            data["min_tokens"] = min_tokens
        if args.apply_chat_template_kwargs:
            data["chat_template_kwargs"] = args.apply_chat_template_kwargs
        if args.rollout_top_logprobs_num:
            data["logprobs"] = args.rollout_top_logprobs_num
        return SamplingConfig.model_validate(data)

    def _client_config(self, args):
        from renderers.backends.base import SGLangOptions

        # Router configuration imports the engine's optional router package.
        from miles.router.backend import make_router_backend

        template = self.config.client
        options = dict(template.backend_options)
        # An explicit CPU-visible DNS name normally still refers to Miles' router.
        # Direct-engine deployments can explicitly select endpoint cancellation.
        options.setdefault("abort_strategy", "workers")
        options["sampling_constraints"] = (
            {
                "temperature": args.rollout_temperature,
                "top_p": args.rollout_top_p,
                "top_k": args.rollout_top_k if args.rollout_top_k is not None else -1,
                "min_p": 0.0,
            }
            if args is self.args
            else {}
        )
        options["min_top_logprobs"] = args.rollout_top_logprobs_num or None
        backend = make_router_backend(self.args)
        required = {
            "max_prompt_len": args.rollout_max_prompt_len,
            "max_context_len": args.rollout_max_context_len,
            "max_response_tokens": args.rollout_max_response_len,
            "skip_special_tokens": args.rollout_skip_special_tokens,
            "stop": args.rollout_stop,
            "stop_token_ids": args.rollout_stop_token_ids,
            "return_sampling_mask": args.use_sampling_support_replay,
            "return_routed_experts": args.use_rollout_routing_replay,
            "return_indexer_topk": args.use_rollout_indexer_replay,
            "lora_path": LORA_ADAPTER_NAME if lora_rollout_enabled(args) else None,
            "routing_header": (
                backend.features.routing_key_header if backend.routes_by_key(self.args.sglang.models) else None
            ),
        }
        # Miles owns launch constraints; reject contradictory pins. Evaluation
        # intentionally disables training-only data collection and uses eval caps.
        for key, value in required.items():
            if key in options and options[key] != value and args is self.args:
                raise ValueError(f"client.backend_options.{key} conflicts with Miles' resolved value {value!r}")
            options[key] = value
        options = SGLangOptions.model_validate(options).model_dump(mode="json", exclude_none=True)
        tokenizer = template.renderer_model_name or self.args.sglang.common_value("tokenizer_path") or self.model
        # Resolve router after Miles has launched engines; an explicit URL can
        # name the same service through DNS reachable from CPU workers.
        url = template.base_url if "base_url" in template.model_fields_set else _generate_url(self.args, "/v1")
        return type(template).model_validate(
            {**template.model_dump(), "base_url": url, "renderer_model_name": tokenizer, "backend_options": options}
        )

    async def _ensure_remote(self):
        from miles.env_server import EnvClientPool
        from miles.env_server.config import config_fingerprint

        async with self._start_lock:
            if self._remote is not None:
                return
            addresses = list(self.config.remote.addresses)
            if self.config.remote.server_config is not None:
                addresses.append(await self._managed.start())
            remote = EnvClientPool(
                addresses, timeout=self.config.remote.request_timeout, cancel_timeout=self.config.remote.cancel_timeout
            )
            try:
                await remote.wait_for_server_startup(timeout=self.config.remote.startup_timeout)
                expected = config_fingerprint(self.env_config.model_dump(mode="json", serialize_as_any=True))
                description = await remote.describe()
                if description.config_fingerprint != expected:
                    raise ValueError("Environment server config differs from adapter [env] config")
            except BaseException:
                try:
                    await remote.close()
                finally:
                    await self._managed.close()
                raise
            self._remote = remote

    async def _run_task_group(self, task, n, semaphore, seed_base, *, evaluation=False, namespace=None):
        await self._ensure_remote()
        from miles.env_server.payload import materialize_episode

        args = self.eval_args if evaluation else self.args
        base = self.eval_sampling if evaluation else self.sampling
        client = self._client_config(args)
        count = self.config.remote.eval_episodes_per_prompt if evaluation else self.config.remote.episodes_per_prompt
        count = count or n

        async def run_one(offset):
            sampling = base.model_copy(deep=True)
            if self.args.sglang.common_value("enable_deterministic_inference"):
                sampling = sampling.model_copy(update={"seed": seed_base + offset * n})
            if namespace is not None:
                sampling = sampling.model_copy(update={"cache_salt": namespace})
            async with semaphore:
                episode = await self._remote.run(
                    task_data=task.data.model_dump(mode="json", serialize_as_any=True),
                    client=client,
                    model=self.model,
                    sampling=sampling,
                )
                await materialize_episode(episode)
                return episode

        futures = [asyncio.create_task(run_one(offset)) for offset in range(count)]
        try:
            return await asyncio.gather(*futures)
        finally:
            await self._cancel_pending(futures)

    def _convert_group(self, episodes, *, group_index, args=None):
        args = self.args if args is None else args
        samples, traces = [], []
        for episode in episodes:
            for trace in episode.traces:
                converted = trace_to_samples(
                    args,
                    trace,
                    group_index=group_index,
                    index_start=self._next_sample_index,
                    rollout_id=self._next_rollout_index,
                )
                if not converted:
                    continue
                traces.append(trace)
                self._next_sample_index += len(converted)
                self._next_rollout_index += 1
                for sample, (branch, _) in zip(converted, _trainable_branches(trace), strict=True):
                    if has_image_inputs(branch):
                        if branch.mm_token_type_ids is None:
                            raise ValueError("Image traces have no multimodal token attribution")
                        if self._processor is None:
                            from miles.utils.processing_utils import load_processor

                            self._processor = load_processor(self.model, trust_remote_code=True)
                        add_multimodal_inputs(sample, branch, self._processor)
                    sample.metadata["verifiers"]["episode_id"] = episode.id
                samples.extend(converted)
        return samples, traces

    async def _apply_miles_rewards(self, group):
        if not (self.args.group_rm or self.args.custom_rm_path is not None or self.args.rm_type):
            return
        # Optional Miles graders are unnecessary when rewards came from the env.
        from miles.rollout.rm_hub import async_rm, batched_async_rm

        samples = _flatten_samples(group)
        by_rollout = {}
        for sample in samples:
            by_rollout.setdefault(sample.rollout_id, sample)
        representatives = list(by_rollout.values())
        if self.args.group_rm:
            rewards = await batched_async_rm(self.args, representatives)
        elif self.args.custom_rm_path is not None or self.args.rm_type:
            rewards = await asyncio.gather(*(async_rm(self.args, sample) for sample in representatives))
        if rewards is None or len(rewards) != len(representatives):
            raise ValueError("Miles reward model returned an unexpected number of rewards")
        # Physical graph branches are one logical rollout and must share its reward.
        by_reward = dict(zip(by_rollout, rewards, strict=True))
        for sample in samples:
            sample.reward = by_reward[sample.rollout_id]

    async def _postprocess_train_samples(self, data, all_data):
        from miles.rollout.generate_utils.prefill_logprobs import recompute_samples_rollout_logprobs_via_prefill
        from miles.utils.function_registry import load_function

        if function := load_function(self.args.rollout_sample_filter_path):
            function(self.args, data)
        if function := load_function(self.args.rollout_all_samples_process_path):
            function(self.args, all_data, self.data_source)
        await recompute_samples_rollout_logprobs_via_prefill(
            self.args,
            _flatten_samples(data),
            url=_generate_url(self.args),
            sampling_params={
                "temperature": self.args.rollout_temperature,
                "top_p": self.args.rollout_top_p,
                "top_k": self.args.rollout_top_k,
                "max_new_tokens": self.args.rollout_max_response_len,
            },
        )

    async def _cancel_pending(self, futures):
        futures = list(futures)
        for future in futures:
            if not future.done():
                future.cancel()
        # EnvClient waits for the terminal reply and cleanup. Never abort_all on shared engines.
        draining = asyncio.gather(*futures, return_exceptions=True)
        cancelled = False
        while not draining.done():
            try:
                await asyncio.shield(draining)
            except asyncio.CancelledError:
                cancelled = True
        results = draining.result()
        from miles.env_server.client import EnvDrainError

        for result in results:
            if isinstance(result, EnvDrainError):
                raise result
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                raise result
        if cancelled:
            raise asyncio.CancelledError

    async def __call__(self, input):
        return await (self._call_eval(input) if input.evaluation else self._call_train(input))

    async def _call_train(self, input):
        from miles.utils import dumper_utils

        await dumper_utils.configure_sglang(self.args)
        await self._ensure_remote()
        target = self.args.rollout_batch_size
        groups, all_groups, all_traces = [], [], []
        metrics = MetricGatherer()
        semaphore, pending, dispatched = asyncio.Semaphore(self.max_concurrent), set(), 0
        namespace = compute_kv_cache_namespace(self.args, input)
        try:
            while len(groups) < target:
                capacity = max(target - len(groups), self.args.over_sampling_batch_size)
                while len(pending) < capacity and dispatched < self.config.remote.max_task_attempts:
                    task = self._tasks[self._next_train_task_idx % len(self._tasks)]
                    self._next_train_task_idx += 1
                    seed = self._next_seed
                    count = self.config.remote.episodes_per_prompt or self.args.n_samples_per_prompt
                    self._next_seed += count * self.args.n_samples_per_prompt
                    future = asyncio.create_task(
                        self._run_task_group(
                            task, self.args.n_samples_per_prompt, semaphore, seed, namespace=namespace
                        )
                    )
                    pending.add(future)
                    self._active.add(future)
                    dispatched += 1
                if not pending:
                    raise RuntimeError("No batch after remote.max_task_attempts; inspect environment/filter errors")
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for future in done:
                    self._active.discard(future)
                    episodes = future.result()
                    _raise_for_fatal_episode_errors(episodes)
                    all_traces.extend(trace for episode in episodes for trace in episode.traces)
                    if any(not episode.ok for episode in episodes):
                        metrics.on_dynamic_filter_drop(reason="episode_error")
                        continue
                    group, traces = self._convert_group(episodes, group_index=self._next_group_index)
                    self._next_group_index += 1
                    if len(traces) != self.args.n_samples_per_prompt:
                        raise ValueError(
                            f"Environment returned {len(traces)} trainable runs; Miles requires {self.args.n_samples_per_prompt}. Set remote.episodes_per_prompt and env's attempt count together"
                        )
                    await self._apply_miles_rewards(group)
                    all_groups.append(group)
                    metrics.on_group_before_dynamic_filter(self.args, group)
                    filtered = apply_preput_filters(self.args, self.dynamic_filter, group)
                    if not filtered.keep:
                        metrics.on_dynamic_filter_drop(reason=filtered.reason)
                    elif len(groups) < target:
                        groups.append(group)
        finally:
            try:
                await self._cancel_pending(self._active)
            finally:
                self._active.clear()
        groups.sort(key=lambda group: group[0].index)
        await self._postprocess_train_samples(groups, all_groups)
        return RolloutFnTrainOutput(samples=groups, metrics={**metrics.collect(), **_trace_metrics(all_traces)})

    async def _call_eval(self, input):
        if self.args.group_rm:
            raise ValueError("Group RM is not supported for eval rollout")
        await self._ensure_remote()
        semaphore = asyncio.Semaphore(self.max_concurrent)
        namespace = compute_kv_cache_namespace(self.args, input)
        futures = [
            asyncio.create_task(
                self._run_task_group(
                    task,
                    self.args.n_samples_per_eval_prompt,
                    semaphore,
                    self.args.rollout_seed + index * self.args.n_samples_per_eval_prompt**2,
                    evaluation=True,
                    namespace=namespace,
                )
            )
            for index, task in enumerate(self._tasks)
        ]
        self._active.update(futures)
        try:
            episode_groups = await asyncio.gather(*futures)
        finally:
            try:
                await self._cancel_pending(futures)
            finally:
                self._active.difference_update(futures)
        samples, rewards, truncated, all_traces = [], [], [], []
        for index, episodes in enumerate(episode_groups):
            _raise_for_fatal_episode_errors(episodes)
            group, traces = self._convert_group(episodes, group_index=index, args=self.eval_args)
            all_traces.extend(trace for episode in episodes for trace in episode.traces)
            await self._apply_miles_rewards(group)
            samples.extend(group)
            if any(not episode.ok for episode in episodes):
                # A failed finalizer invalidates the complete comparison; do not
                # report partial per-trace rewards as successful eval outcomes.
                rewards.extend([None] * self.args.n_samples_per_eval_prompt)
                truncated.extend(
                    [any(episode.is_timeout for episode in episodes)] * self.args.n_samples_per_eval_prompt
                )
                continue
            if len(traces) != self.args.n_samples_per_eval_prompt:
                raise ValueError(
                    "Eval trainable-run count differs from n_samples_per_eval_prompt; check remote.eval_episodes_per_prompt"
                )
            for trace in traces:
                own = [sample for sample in group if sample.routing_key == trace.id]
                reward = (
                    sum(s.get_reward_value(self.eval_args) for s in own) / len(own)
                    if self.args.custom_rm_path or self.args.rm_type
                    else _trace_eval_reward(trace, self.eval_args.reward_key)
                )
                rewards.append(reward)
                truncated.append(trace.is_truncated)
        return RolloutFnEvalOutput(
            data={self.env_config.env_id: {"rewards": rewards, "truncated": truncated, "samples": samples}},
            metrics=_trace_metrics(all_traces),
        )

    def save(self, directory):
        if self._active:
            raise RuntimeError("Drain environment requests before checkpointing")
        directory.mkdir(parents=True, exist_ok=True)
        fields = (
            "_next_train_task_idx",
            "_next_group_index",
            "_next_sample_index",
            "_next_rollout_index",
            "_next_seed",
            "_task_fingerprint",
        )
        temporary = directory / "verifiers-state.json.tmp"
        temporary.write_text(
            json.dumps({name: getattr(self, name) for name in fields})
        )  # config-access-exempt: fixed checkpoint field list, not optional launch configuration
        temporary.replace(directory / "verifiers-state.json")

    def load(self, directory):
        state = json.loads((directory / "verifiers-state.json").read_text())
        if state.pop("_task_fingerprint") != self._task_fingerprint:
            raise ValueError("Cannot resume a different taskset or shuffle order")
        for name in (
            "_next_train_task_idx",
            "_next_group_index",
            "_next_sample_index",
            "_next_rollout_index",
            "_next_seed",
        ):
            setattr(self, name, state[name])

    async def dispose(self):
        try:
            await self._cancel_pending(self._active)
        finally:
            self._active.clear()
            try:
                if self._remote is not None:
                    await self._remote.close()
                    self._remote = None
            finally:
                await self._managed.close()


_LEGACY_INSTANCES = {}


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from miles.utils.async_utils import run

    key = (id(args), id(data_source), evaluation)
    if key not in _LEGACY_INSTANCES:
        _LEGACY_INSTANCES[key] = VerifiersRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))
    input = RolloutFnEvalInput(rollout_id) if evaluation else RolloutFnTrainInput(rollout_id)
    return run(_LEGACY_INSTANCES[key](input))
