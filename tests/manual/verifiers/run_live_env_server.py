"""Exercise real sandboxes and a live SGLang endpoint through Miles env-server.

This manual qualification does not launch a GPU engine or alter existing services.
Run ``python tests/manual/verifiers/run_live_env_server.py --help`` for arguments.
Model-driven cases reuse the verifiers checkout's task definitions. Runtime fault
contracts use explicitly scripted programs in real sandboxes; they are not model
accuracy tests. Every container belongs to this run's unique VF_RUN_ID.
"""

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import pickle
import re
import sys
import time
import traceback
import uuid
from argparse import Namespace
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifiers-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True, help="Native SGLang service's OpenAI-style /v1 base URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--renderer-model-name", required=True)
    parser.add_argument("--runtime-image", default="python:3.11-slim")
    parser.add_argument("--runtime-config", type=Path, help="Optional VF runtime JSON, including a custom backend")
    parser.add_argument(
        "--preinstalled-harness-python",
        help="Validate and reuse this sandbox interpreter for the bundled chat program",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--abort-strategy", choices=["endpoint", "workers"], default="endpoint")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["runtime", "code-golf", "reverse-text", "bash-tool", "multi-turn", "tool-error", "cancel"],
        default=["runtime", "code-golf", "reverse-text", "bash-tool", "multi-turn", "tool-error", "cancel"],
    )
    return parser.parse_args()


def configure_paths(root):
    paths = [
        root,
        root / "tests/integration/fixtures",
        root / "environments/code_golf",
        root / "environments/reverse_text",
    ]
    sys.path[:0] = [str(path) for path in paths]
    os.environ["PYTHONPATH"] = os.pathsep.join([*(str(path) for path in paths), os.environ.get("PYTHONPATH", "")])


def install_preinstalled_harness(interpreter):
    """Skip package downloading only after checking the exact PEP 723 contract.

    Applied in each spawned worker by its import of this module. Other programs
    and all actual execution, interception, and grading keep their normal paths.
    """
    import tomllib

    from verifiers.v1.harnesses.utils.launch import CHAT_PROGRAM_SOURCE
    from verifiers.v1.runtimes.runtime import Runtime

    if interpreter is None:
        return
    metadata = CHAT_PROGRAM_SOURCE.split("# /// script\n", 1)[1].split("# ///", 1)[0]
    declared = tomllib.loads("\n".join(line.removeprefix("# ") for line in metadata.splitlines()))
    requirements = declared.get("dependencies", [])
    check = """import importlib.metadata, json, sys
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
assert SpecifierSet(sys.argv[2]).contains('.'.join(map(str,sys.version_info[:3]))), sys.version
for raw in json.loads(sys.argv[1]):
    requirement = Requirement(raw)
    if requirement.marker is None or requirement.marker.evaluate():
        installed = importlib.metadata.version(requirement.name)
        assert requirement.specifier.contains(installed), (raw, installed)
print('declared dependencies satisfied')
"""
    original = Runtime.prepare_uv_script

    async def prepare(runtime, script, env=None, *, activate=True):
        data = script.encode() if isinstance(script, str) else script
        if data != CHAT_PROGRAM_SOURCE.encode() or runtime.user is not None:
            return await original(runtime, script, env, activate=activate)
        path = f"{runtime.backend.scripts_dir}/{hashlib.sha256(data).hexdigest()}.py"
        if path not in runtime.backend.cache.uv_interpreters:
            checked = await runtime.run(
                [interpreter, "-c", check, json.dumps(requirements), declared["requires-python"]], {}
            )
            if checked.exit_code:
                raise RuntimeError(f"Preinstalled harness dependencies do not match: {checked.stderr}")
            await runtime.write(path, data)
            runtime.backend.cache.uv_interpreters[path] = interpreter
        return await original(runtime, script, env, activate=activate)

    Runtime.prepare_uv_script = prepare


# multiprocessing imports this file in a fresh interpreter before entering the
# worker. Keep the opt-in hook explicit and inherited without modifying VF code.
if os.environ.get("MILES_LIVE_PREINSTALLED_HARNESS_PYTHON"):
    install_preinstalled_harness(os.environ["MILES_LIVE_PREINSTALLED_HARNESS_PYTHON"])


async def docker_containers(run_id):
    process = await asyncio.create_subprocess_exec(
        "docker",
        "ps",
        "-aq",
        "--filter",
        f"label=verifiers.run={run_id}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    output, error = await process.communicate()
    if process.returncode:
        raise RuntimeError(f"Cannot inspect this run's containers: {error.decode()}")
    return output.decode().split()


async def assert_no_containers(runtime, run_id):
    if runtime["type"] == "docker":
        remaining = await docker_containers(run_id)
        assert not remaining, f"Owned sandbox containers leaked: {remaining}"


async def wait_until_running(event, task, timeout):
    """Surface startup failure immediately instead of misreporting a timeout."""
    waiter = asyncio.create_task(event.wait())
    try:
        done, _ = await asyncio.wait([waiter, task], timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            await task
            raise AssertionError("Task completed before the cancellation boundary")
        if waiter not in done:
            raise TimeoutError("Task did not reach the cancellation boundary")
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


def scripted_trace(task, program):
    import verifiers.v1 as vf
    from verifiers.v1.graph import MessageNode

    return vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[MessageNode(message=vf.AssistantMessage(content=f"```python\n{program}\n```"), sampled=True)],
        ok=True,
        is_completed=True,
    )


async def runtime_contracts(runtime_data, run_id):
    """Known programs isolate runtime/grading correctness from model capability."""
    import verifiers.v1 as vf
    from code_golf.taskset import CodeGolfData, CodeGolfEnv, CodeGolfEnvConfig, CodeGolfTask
    from pydantic import TypeAdapter
    from verifiers.v1.runtimes import RuntimeConfig, provision_runtime

    config = TypeAdapter(RuntimeConfig).validate_python(runtime_data)
    task = CodeGolfTask(CodeGolfData(prompt="runtime grading contract", expected="17"))
    traces = [scripted_trace(task, "print(17)"), scripted_trace(task, "raise SystemExit(7)")]
    async with provision_runtime(config) as runtime:
        container = runtime.backend.info.id
        devices = await runtime.run(["python3", "-c", "import glob; assert not glob.glob('/dev/nvidia*')"], {})
        assert devices.exit_code == 0, "The environment sandbox unexpectedly has GPU devices"
        binary = bytes(range(256))
        await runtime.write("nested/binary.dat", binary)
        assert await runtime.read("nested/binary.dat") == binary
        result = await runtime.run(
            ["python3", "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(7)"], {}
        )
        assert (result.exit_code, result.stdout.strip(), result.stderr.strip()) == (7, "out", "err")
        for trace in traces:
            await task.score(trace, runtime)
        assert [trace.metrics["passed"] for trace in traces] == [1, 0]
        assert [trace.reward for trace in traces] == [1, 0]
        assert all(0 < trace.metrics["latency"] < 30 for trace in traces)
        process = await runtime.exec(
            ["python3", "-u", "-c", "import time; print('ready',flush=True); time.sleep(60)"], {}
        )
        assert b"ready" in await asyncio.wait_for(anext(process.stdout), 10)
        try:
            async with asyncio.timeout(0.2):
                await process.wait()
        except TimeoutError:
            await process.kill()
            assert await asyncio.wait_for(process.wait(), 10) != 0
        else:
            raise AssertionError("The sleeping process did not time out")
        assert (await runtime.run(["true"], {})).exit_code == 0
    await assert_no_containers(runtime_data, run_id)
    env = CodeGolfEnv(CodeGolfEnvConfig(taskset={"id": "code-golf"}))
    await env.finalize(task, vf.Episode(task=traces[0].task, traces=traces))
    assert [trace.reward for trace in traces] == [2, 0]
    failed = traces[1].model_copy(deep=True)
    await env.finalize(task, vf.Episode(task=failed.task, traces=[failed]))
    assert failed.reward == 0
    entered = asyncio.Event()

    async def cancelled_owner():
        async with provision_runtime(config) as runtime:
            process = await runtime.exec(
                ["python3", "-u", "-c", "import time; print('ready',flush=True); time.sleep(60)"], {}
            )
            assert b"ready" in await asyncio.wait_for(anext(process.stdout), 10)
            entered.set()
            await process.wait()

    owner = asyncio.create_task(cancelled_owner())
    try:
        await wait_until_running(entered, owner, 60)
        owner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(owner, 40)
        assert owner.cancelled(), "Runtime context swallowed task cancellation"
    finally:
        if not owner.done():
            owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
    await assert_no_containers(runtime_data, run_id)
    return {
        "kind": "scripted-real-runtime-contract",
        "runtime_id": container,
        "gpu_devices": [],
        "binary_io": True,
        "nonzero_exit": 7,
        "joint_rewards": [2, 0],
        "all_failed_reward": 0,
        "process_timeout_killed": True,
        "cancelled_owner_cleaned": True,
    }


@dataclass(frozen=True)
class Case:
    name: str
    env: dict
    tasks: list[dict]
    traces_per_episode: int = 1


def cases(runtime):
    from verifiers.v1.utils.loaders import load_taskset, resolve_env_config

    def build(name, taskset, *, harness="null", role="agent", extra=None):
        seat = {
            "harness": {"id": harness},
            "runtime": runtime,
            "timeout": {"rollout": 180, "scoring": 20},
            "max_turns": 6,
        }
        if harness == "bash":
            seat["harness"].update(edit=False, compaction=None)
        env = {"taskset": {"id": taskset}, role: seat, **(extra or {})}
        data = [task.data.model_dump(mode="json") for task in load_taskset(resolve_env_config(env).taskset)]
        return Case(name, env, data, env.get("attempts", 1))

    result = {
        "code-golf": build("code-golf", "code-golf", role="golfer", extra={"attempts": 4, "max_concurrent_agents": 4}),
        "bash-tool": build("bash-tool", "echo_agentic", harness="bash"),
        "multi-turn": build("multi-turn", "echo_user_sim"),
        "tool-error": build("tool-error", "tool_error"),
    }
    # Do not download the full HF dataset for a boundary qualification: these
    # fixed TaskData values use the genuine reverse-text task and reward.
    from reverse_text.taskset import ReverseTextData, SYSTEM

    strings = ["abc", "A1 b2", "你好éA"]
    env = {
        "taskset": {"id": "reverse-text"},
        "agent": {"harness": {"id": "null"}, "runtime": runtime, "timeout": {"rollout": 180}, "max_turns": 1},
    }
    result["reverse-text"] = Case(
        "reverse-text",
        env,
        [
            ReverseTextData(idx=i, prompt=s, system_prompt=SYSTEM, answer=s[::-1]).model_dump(mode="json")
            for i, s in enumerate(strings)
        ],
    )
    result["cancel"] = Case("cancel", result["code-golf"].env, result["code-golf"].tasks[:1], 4)
    return result


def check_episode(case, episode, index):
    from examples.experimental.verifiers.verifiers_samples import trace_to_samples

    assert episode.ok, episode.errors
    assert len(episode.traces) == case.traces_per_episode
    args = Namespace(
        reward_key=None,
        use_sampling_support_replay=False,
        rollout_top_logprobs_num=2,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
    )
    samples = []
    for rollout_id, trace in enumerate(episode.traces):
        samples.extend(
            trace_to_samples(args, trace, group_index=index, index_start=len(samples), rollout_id=rollout_id)
        )
    assert samples and all(sample.response_length > 0 for sample in samples)
    for sample in samples:
        sample.validate()
        assert all(math.isfinite(value) for value in sample.rollout_log_probs)
    for trace in episode.traces:
        assert trace.num_turns and not trace.has_error, trace.errors
        assert all(math.isfinite(trace.reward) for trace in episode.traces)
    independently_scored = []
    if case.name in {"code-golf", "cancel"}:
        from code_golf.taskset import extract_program

        passing = [trace for trace in episode.traces if not trace.is_format_violation and trace.metrics.get("passed")]
        shortest = min((len(extract_program(trace)) for trace in passing), default=None)
        fastest = min((trace.metrics["latency"] for trace in passing), default=None)
        for trace in episode.traces:
            if trace.is_format_violation:
                assert trace.reward == 0
                independently_scored.append(0.0)
                continue
            assert trace.rewards["correct"].value == float(trace in passing)
            assert trace.rewards["most_concise"].value == (
                0.5 if trace in passing and len(extract_program(trace)) == shortest else 0
            )
            assert trace.rewards["fastest"].value == (
                0.5 if trace in passing and trace.metrics["latency"] == fastest else 0
            )
            independently_scored.append(
                float(trace in passing)
                + (0.5 if trace in passing and len(extract_program(trace)) == shortest else 0)
                + (0.5 if trace in passing and trace.metrics["latency"] == fastest else 0)
            )
    if case.name == "reverse-text":
        for trace in episode.traces:
            assert trace.num_turns == 1 and trace.num_branches == 1
            match = re.search(r"<reversed_text>(.*?)</reversed_text>", trace.last_reply or "", re.DOTALL)
            answer = match.group(1).strip() if match else ""
            expected = (
                0.0
                if trace.is_format_violation
                else SequenceMatcher(None, answer, case.tasks[index]["answer"]).ratio()
            )
            assert math.isclose(trace.reward, expected), (answer, trace.reward, expected)
            independently_scored.append(expected)
    if case.name in {"bash-tool", "tool-error"}:
        assert all(trace.tool_messages and trace.num_turns >= 2 and trace.reward == 1 for trace in episode.traces)
    if case.name == "multi-turn":
        assert all(trace.num_turns == 2 and trace.num_branches == 1 and trace.reward == 1 for trace in episode.traces)
    return {
        "task": episode.task.data.model_dump(mode="json"),
        "rewards": [trace.reward for trace in episode.traces],
        "turns": [trace.num_turns for trace in episode.traces],
        "branches": [trace.num_branches for trace in episode.traces],
        "sampled_tokens": sum(sum(sample.loss_mask) for sample in samples),
        "samples": len(samples),
        "independently_scored_rewards": independently_scored,
        "runtime_ids": [trace.agent.runtime.id for trace in episode.traces if trace.agent.runtime],
    }


async def model_case(args, case, runtime, run_id):
    from renderers.configs import Qwen3RendererConfig
    from verifiers.v1.configs.client import TrainClientConfig
    from verifiers.v1.configs.sampling import SamplingConfig
    from miles.env_server.client_pool import EnvClientPool
    from miles.env_server.config import EnvServerConfig
    from miles.env_server.payload import materialize_episode
    from miles.env_server.pool import EnvServerPool

    server = EnvServerPool(
        EnvServerConfig(
            env=case.env, serve={"pool": {"type": "static", "num_workers": args.workers}, "shutdown_timeout": 40}
        )
    )
    serving = asyncio.create_task(server.run())
    pool = EnvClientPool([server.address], timeout=240, cancel_timeout=40)
    client = TrainClientConfig(
        base_url=args.base_url,
        backend="sglang",
        renderer_model_name=args.renderer_model_name,
        renderer=Qwen3RendererConfig(enable_thinking=False),
        backend_options={
            "max_context_len": 4096,
            "abort_timeout": 15,
            "abort_strategy": args.abort_strategy,
            "min_top_logprobs": 2,
        },
    )
    sampling = SamplingConfig(max_tokens=512, temperature=0.2, seed=100, logprobs=2)
    try:
        await pool.wait_for_server_startup(timeout=120)
        assert (await pool.describe()).ready_workers == args.workers
        if case.name == "cancel":
            pending = asyncio.Event()
            running = asyncio.create_task(
                pool.run(
                    client,
                    args.model,
                    SamplingConfig(
                        max_tokens=2000, min_tokens=2000, ignore_eos=True, temperature=0.8, seed=991, logprobs=2
                    ),
                    case.tasks[0],
                    on_delta=lambda delta: pending.set() if delta.get("pending") else None,
                )
            )
            try:
                await wait_until_running(pending, running, 180)
                await asyncio.sleep(0.25)
                assert not running.done(), "Generation completed before the cancellation test"
                active = await docker_containers(run_id) if runtime["type"] == "docker" else []
                if runtime["type"] == "docker":
                    assert active, "Cancellation must reach a live environment sandbox"
                running.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await running
                assert running.cancelled()
                await assert_no_containers(runtime, run_id)
            finally:
                if not running.done():
                    running.cancel()
                    await asyncio.gather(running, return_exceptions=True)
            episode = await pool.run(client, args.model, sampling, case.tasks[0])
            await materialize_episode(episode)
            recovery = check_episode(case, episode, 0)
            return {
                "kind": "model-in-loop",
                "cancellation_boundary": "episode with live environment sandboxes; GPU request state not asserted",
                "containers_before_cancel": active,
                "cancelled": True,
                "recovery_episode": recovery,
                "owned_containers_removed": True,
            }
        episodes = await asyncio.gather(
            *(
                pool.run(client, args.model, sampling.model_copy(update={"seed": 100 + i * 10}), data)
                for i, data in enumerate(case.tasks)
            )
        )
        await asyncio.gather(*(materialize_episode(episode) for episode in episodes))
        # Materialized training arrays may contain arbitrary packed bytes; retain
        # them losslessly instead of asking a JSON encoder to decode them as UTF-8.
        with (args.output / f"{case.name}-episodes.pkl").open("wb") as artifact:
            pickle.dump([episode.model_dump(mode="python") for episode in episodes], artifact)
        return {
            "kind": "model-in-loop",
            "episodes": [check_episode(case, episode, i) for i, episode in enumerate(episodes)],
            "owned_containers_removed": True,
        }
    finally:
        try:
            await pool.close()
        finally:
            serving.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await serving
            await assert_no_containers(runtime, run_id)


async def run(args):
    runtime = (
        json.loads(args.runtime_config.read_text())
        if args.runtime_config
        else {"type": "docker", "image": args.runtime_image, "cpu": 1, "memory": 2}
    )
    run_id = f"miles-live-{uuid.uuid4().hex}"
    os.environ["VF_RUN_ID"] = run_id
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"run_id": run_id, "model": args.model, "runtime_type": runtime["type"], "cases": {}}
    available = cases(runtime)
    for name in args.cases:
        print(f"Starting {name}", flush=True)
        started = time.monotonic()
        try:
            result = (
                await runtime_contracts(runtime, run_id)
                if name == "runtime"
                else await model_case(args, available[name], runtime, run_id)
            )
            report["cases"][name] = {"ok": True, **result}
        except Exception as error:
            report["cases"][name] = {"ok": False, "error": str(error), "traceback": traceback.format_exc()}
        report["cases"][name]["seconds"] = round(time.monotonic() - started, 3)
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({name: report["cases"][name]}, ensure_ascii=False), flush=True)
    return all(case["ok"] for case in report["cases"].values())


if __name__ == "__main__":
    arguments = parse_args()
    configure_paths(arguments.verifiers_root.resolve())
    if arguments.preinstalled_harness_python:
        os.environ["MILES_LIVE_PREINSTALLED_HARNESS_PYTHON"] = arguments.preinstalled_harness_python
        install_preinstalled_harness(arguments.preinstalled_harness_python)
    raise SystemExit(0 if asyncio.run(run(arguments)) else 1)
