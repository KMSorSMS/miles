"""A deterministic CPU plugin exercising real Episode.run_slot/finalize machinery."""

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import verifiers.v1 as vf
from verifiers.v1.errors import InferenceCleanupError
from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import AgentInfo, TraceTask
from verifiers.v1.types import AssistantMessage, SamplingMask, TopLogprobs


class Data(vf.TaskData):
    value: int = 7
    mode: str = "normal"
    marker: str | None = None


class Task(vf.Task[Data]):
    pass


class Taskset(vf.Taskset[Task]):
    def load(self):
        raise AssertionError("The worker must rebuild sent TaskData; never load the dataset")


class Config(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig()
    attempts: int = 2
    startup_delay: float = 0


class Env(vf.Env[Config]):
    def __init__(self, config):
        self.config = config
        self.taskset = Taskset(config.taskset)

    @asynccontextmanager
    async def serving(self):
        await asyncio.sleep(self.config.startup_delay)
        yield

    def _episode_agents(self, ctx, completed, on_trace, on_discard):
        async def run(task):
            trace = vf.Trace(
                task=TraceTask(type="Task", data=task.data),
                agent=AgentInfo(config=self.config.solver, name="solver"),
                ok=True,
                is_completed=True,
            )
            completed.append(trace)
            on_trace(trace)
            trace.nodes.append(
                MessageNode(
                    message=AssistantMessage(content=str(task.data.value)),
                    sampled=True,
                    token_ids=[11, task.data.value],
                    mask=[False, True],
                    logprobs=[-0.125],
                    routed_experts=np.arange(4, dtype=np.uint8).reshape(2, 1, 2),
                    indexer_topk=np.arange(6, dtype=np.int32).reshape(2, 1, 3),
                    sampling_mask=SamplingMask(
                        ids=np.array([task.data.value, 2], np.int32), counts=np.array([2], np.int32)
                    ),
                    top_logprobs=TopLogprobs(
                        ids=np.array([task.data.value, 2], np.int32),
                        counts=np.array([2], np.int32),
                        logprobs=np.array([-0.125, -1.5], np.float32),
                    ),
                )
            )
            if task.data.mode == "no_aux":
                for field in ("routed_experts", "indexer_topk", "sampling_mask", "top_logprobs"):
                    setattr(trace.nodes[-1], field, None)
            trace.info["pid"] = os.getpid()
            trace.notify()
            await asyncio.sleep(0.02)
            return trace

        return SimpleNamespace(solver=SimpleNamespace(run=run))

    async def run(self, task, agents):
        if task.data.mode == "crash":
            os._exit(17)
        if task.data.mode in {"wait", "abort_failure"}:
            Path(task.data.marker + ".started").write_text("started")
            try:
                await asyncio.sleep(60)
            finally:
                await asyncio.sleep(0.05)
                Path(task.data.marker).write_text("cleaned")
                if task.data.mode == "abort_failure":
                    raise InferenceCleanupError("Engine refused abort acknowledgement")
        await asyncio.gather(*(agents.solver.run(task) for _ in range(self.config.attempts)))

    async def finalize(self, task, episode):
        for trace in episode.traces:
            trace.record_reward("joint", 1 / len(episode.traces), 1)


__all__ = ["Env", "Taskset"]
