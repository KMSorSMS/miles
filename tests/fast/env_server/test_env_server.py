import asyncio
import contextlib
import os
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path

import httpx
import numpy as np
import pytest

vf = pytest.importorskip("verifiers.v1")
if (
    not hasattr(vf, "TrainClientConfig")
    or "backend" not in vf.TrainClientConfig.model_fields
    or not hasattr(vf, "InferenceCleanupError")
):
    pytest.skip("Requires the native-SGLang Verifiers SDK with inference cleanup propagation", allow_module_level=True)

from examples.experimental.verifiers.verifiers_samples import trace_to_samples
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.episode import WireEpisode
from verifiers.v1.utils.loaders import resolve_env_config

from miles.env_server.client import EnvClient, EnvDrainError
from miles.env_server.client_pool import EnvClientPool
from miles.env_server.config import ElasticPoolConfig, EnvServerConfig, PayloadConfig, config_fingerprint
from miles.env_server.delta import EpisodeAssembly, TraceSummary
from miles.env_server.payload import PAYLOAD_INFO_KEY, PayloadServer, materialize_episode
from miles.env_server.pool import EnvServerPool
from miles.env_server.server import EnvServer

sys.path.insert(0, str(Path(__file__).parent))


def request(mode="normal", **data):
    return dict(
        client=EvalClientConfig(),
        model="unused",
        sampling=SamplingConfig(),
        task_data={"prompt": "sent in full", "value": 17, "mode": mode, **data},
    )


def config(workers=2, **serve):
    return EnvServerConfig(
        env={"taskset": {"id": "miles_env_fixture"}, "attempts": 2},
        serve={"pool": {"type": "static", "num_workers": workers}, **serve},
    )


async def wait_path(path):
    async with asyncio.timeout(10):
        while not path.exists():
            await asyncio.sleep(0.01)


async def test_process_workers_episode_payload_cancel_and_recovery(tmp_path):
    pool = EnvServerPool(config(startup_timeout=10, shutdown_timeout=3))
    serving = asyncio.create_task(pool.run())
    client = EnvClient(pool.address, timeout=10, cancel_timeout=5)
    try:
        # A listening broker is not readiness until workers enter env.serving().
        description = await client.describe()
        assert not description.success and description.ready_workers == 0
        await client.wait_for_server_startup(timeout=15)
        description = await client.describe()
        assert description.success and description.ready_workers == 2
        assert description.env_config["attempts"] == 2
        episodes = await asyncio.gather(*(client.run(**request()) for _ in range(4)))
        assert len({episode.traces[0].info["pid"] for episode in episodes}) == 2
        for episode in episodes:
            assert isinstance(episode, WireEpisode) and episode.ok
            assert len(episode.traces) == 2
            assert [trace.reward for trace in episode.traces] == [0.5, 0.5]
            assert episode.task.data.value == 17
            urls = [trace.info[PAYLOAD_INFO_KEY]["url"] for trace in episode.traces]
            await materialize_episode(episode)
            for trace in episode.traces:
                node = trace.nodes[0]
                np.testing.assert_array_equal(node.token_ids, [11, 17])
                np.testing.assert_allclose(node.logprobs, [-0.125], rtol=0, atol=0)
                np.testing.assert_array_equal(node.routed_experts, np.arange(4).reshape(2, 1, 2))
                np.testing.assert_array_equal(node.indexer_topk, np.arange(6).reshape(2, 1, 3))
                np.testing.assert_allclose(node.top_logprobs.logprobs, [-0.125, -1.5])
            args = Namespace(
                reward_key=None,
                use_sampling_support_replay=True,
                rollout_top_logprobs_num=2,
                use_rollout_routing_replay=True,
                use_rollout_indexer_replay=True,
            )
            samples = [
                trace_to_samples(args, trace, group_index=4, index_start=i, rollout_id=i)[0]
                for i, trace in enumerate(episode.traces)
            ]
            assert [sample.reward for sample in samples] == [0.5, 0.5]
            assert [sample.rollout_id for sample in samples] == [0, 1]
            for sample in samples:
                assert sample.tokens == [11, 17]
                assert sample.rollout_log_probs == [-0.125]
                assert sample.loss_mask == [1]
                assert sample.rollout_topk_token_ids.tolist() == [[17, 2]]
                assert len(sample.rollout_sampling_mask) == 1
                np.testing.assert_array_equal(sample.rollout_routed_experts, [[[0, 1]]])
                np.testing.assert_array_equal(sample.rollout_indexer_topk, [[[0, 1, 2]]])
                sample.validate()
            async with httpx.AsyncClient(trust_env=False) as http:
                for url in urls:
                    assert (await http.get(url)).status_code == 404
        marker = tmp_path / "cancelled"
        running = asyncio.create_task(client.run(**request("wait", marker=str(marker))))
        await wait_path(Path(str(marker) + ".started"))
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert marker.read_text() == "cleaned"  # cancellation waits for finally
        assert not client._pending
        marker = tmp_path / "abort-failed"
        running = asyncio.create_task(client.run(**request("abort_failure", marker=str(marker))))
        await wait_path(Path(str(marker) + ".started"))
        request_id = next(key for key, pending in client._pending.items() if pending.method == "run")
        running.cancel()
        with pytest.raises(EnvDrainError):
            await running
        # A late cancel must retain the same failed-cleanup standing, not a clean miss.
        with pytest.raises(EnvDrainError):
            await client.cancel_and_wait(request_id)
        with pytest.raises(EnvDrainError, match="worker lost"):
            await client.run(**request("crash"))
        await client.wait_for_server_startup(timeout=15)
        assert (await client.describe()).ready_workers == 2
        assert (await client.run(**request())).ok
    finally:
        await client.close()
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serving
    assert all(not worker.process.is_alive() for worker in pool.workers)


async def test_core_training_arrays_stream_without_auxiliary_payload():
    server = EnvServer(resolve_env_config(config().env), "tcp://127.0.0.1:0", 8)
    serving = asyncio.create_task(server.run())
    client = EnvClient(server.address, timeout=10)
    try:
        await client.wait_for_server_startup()
        episode = await client.run(**request("no_aux"))
        assert episode.ok and len(episode.traces) == 2
        # Basic training evidence rides the delta stream. HTTP payloads are only
        # needed when optional replay/support/top-k arrays were actually returned.
        assert all(PAYLOAD_INFO_KEY not in trace.info for trace in episode.traces)
        assert server._hook._server is None
        await materialize_episode(episode)
        args = Namespace(
            reward_key=None,
            use_sampling_support_replay=False,
            rollout_top_logprobs_num=0,
            use_rollout_routing_replay=False,
            use_rollout_indexer_replay=False,
        )
        for index, trace in enumerate(episode.traces):
            sample = trace_to_samples(args, trace, group_index=0, index_start=index)[0]
            assert sample.tokens == [11, 17]
            assert sample.loss_mask == [1]
            assert sample.rollout_log_probs == [-0.125]
            assert sample.reward == 0.5
            assert sample.rollout_topk_token_ids is None
            assert sample.rollout_sampling_mask is None
            assert sample.rollout_routed_experts is None
            assert sample.rollout_indexer_topk is None
    finally:
        await client.close()
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serving


async def test_replica_balance_config_mismatch_and_timeout_cleanup(tmp_path):
    env_config = resolve_env_config(config().env)
    servers = [EnvServer(env_config, "tcp://127.0.0.1:0", 8) for _ in range(2)]
    tasks = [asyncio.create_task(server.run()) for server in servers]
    clients = EnvClientPool([server.address for server in servers], timeout=3, cancel_timeout=2)
    try:
        await clients.wait_for_server_startup()
        results = await asyncio.gather(*(clients.run(**request()) for _ in range(6)))
        assert all(result.ok for result in results)
        assert all(server._delivered for server in servers)
        for result in results:
            await materialize_episode(result)
        marker = tmp_path / "timeout"
        with pytest.raises(TimeoutError):
            await clients.clients[0].run(**request("wait", marker=str(marker)), timeout=0.1)
        assert marker.read_text() == "cleaned"
        servers[1].env_config = {"changed": True}
        with pytest.raises(ValueError, match="different resolved"):
            await clients.describe()
    finally:
        await clients.close()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def test_payload_store_bound_ttl_and_configuration():
    store = PayloadServer(PayloadConfig(ttl_seconds=0.03, max_bytes=4))
    try:
        key = store.put(b"1234")
        assert store.get(key) == b"1234"
        with pytest.raises(BufferError):
            store.put(b"x")
        time.sleep(0.05)
        assert store.get(key) is None
        key = store.put(b"abcd")
        store.delete(key)
        store.delete(key)
        assert store.get(key) is None
    finally:
        store.close()
    with pytest.raises(ValueError, match="advertised_host"):
        PayloadConfig(bind_host="0.0.0.0")


def test_missing_trace_delta_is_never_silent():
    assembly = EpisodeAssembly()
    with pytest.raises(RuntimeError, match="missing trace"):
        assembly.finish({}, [TraceSummary(id="lost", nodes=1, calls=0)])


async def test_cancel_waits_for_payload_thread_and_close_drains_runs(tmp_path):
    server = EnvServer(resolve_env_config(config().env), "tcp://127.0.0.1:0", 8)
    serving = asyncio.create_task(server.run())
    client = EnvClient(server.address, timeout=5, cancel_timeout=3)
    entered, release = threading.Event(), threading.Event()
    original = server._hook.keep

    def slow_keep(episode, sampling):
        entered.set()
        assert release.wait(3)
        original(episode, sampling)

    server._hook.keep = slow_keep
    try:
        running = asyncio.create_task(client.run(**request()))
        assert await asyncio.to_thread(entered.wait, 3)
        running.cancel()
        await asyncio.sleep(0.03)
        # Cancellation cannot acknowledge while a CPU keep may still publish blobs.
        assert not running.done()
        for remote in server._running.values():
            remote.cancel()  # a concurrent shutdown/second cancel must also drain
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert not server._hook.server._blobs
        server._hook.keep = original
        marker = tmp_path / "client-close"
        running = asyncio.create_task(client.run(**request("wait", marker=str(marker))))
        await wait_path(Path(str(marker) + ".started"))
        await client.close()
        with pytest.raises(RuntimeError, match="Cancelled"):
            await running
        assert marker.read_text() == "cleaned"
    finally:
        release.set()
        await client.close()
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serving


async def test_payload_corruption_is_fatal_and_still_deleted():
    server = EnvServer(resolve_env_config(config().env), "tcp://127.0.0.1:0", 8)
    serving = asyncio.create_task(server.run())
    client = EnvClient(server.address)
    try:
        episode = await client.run(**request())
        reference = episode.traces[0].info[PAYLOAD_INFO_KEY]
        store = server._hook.server
        created, blob = store._blobs[reference["id"]]
        store._blobs[reference["id"]] = (created, blob[:-1] + bytes([blob[-1] ^ 1]))
        with pytest.raises(ValueError, match="checksum"):
            await materialize_episode(episode)
        assert not store._blobs
    finally:
        await client.close()
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serving


async def test_elastic_pool_scales_and_waits_for_worker_readiness():
    conf = config()
    conf.serve.pool = ElasticPoolConfig(max_workers=2, multiplex=1)
    pool = EnvServerPool(conf)
    serving = asyncio.create_task(pool.run())
    client = EnvClient(pool.address, timeout=10)
    try:
        await client.wait_for_server_startup(timeout=15)
        assert (await client.describe()).ready_workers == 1
        episode = await client.run(**request())
        await materialize_episode(episode)
        async with asyncio.timeout(15):
            while (await client.describe()).ready_workers < 2:
                await asyncio.sleep(0.02)
        assert len(pool.workers) == 2
    finally:
        await client.close()
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serving


@pytest.mark.parametrize("mode", ["wait", "abort_failure"])
async def test_cli_publishes_ready_address_and_sigterm_cleans_up(tmp_path, mode):
    address_file = tmp_path / "ready.address"
    conf = config(workers=1, shutdown_timeout=3)
    conf.address_file = address_file
    config_file = tmp_path / "server.json"
    config_file.write_text(conf.model_dump_json())
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), str(Path.cwd())])
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "miles.env_server",
        "--config",
        str(config_file),
        env=environment,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    client = None
    try:
        await wait_path(address_file)
        client = EnvClient(address_file.read_text().strip(), timeout=5, cancel_timeout=2)
        assert (await client.describe()).ready_workers == 1
        probe = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "miles.env_server",
            "--health-check",
            client.address,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, errors = await probe.communicate()
        assert probe.returncode == 0, errors.decode()
        marker = tmp_path / "shutdown"
        running = asyncio.create_task(client.run(**request(mode, marker=str(marker))))
        await wait_path(Path(str(marker) + ".started"))
        process.terminate()
        await asyncio.wait_for(process.wait(), 8)
        assert marker.read_text() == "cleaned"
        with pytest.raises(EnvDrainError if mode == "abort_failure" else RuntimeError):
            await running
        assert process.returncode == 0, (await process.stderr.read()).decode()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        if client is not None:
            await client.close()


async def test_repeated_materialization_cancel_still_deletes_payloads(monkeypatch):
    import miles.env_server.payload as payload_module

    server = EnvServer(resolve_env_config(config().env), "tcp://127.0.0.1:0", 8)
    serving = asyncio.create_task(server.run())
    client = EnvClient(server.address)
    entered, release = asyncio.Event(), asyncio.Event()
    original = payload_module._delete

    async def slow_delete(http, reference):
        entered.set()
        await release.wait()
        await original(http, reference)

    monkeypatch.setattr(payload_module, "_delete", slow_delete)
    try:
        episode = await client.run(**request())
        materializing = asyncio.create_task(materialize_episode(episode))
        await entered.wait()
        materializing.cancel()
        await asyncio.sleep(0)
        materializing.cancel()
        await asyncio.sleep(0)
        assert not materializing.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await materializing
        assert not server._hook.server._blobs
    finally:
        release.set()
        await client.close()
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serving


def test_cli_overrides_are_applied_before_validation(tmp_path):
    from miles.env_server.__main__ import _config

    path = tmp_path / "server.json"
    path.write_text('{"env": {}, "serve": {"address": "tcp://0.0.0.0:5555", "payload": {"bind_host": "0.0.0.0"}}}')
    with pytest.raises(ValueError, match="advertised_host"):
        _config(path, address=None, payload_advertised_host=None)
    resolved = _config(path, address="tcp://127.0.0.1:7777", payload_advertised_host="10.0.1.9")
    assert resolved.serve.address == "tcp://127.0.0.1:7777"
    assert resolved.serve.payload.advertised_host == "10.0.1.9"
    assert resolved.serve.payload.bind_host == "0.0.0.0"


async def test_readiness_probe_imports_no_environment_runtime():
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import sys; from miles.env_server.health import ready; "
        "assert not {'torch', 'ray', 'prime_rl', 'verifiers', 'renderers'} & sys.modules.keys()",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, errors = await process.communicate()
    assert process.returncode == 0, errors.decode()


def test_config_fingerprint_survives_pydantic_default_number_coercion():
    first = resolve_env_config(config().env).model_dump(mode="json", serialize_as_any=True)
    roundtripped = resolve_env_config(first).model_dump(mode="json", serialize_as_any=True)
    assert config_fingerprint(first) == config_fingerprint(roundtripped)
    assert config_fingerprint({"a": [3600, {"b": 0}]}) == config_fingerprint({"a": [3600.0, {"b": 0.0}]})
    assert config_fingerprint({"a": False}) != config_fingerprint({"a": 0})
