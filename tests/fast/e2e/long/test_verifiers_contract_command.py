import runpy
import shlex
from pathlib import Path

import pytest
import tomllib
from tests.e2e.long.verifiers_contract import verify_sdk_contract


class TestVerifiersContractCommand:
    def test_gpu_recipe_runs_joint_attempts_inside_two_cpu_workers(self, tmp_path: Path) -> None:
        """The GPU recipe's CPU-service config preserves one complete group per RPC."""
        root = Path(__file__).resolve().parents[4]
        recipe = runpy.run_path(str(root / "tests/e2e/long/test_qwen3_0.6B_verifiers.py"))
        client_path = tmp_path / "client.toml"
        recipe["_write_remote_configs"](client_path)
        client = tomllib.loads(client_path.read_text())
        server = tomllib.loads((tmp_path / "server.toml").read_text())

        assert client["env"] == server["env"]
        assert client["env"]["taskset"]["id"] == "code-golf"
        assert client["env"]["attempts"] == 4
        assert client["env"]["golfer"]["runtime"]["type"] == "subprocess"
        assert client["remote"]["episodes_per_prompt"] == 1
        assert client["remote"]["eval_episodes_per_prompt"] == 1
        assert server["serve"]["pool"] == {"type": "static", "num_workers": 2}

    def test_sdk_witnesses_run_in_the_requested_environment_with_shell_safe_paths(self, tmp_path: Path) -> None:
        """SDK checks use their dedicated environment and preserve paths containing spaces as single arguments."""
        report = tmp_path / "result report.xml"
        root = tmp_path / "source tree"
        venv = tmp_path / "sdk environment"
        calls: list[list[str]] = []

        def execute(command: str) -> None:
            calls.append(shlex.split(command))
            report.write_text("<testsuite>" + "<testcase/>" * 3 + "</testsuite>")

        verify_sdk_contract(
            execute=execute,
            venv=venv,
            repo_root=root,
            report_path=report,
            pythonpath="sdk path:source path",
        )

        test_file = root / "tests/fast/examples/experimental/verifiers/test_verifiers_rollout.py"
        assert calls == [
            [
                "env",
                "CUDA_VISIBLE_DEVICES=",
                f"VIRTUAL_ENV={venv}",
                "PYTHONPATH=sdk path:source path",
                "uv",
                "run",
                "--no-project",
                "--active",
                "python",
                "-m",
                "pytest",
                f"{test_file}::test_wire_trace_preserves_training_arrays",
                f"{test_file}::test_episode_finalizes_joint_rewards",
                "-v",
                f"--junitxml={report}",
            ]
        ]

    def test_failed_sdk_execution_is_propagated_without_a_stale_report_pass(self, tmp_path: Path) -> None:
        """A previous passing report cannot hide failure of the current SDK invocation."""
        report = tmp_path / "report.xml"
        report.write_text("<testsuite>" + "<testcase/>" * 3 + "</testsuite>")

        def execute(command: str) -> None:
            raise RuntimeError("SDK process failed")

        with pytest.raises(RuntimeError, match="SDK process failed"):
            verify_sdk_contract(execute=execute, venv=tmp_path, repo_root=tmp_path, report_path=report, pythonpath="")
