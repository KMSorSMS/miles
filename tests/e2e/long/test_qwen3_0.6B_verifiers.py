import math
import os
import shlex
import shutil
import site
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.e2e.long.verifiers_contract import verify_sdk_contract

from miles.utils.external_utils import command_utils

register_cuda_ci(
    est_time=300, suite="stage-c-2-gpu-h200", labels=["short"], hardware=["hopper", "blackwell"], num_gpus=2
)
register_rocm_ci(est_time=900, suite="nightly-stage-c-2-gpu-mi350", labels=["short"])

MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 2
MODEL_DIR = Path(os.environ.get("MILES_E2E_MODEL_DIR", "/root/models"))
MEGATRON_PATH = Path(os.environ.get("MILES_E2E_MEGATRON_PATH", "/root/Megatron-LM"))
RUN_DIR = Path(os.environ.get("MILES_E2E_RUN_DIR", "/tmp/miles-verifiers-e2e"))
VERIFIERS_DIR = Path(os.environ.get("MILES_E2E_VERIFIERS_DIR", str(RUN_DIR / "verifiers-source")))
ADAPTER_DIR = Path(command_utils.repo_base_dir) / "examples" / "experimental" / "verifiers"
VERIFIERS_VENV = RUN_DIR / "verifiers-venv"
VERIFIERS_SITE_PACKAGES = (
    VERIFIERS_VENV / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
)
CODE_GOLF_DIR = RUN_DIR / "environments" / "code_golf"


def prepare():
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu(f"mkdir -p {MODEL_DIR} {RUN_DIR}")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir {MODEL_DIR}/{MODEL_NAME}")
    U.exec_command_cpu(f"uv venv --clear --seed --python {sys.executable} --system-site-packages {VERIFIERS_VENV}")
    launcher_site = site.getsitepackages()
    pth_lines = list(launcher_site)
    # --system-site-packages exposes the base interpreter's packages. When the launcher itself runs
    # from a venv (the ROCm image's /opt/venv), its site-packages arrive only through this file, and
    # a bare path line skips the .pth files there, which is how the editable SGLang install is found.
    if sys.prefix != sys.base_prefix:
        pth_lines += [f"import site; site.addsitedir({path!r})" for path in launcher_site]
    (VERIFIERS_SITE_PACKAGES / "launcher-site.pth").write_text("\n".join(pth_lines) + "\n")
    U.exec_command_cpu(
        # These forks have no release tags; scope version overrides to their builds.
        "env SETUPTOOLS_SCM_PRETEND_VERSION_FOR_RENDERERS=0.2.1.dev0 "
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VERIFIERS=0.4.2.dev0 "
        f"{VERIFIERS_VENV}/bin/python -m pip install "
        f"-r {command_utils.repo_base_dir}/examples/experimental/verifiers/requirements.txt"
    )
    if not VERIFIERS_DIR.exists():
        U.exec_command_cpu(
            shlex.join(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    "codex/distributed-verifiers-sglang",
                    "https://github.com/KMSorSMS/verifiers.git",
                    str(VERIFIERS_DIR),
                ]
            )
        )
        # Match the source pinned by the example's requirements, including its
        # taskset package; do not let a later branch update change this recipe.
        requirements = (ADAPTER_DIR / "requirements.txt").read_text()
        revision = next(
            line.rsplit("@", 1)[1] for line in requirements.splitlines() if line.startswith("verifiers @ git+")
        )
        U.exec_command_cpu(shlex.join(["git", "-C", str(VERIFIERS_DIR), "fetch", "--depth", "1", "origin", revision]))
        U.exec_command_cpu(shlex.join(["git", "-C", str(VERIFIERS_DIR), "checkout", "--detach", "FETCH_HEAD"]))
    shutil.copytree(VERIFIERS_DIR / "environments" / "code_golf", CODE_GOLF_DIR, dirs_exist_ok=True)
    U.exec_command_cpu(
        shlex.join([str(VERIFIERS_VENV / "bin/python"), "-m", "pip", "install", "--no-deps", str(CODE_GOLF_DIR)])
    )
    verify_sdk_contract(
        execute=U.exec_command_cpu,
        venv=VERIFIERS_VENV,
        repo_root=Path(command_utils.repo_base_dir),
        report_path=RUN_DIR / "sdk-contract.xml",
        pythonpath=f"{VERIFIERS_SITE_PACKAGES}:{command_utils.repo_base_dir}:{MEGATRON_PATH}:{os.environ.get('PYTHONPATH', '')}",
    )
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst=str(MODEL_DIR),
        hf_checkpoint=str(MODEL_DIR / MODEL_NAME),
        megatron_path=str(MEGATRON_PATH),
    )


def execute():
    U = command_utils.default_config().create_backend()
    config_path = RUN_DIR / "code-golf.toml"
    _write_remote_configs(config_path)
    dump_dir = RUN_DIR / "dump"

    train_args = " ".join(
        [
            f"--hf-checkpoint {MODEL_DIR}/{MODEL_NAME}",
            "--sglang-tokenizer-path Qwen/Qwen3-0.6B",
            f"--ref-load {MODEL_DIR}/{MODEL_NAME}_torch_dist",
            "--rollout-function-path verifiers_rollout.generate_rollout",
            "--disable-rollout-global-dataset",
            "--num-rollout 1",
            "--rollout-batch-size 3",
            "--n-samples-per-prompt 4",
            "--over-sampling-batch-size 3",
            "--rollout-max-response-len 512",
            "--rollout-max-context-len 2048",
            "--rollout-temperature 0.8",
            "--global-batch-size 12",
            "--balance-data",
            "--advantage-estimator grpo",
            "--entropy-coef 0.0",
            "--eps-clip 0.2",
            "--eps-clip-high 0.28",
            "--optimizer adam",
            "--lr 1e-6",
            "--lr-decay-style constant",
            "--weight-decay 0.1",
            "--adam-beta1 0.9",
            "--adam-beta2 0.98",
            "--no-gradient-accumulation-fusion",
            "--rollout-num-gpus-per-engine 1",
            "--sglang-mem-fraction-static 0.6",
            "--sglang-enable-metrics",
            "--tensor-model-parallel-size 1",
            "--pipeline-model-parallel-size 1",
            "--context-parallel-size 1",
            "--use-dynamic-batch-size",
            "--max-tokens-per-gpu 4096",
            "--attention-backend flash",
            "--actor-num-nodes 1",
            f"--actor-num-gpus-per-node {NUM_GPUS}",
            "--colocate",
            f"--dump-details {dump_dir}",
            command_utils.get_default_wandb_args(__file__),
        ]
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={
            "VF_LOG_LEVEL": "INFO",
            "PYTHONPATH": (
                f"{VERIFIERS_SITE_PACKAGES}:{CODE_GOLF_DIR}:{MEGATRON_PATH}:{ADAPTER_DIR}:{command_utils.repo_base_dir}"
            ),
            "VERIFIERS_CONFIG": str(config_path),
        },
        megatron_path=str(MEGATRON_PATH),
    )
    verify(dump_dir)


def _write_remote_configs(config_path: Path):
    # The env owns the four jointly scored attempts; one RPC returns the entire
    # episode. Both client and workers validate the same environment definition.
    env = (
        "[env]\nattempts = 4\nmax_concurrent_agents = 4\n"
        '\n[env.taskset]\nid = "code-golf"\n'
        '\n[env.golfer.harness]\nid = "null"\n'
        # Trusted test-only programs run inside the isolated GPU test container.
        '\n[env.golfer.runtime]\ntype = "subprocess"\n'
        "\n[env.golfer.timeout]\nrollout = 300\n"
    )
    config_path.write_text(
        env + '\n[remote]\nserver_config = "server.toml"\nepisodes_per_prompt = 1\neval_episodes_per_prompt = 1\n'
    )
    (config_path.parent / "server.toml").write_text(env + '\n[serve.pool]\ntype = "static"\nnum_workers = 2\n')


def verify(dump_dir: Path):
    samples = torch.load(dump_dir / "rollout_data" / "0.pt", weights_only=False)["samples"]

    assert len(samples) == 12
    assert Counter(sample["group_index"] for sample in samples) == {0: 4, 1: 4, 2: 4}
    assert {sample["status"] for sample in samples} <= {"completed", "truncated"}
    assert any(sample["status"] == "completed" for sample in samples)
    assert all(math.isfinite(sample["reward"]) for sample in samples)
    assert all(
        len(sample["rollout_log_probs"]) == len(sample["loss_mask"]) == sample["response_length"] for sample in samples
    )
    assert all(math.isfinite(value) for sample in samples for value in sample["rollout_log_probs"])

    for sample in samples:
        metadata = sample["metadata"]["verifiers"]
        assert set(metadata["rewards"]) == {"correct", "fastest", "most_concise"}
        assert metadata["metrics"]["passed"] in {0.0, 1.0}
        assert math.isfinite(metadata["metrics"]["latency"])
        assert "error" not in metadata

    assert any(sample["metadata"]["verifiers"]["metrics"]["passed"] for sample in samples)
    for group_index in range(3):
        group = [sample for sample in samples if sample["group_index"] == group_index]
        passing = [sample for sample in group if sample["metadata"]["verifiers"]["metrics"]["passed"]]
        fastest = min((sample["metadata"]["verifiers"]["metrics"]["latency"] for sample in passing), default=None)
        for sample in group:
            metadata = sample["metadata"]["verifiers"]
            expected = 0.5 if metadata["metrics"]["passed"] and metadata["metrics"]["latency"] == fastest else 0.0
            assert metadata["rewards"]["fastest"] == pytest.approx(expected)


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
