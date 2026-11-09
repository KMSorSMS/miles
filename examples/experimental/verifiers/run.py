"""Train Qwen3-0.6B using complete CodeGolf episodes on CPU env-server workers.

Defaults reproduce the two-GPU smoke configuration: 3 GRPO steps against the
`code-golf` taskset. Point --verifiers-config at a distributed adapter TOML.

Install the matching renderers/verifiers forks and the code_golf package on
the client and CPU workers. Runtime credentials belong on the workers.

Usage:
    python run.py
    python run.py --verifiers-config /path/to/verifiers.toml --num-rollout 50
    python run.py --eval-interval 5      # evaluate the whole taskset every 5 rollouts
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import typer

from miles.utils.external_utils import command_utils

SCRIPT_DIR = Path(__file__).resolve().parent
LEGACY_ROLLOUT_ENV = "MILES_USE_LEGACY_ROLLOUT_V1"


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    megatron_model_type: str = "qwen3-0.6B"
    num_gpus_per_node: int = 2
    megatron_path: str = "/root/Megatron-LM"

    # Paths
    skip_prepare: bool = False
    model_name: str = "Qwen3-0.6B"
    hf_checkpoint: str = "/root/models/Qwen3-0.6B"
    # Renderers resolves its renderer from the model identity, which a local
    # snapshot path does not carry; keep the registered id here.
    sglang_tokenizer_path: str = "Qwen/Qwen3-0.6B"
    ref_load: str = "/root/models/Qwen3-0.6B_torch_dist"
    save_dir: str = "/root/Qwen3-0.6B_verifiers/"

    # Adapter config. A matching local server config is generated when absent.
    verifiers_config: str = "/root/verifiers-code-golf.toml"
    taskset_id: str = "code-golf"
    env_server_addresses: str = ""
    """Comma-separated predeployed CPU server addresses; empty starts an owned pool."""
    env_server_workers: int = 2

    # Training settings (smoke scale)
    rollout_max_response_len: int = 512
    rollout_max_context_len: int = 2048
    num_rollout: int = 3
    rollout_batch_size: int = 3
    n_samples_per_prompt: int = 4
    global_batch_size: int = 12

    # Evaluation over the whole taskset, every N training rollouts. 0 disables it.
    # Group rewards need at least two rollouts per task to rank.
    eval_interval: int = 0
    n_samples_per_eval_prompt: int = 4


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {args.hf_checkpoint}")
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=str(Path(args.hf_checkpoint).parent),
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def _write_configs(args: ScriptArgs, config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    env = f"[env.taskset]\nid = {json.dumps(args.taskset_id)}\n"
    remote = "[remote]\n"
    if args.taskset_id == "code-golf":
        if args.eval_interval and args.n_samples_per_eval_prompt % args.n_samples_per_prompt:
            raise ValueError("CodeGolf eval sample count must be a multiple of the complete episode's attempts")
        env = f"[env]\nattempts = {args.n_samples_per_prompt}\n" + env
        env += '\n[env.golfer.harness]\nid = "null"\n'
        remote += "episodes_per_prompt = 1\n"
        remote += f"eval_episodes_per_prompt = {max(1, args.n_samples_per_eval_prompt // args.n_samples_per_prompt)}\n"
    addresses = [value.strip() for value in args.env_server_addresses.split(",") if value.strip()]
    if addresses:
        remote += f"addresses = {json.dumps(addresses)}\n"
    else:
        server_path = config_path.with_name(config_path.stem + "-server.toml")
        server_path.write_text(env + f'\n[serve.pool]\ntype = "static"\nnum_workers = {args.env_server_workers}\n')
        remote += f"server_config = {json.dumps(str(server_path.resolve()))}\n"
    client = '\n[client]\nbackend = "sglang"\n' f"renderer_model_name = {json.dumps(args.sglang_tokenizer_path)}\n"
    config_path.write_text(env + "\n" + remote + client)


def execute(args: ScriptArgs):
    U = args.create_backend()
    config_path = Path(args.verifiers_config)
    if not config_path.exists():
        _write_configs(args, config_path)

    extra_env_vars = {
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{command_utils.repo_base_dir}",
        "VERIFIERS_CONFIG": str(config_path),
    }
    if LEGACY_ROLLOUT_ENV in os.environ:
        extra_env_vars[LEGACY_ROLLOUT_ENV] = os.environ[LEGACY_ROLLOUT_ENV]
    extra_env_vars = command_utils.resolve_extra_env_vars(extra_env_vars, args)

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--sglang-tokenizer-path {args.sglang_tokenizer_path} "
        f"--ref-load {args.ref_load} "
        f"--save {args.save_dir} "
        "--save-interval 1000 "
    )

    # Verifiers owns the taskset, so Miles loads no prompt data; the rollout
    # function plug-point selects the adapter, which resolves as a bare module
    # because PYTHONPATH carries this directory into the rollout actor.
    rollout_fn = (
        "verifiers_rollout.generate_rollout"
        if bool(int(extra_env_vars.get(LEGACY_ROLLOUT_ENV, "0")))
        else "verifiers_rollout.VerifiersRolloutFn"
    )
    rollout_args = (
        f"--rollout-function-path {rollout_fn} "
        "--disable-rollout-global-dataset "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--over-sampling-batch-size {args.rollout_batch_size} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--rollout-max-context-len {args.rollout_max_context_len} "
        "--rollout-temperature 0.8 "
        f"--rollout-num-gpus-per-engine 1 "
    )

    # Workaround: the taskset is the evaluation set, but Miles asserts that eval
    # datasets are configured whenever --eval-interval is set, so name the taskset
    # and point the placeholder at the config it is defined in -- the adapter serves
    # eval, so the built-in loader never opens this path. Worth replacing with a
    # Miles-side fix once a second rollout function owns its evaluation set.
    eval_args = (
        (
            f"--eval-interval {args.eval_interval} "
            f"--n-samples-per-eval-prompt {args.n_samples_per_eval_prompt} "
            f"--eval-prompt-data verifiers-taskset {config_path} "
        )
        if args.eval_interval
        else ""
    )

    grpo_args = (
        "--advantage-estimator grpo "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
        "--no-gradient-accumulation-fusion "
    )

    sglang_args = "--sglang-mem-fraction-static 0.6 --sglang-enable-metrics "

    misc_args = (
        "--attention-backend flash "
        "--colocate "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args}{rollout_args}{eval_args}{grpo_args}{optimizer_args}{perf_args}{sglang_args}{misc_args}"
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
