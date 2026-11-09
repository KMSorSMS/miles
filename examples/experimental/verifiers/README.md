# Distributed Verifiers rollouts

Miles sends complete task data to CPU environment servers owned by Miles. Each
worker runs a complete Verifiers Episode, including `Env.finalize()`, and returns
its traces and training arrays. Miles retains sampling constraints, filtering,
advantage calculation, training and weight updates. No component imports or
installs `prime_rl`.

- **Miles:** `miles/env_server/` owns RPC, server pools, worker lifecycle, trace
  deltas and array transfer. `verifiers_rollout.py` is the training client;
  `verifiers_samples.py` converts traces to samples.
- **Verifiers:** owns environment execution, runtimes, interception and scoring.
  Its serializable `TrainClientConfig` selects inference on CPU workers.
- **renderers:** a separate Python library, not a service. Its SGLang backend
  calls native `/generate` with rendered token IDs and preserves actual output
  IDs and probabilities. The adapter-local `MilesSGLangTransport` and custom
  `TrainClient` have been removed.

## Install

Requires Python 3.11+ and the matching forks in `requirements.txt`. On an existing
Miles training image, install the integration dependencies:

```bash
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_RENDERERS=0.2.1.dev0
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VERIFIERS=0.4.2.dev0
python -m pip install -r examples/experimental/verifiers/requirements.txt
```

These fork builds use explicit development versions because the forks do not
carry upstream release tags. The distribution-specific settings affect only
these two packages; they do not alter dependency constraints or publish releases.
The Git revisions in the requirements identify the actual source being installed.
The default CodeGolf taskset is included.

Verifiers requires OpenAI >=2.54,<3. Some SGLang images declare an older exact
OpenAI pin. Check that image's compatibility before installing; lowering the
Verifiers requirement does not resolve its SDK usage. The native-backend test
passed with OpenAI 2.54.0 and SGLang 0.5.22.dev42; this does not qualify every
SGLang image. Preserve the image's GPU wheels when resolving dependencies.

For development, install the sibling checkouts instead:

```bash
python -m pip install -e ../renderers -e ../verifiers pyzmq msgpack
python -m pip install --no-deps ../verifiers/environments/code_golf
```

Install the same taskset/environment packages on the rollout actor and every CPU
worker. CPU deployments need only the example requirements and this Miles
checkout on `PYTHONPATH`; they do not need Miles' GPU/Ray dependency set. The CPU
image below follows that installation scheme.

## Configure complete episodes

For CodeGolf, write `adapter.toml`:

```toml
[env]
attempts = 4
max_concurrent_agents = 8

[env.taskset]
id = "code-golf"

[env.golfer.harness]
id = "null"

[remote]
addresses = ["tcp://cpu-a:5555", "tcp://cpu-b:5555"]
episodes_per_prompt = 1
eval_episodes_per_prompt = 1
request_timeout = 3600
cancel_timeout = 60

[client]
backend = "sglang"
renderer_model_name = "Qwen/Qwen3-0.6B"
# Optional: DNS name of the Miles router reachable from CPU workers.
# base_url = "http://miles-inference:30000/v1"

[client.backend_options]
compact_logprobs = true
# Default: discover router workers and cancel the request on its engine.
abort_strategy = "workers"
# For an explicit base_url pointing directly to ONE engine, use "endpoint".
```

One RPC returns an Episode with four jointly scored agent runs. Set Miles
`--n-samples-per-prompt 4` and `--n-samples-per-eval-prompt 4`. An environment
producing one trainable run per Episode can omit `episodes_per_prompt`; the
adapter then sends `n_samples_per_prompt` independent RPCs. Joint scoring always
stays within its original Episode. The adapter validates the total trainable
trace count; graph branches retain one logical rollout identity and do not
multiply the GRPO sample count.

`[select]` supports Verifiers task selection. Training cycles the selected tasks;
evaluation covers them. Task order/fingerprint, cursor, sample/rollout IDs and
seed cursor are checkpointed. The client sends full serialized `TaskData`,
including its index. Workers reconstruct its task type; they do not load the
whole dataset per request.

The client and servers must have matching resolved `[env]` configurations;
startup rejects incompatible configuration fingerprints and protocol versions.
Runtime credentials belong on CPU workers. The default is Verifiers' Prime
sandbox provider, which is independent of PrimeRL. Other Verifiers runtimes use
their existing configuration.

## Deploy CPU servers

Copy the exact `[env]` configuration above to `server.toml`, then append:

```toml
[serve]
address = "tcp://0.0.0.0:5555"
max_concurrent = 128
restart_workers = true
max_worker_restarts = 3

[serve.pool]
type = "static"
num_workers = 4

[serve.payload]
bind_host = "0.0.0.0"
advertised_host = "cpu-a"
port = 0
ttl_seconds = 3600
max_bytes = 4294967296
```

```bash
python -m miles.env_server --config server.toml
python -m miles.env_server --health-check tcp://cpu-a:5555
```

Start services on additional machines with their own reachable payload hostname,
then add their RPC addresses to `remote.addresses`. The client balances by
in-flight count. Readiness requires workers to enter `env.serving()`. For an
elastic pool, replace `[serve.pool]` with `type="elastic"`, `max_workers=16`,
`multiplex=128`.

Each worker serves array payloads over HTTP on a separate ephemeral port. Both
RPC and payload ports must be reachable from the rollout actor. The client
validates lengths/checksums, restores arrays and deletes payloads. TTL and byte
limits bound abandoned storage. Use private trusted networking: this protocol
does not provide authentication or TLS.

`env-server.k8s.yaml` supplies a headless Service and two StatefulSet replicas,
each with four CPU workers. Build from the Miles checkout, replace the image
and environment configuration in the manifest, then deploy:

```bash
docker build -f examples/experimental/verifiers/Dockerfile.env-server -t YOUR_REGISTRY/miles-env-server:integration .
kubectl apply -f examples/experimental/verifiers/env-server.k8s.yaml
```

The client addresses are `tcp://miles-verifiers-env-0.miles-verifiers-env:5555`
and `tcp://miles-verifiers-env-1.miles-verifiers-env:5555`. Payloads advertise
Pod IPs; allow Pod-to-Pod traffic to worker payload ports. Supply runtime secrets
through the deployment. The manifest contains no credentials.

For a managed local pool, replace `remote.addresses` with
`server_config = "server.toml"`. Miles starts the same CLI, waits for readiness
and owns process-group cleanup. External servers stay owned by their deployment.
SIGTERM drains active work. Timeout/cancellation drains only the caller's
requests, never all engine work. Unconfirmed GPU cleanup and worker loss raise
errors rather than claiming successful cancellation.

## Run Miles

```bash
python examples/experimental/verifiers/run.py --verifiers-config /path/to/adapter.toml
```

Without an existing config, the launcher creates CodeGolf with four attempts
and a managed two-worker pool. `--env-server-addresses` supplies comma-separated
predeployed endpoints when generating the config. `--eval-interval N` evaluates
the selected taskset. Sampling/length limits, full-weight/LoRA training and
weight synchronization remain owned by Miles. Training requests cannot silently
change resolved sampling constraints or replay features. Evaluation uses its
own sampling without training-only replay collection.

An explicitly configured Miles reward model overrides finalized env rewards.
It scores one representative per logical rollout and broadcasts that reward to
its graph branches. Model identity can be set through `client.renderer_model_name`
when a local checkpoint directory has no registered renderer name.

## Training data and limits

- Actual token IDs, logprobs and loss masks survive RPC without re-tokenizing
  generated output. Shared graph prefixes receive loss once.
  These core arrays travel in the trace deltas as binary arrays. The separate
  HTTP payload is created only for requested auxiliary arrays (replay, sampling
  supports or top-k probabilities); its absence is normal for core-only traces.
- Compact logprobs decode to ordinary arrays; engines without compact output
  return their ordinary format. Sampling supports, top-k probabilities, weight
  versions, MoE routes and indexer replay are retained when requested. Headerless
  replay needs engine shapes in `client.backend_options.routed_experts_shape`
  or `indexer_topk_shape`. Missing/misaligned evidence is an error.
- Full-weight training omits `lora_path`; LoRA uses Miles' served adapter name.
- Image sidecars and token spans are retained. The training processor must
  reproduce the exact input sequence; mismatches fail instead of shifting
  tokens. Image-model GPU qualification is separate from the text-model test.
- `--partial-rollout` cannot resume arbitrary live environment state.
  `--chat-template-path` conflicts with renderer ownership; configure the
  renderer instead. `--use-opd` needs a separate teacher scoring path.
  `--use-output-store` is a separate SGLang fast path and is rejected here;
  ordinary Miles Ray/Mooncake data transport remains available.

CPU tests exercise real replicated pools, joint scoring, payload recovery,
worker failures, cancellation and shutdown. The GPU smoke recipe is
`tests/e2e/long/test_qwen3_0.6B_verifiers.py`. It uses subprocess execution for
trusted CodeGolf tasks inside an isolated test container; production deployments
should select their intended runtime isolation.

For real per-attempt Docker isolation against an existing SGLang engine, run:

```bash
python tests/manual/verifiers/run_live_env_server.py \
  --verifiers-root ../verifiers \
  --base-url http://YOUR_ENGINE:30000/v1 \
  --model YOUR_SERVED_MODEL \
  --renderer-model-name Qwen/Qwen3-0.6B \
  --runtime-image python:3.11-slim \
  --output /tmp/miles-runtime-validation
```

The runner owns its CPU workers and uniquely labelled sandbox containers; it
does not launch or change the inference service. Run it where Docker is
reachable and install the matching Verifiers checkout. The default tests cover
all three CodeGolf tasks with four attempts each, three fixed ReverseText
boundary inputs, Bash file tools, a two-turn conversation, a tool error,
cancellation and recovery. Additional scripted programs check binary file I/O,
nonzero exit, timeout, CPU-only device visibility, failed-program grading and
container cleanup. These scripted fault cases are separate from real model
generation; model accuracy is reported rather than treated as an infrastructure
failure. This is an integration qualification, not a dataset benchmark or a
full optimizer-step test.

Use `--abort-strategy workers` when the URL is a Miles router, or the default
`endpoint` for a single engine. `--runtime-config` accepts an existing Verifiers
runtime configuration. `--preinstalled-harness-python` can reuse a container
interpreter only after validating the chat program's declared dependencies;
it skips dependency downloading, not execution, interception or scoring.
