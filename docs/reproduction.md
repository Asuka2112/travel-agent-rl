# Reproduction

## Host prerequisites

- Ubuntu 22.04 x86_64
- Python 3.10
- NVIDIA driver compatible with the selected CUDA wheel set
- Git, at least 200 GB free project storage, and explicitly allocated GPUs

Do not alter the system CUDA toolkit or driver to reproduce this project.
Python and CUDA user-space dependencies live in a project-local environment.

## Foundation environment

```bash
git clone <repository-url> travel-agent-rl
cd travel-agent-rl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==26.2.1
python -m pip install -r requirements/base.txt -r requirements/dev.txt
python scripts/smoke_env.py
pytest
```

## Training dependencies

The CUDA 12.4 stack is created in a separate environment:

```bash
bash scripts/setup_train_env.sh
```

`requirements/train-cu124-lock.txt` records all observed transitive versions.
Read `docs/compatibility.md` before using the stack on a different driver.

## Required secrets

The frozen reproduction uses `WEB_MODE=replay_only`, local database/simulator tools,
and local Base-model Judges. It requires no paid API or live provider credential and
must make zero public-network requests. Online data capture is outside the frozen
reproduction protocol; never commit `.env` or provider keys.

## Reproducibility controls

- Global seed defaults to 20260819.
- Source datasets are recorded by repository/dataset revision and SHA-256.
- The model is recorded by Hugging Face commit revision.
- Prompts and configs are committed and referenced by content hash.
- The frozen test manifest is read-only after its creation commit.
- Training/evaluation commands produce an experiment manifest before work.

## Completed formal protocol

The frozen main experiment is:

```text
model: Qwen3-4B-Instruct-2507
initialization: outputs/models/qwen3_4b_sft_merged
dataset: data/intermediate/grpo/formal_200.jsonl
training: full parameter
num_generations: 6
max_turns: 13
replay policy: v2
snapshot: immutable v4
public network requests: 0
final checkpoint: outputs/grpo/formal_4gpu_v1/v1-20260824-191232/checkpoint-200
```

Resource selection is dynamic. Before every expensive launch, run
`scripts/audit_gpu_resources.py`, record every GPU role/unused reason, and never kill,
reset, or preempt external work. The observed Formal topology was GPU6 rollout plus
GPU1/2/3 trainer, with the fixed Judge co-located on GPU1. See
`outputs/grpo/resource_snapshots/formal_grpo_retry2_topology.json`.

## Unified evaluation

Run Base, SFT, and RL sequentially with identical arguments, changing only `--model`,
`--output-dir`, `--report`, and `--label`:

```bash
CUDA_VISIBLE_DEVICES=6 .venv-train/bin/python scripts/evaluate_agent_model.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --dev data/processed/dev.jsonl \
  --output-dir outputs/evaluation/unified_20260825/base \
  --report reports/EVAL_BASE_UNIFIED.md --label Base \
  --max-turns 13 --max-new-tokens 768 --seed 20260819 \
  --device cuda:0 --attn-impl flash_attn --tool-backend full_replay
```

Use `outputs/models/qwen3_4b_sft_merged` for SFT and
`outputs/grpo/formal_4gpu_v1/v1-20260824-191232/checkpoint-200` for RL. Score all three
record directories with `scripts/evaluate_fixed_local_judge.py` and the immutable Base
model. Then run `scripts/report_unified_evaluation.py` and
`scripts/report_failure_analysis.py`.

True bidirectional pairwise evaluation is:

```bash
CUDA_VISIBLE_DEVICES=6 .venv-train/bin/python scripts/evaluate_pairwise_local_judge.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --base-dir outputs/evaluation/unified_20260825/base \
  --sft-dir outputs/evaluation/unified_20260825/sft \
  --rl-dir outputs/evaluation/unified_20260825/rl \
  --output outputs/final/pairwise_metrics.json \
  --device cuda:0 --seed 20260819
```

Finally generate the audited manifests and validate the repository:

```bash
.venv-train/bin/python scripts/build_final_artifacts.py --output-dir outputs/final
.venv-train/bin/pytest -q
.venv/bin/ruff check .
```

Exact content hashes and model lineage are in `outputs/final/repro_manifest.json`.
