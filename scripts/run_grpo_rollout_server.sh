#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

STAGE=${GRPO_STAGE:-smoke}
PORT=${GRPO_ROLLOUT_PORT:-8008}
MAX_TURNS=${GRPO_MAX_TURNS:-3}
MAX_MODEL_LEN=${GRPO_VLLM_MAX_MODEL_LEN:-8192}
GPU_UTIL=${GRPO_VLLM_GPU_UTIL:-0.40}
ROLLOUT_GPUS=${GRPO_ROLLOUT_GPUS:-0}
ROLLOUT_TP=${GRPO_ROLLOUT_TP:-1}
SNAPSHOT_VERSION=${WEB_SNAPSHOT_VERSION:-4}
MODEL=${GRPO_MODEL:-outputs/models/qwen3_4b_sft_merged}
SECRETS_FILE=${TRAVEL_AGENT_SECRETS_FILE:-/data/home/zhourunfeng/.config/travel-agent-rl/secrets.env}

if [[ -f "${SECRETS_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SECRETS_FILE}"
  set +a
fi
: "${AMAP_API_KEY:?AMAP_API_KEY is required for the frozen eight-tool registry}"

exec env \
  CUDA_VISIBLE_DEVICES="${ROLLOUT_GPUS}" \
  PYTHONPATH=src \
  WEB_MODE=replay_only \
  WEB_BACKEND=selfhosted \
  WEB_SNAPSHOT_VERSION="${SNAPSHOT_VERSION}" \
  WEB_RUN_ID="grpo_${STAGE}_replay_v${SNAPSHOT_VERSION}" \
  GRPO_STAGE="${STAGE}" \
  GRPO_TELEMETRY_PATH="outputs/grpo/rollout_telemetry/${STAGE}.jsonl" \
  TRAVEL_AGENT_RL_ROOT="${ROOT_DIR}" \
  .venv-train/bin/swift rollout \
    --model "${MODEL}" \
    --template qwen3_nothinking \
    --agent_template hermes \
    --infer_backend vllm \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --vllm_tensor_parallel_size "${ROLLOUT_TP}" \
    --vllm_gpu_memory_utilization "${GPU_UTIL}" \
    --vllm_max_model_len "${MAX_MODEL_LEN}" \
    --vllm_max_num_seqs 16 \
    --vllm_enforce_eager true \
    --vllm_use_async_engine true \
    --logprobs true \
    --max_logprobs 1 \
    --external_plugins scripts/grpo_swift_plugin.py \
    --multi_turn_scheduler travel_agent_replay \
    --max_turns "${MAX_TURNS}" \
    --seed 20260819
