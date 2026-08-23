#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

TARGET_LENGTH=${1:-32768}
if [[ "${TARGET_LENGTH}" != "24576" && "${TARGET_LENGTH}" != "32768" ]]; then
  echo "target length must be 24576 or 32768" >&2
  exit 2
fi
PROBE_DIR=outputs/probes/sft_memory/sp4_zero3cpu_${TARGET_LENGTH}_sampled
mkdir -p "${PROBE_DIR}"
SAMPLE_FILE=${PROBE_DIR}/nvidia_smi_samples.csv
TRAIN_LOG=${PROBE_DIR}/train.log

nvidia-smi \
  --query-gpu=timestamp,index,uuid,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop-ms=200 >"${SAMPLE_FILE}" &
MONITOR_PID=$!
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv-train/bin/swift sft \
  --model models/Qwen3-4B-Instruct-2507 \
  --dataset "data/intermediate/sft/probe_${TARGET_LENGTH}_sp4.jsonl" \
  --split_dataset_ratio 0 \
  --train_type lora \
  --lora_rank 128 \
  --lora_alpha 256 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --max_steps 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --max_length "${TARGET_LENGTH}" \
  --agent_template hermes \
  --loss_scale default \
  --attn_impl flash_attn \
  --padding_free true \
  --gradient_checkpointing true \
  --sequence_parallel_size 4 \
  --use_liger_kernel true \
  --deepspeed configs/sft/zero3_cpu_offload.json \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --logging_steps 1 \
  --save_steps 1 \
  --save_total_limit 1 \
  --report_to tensorboard \
  --seed 20260819 \
  --data_seed 20260819 \
  --output_dir "${PROBE_DIR}" 2>&1 | tee "${TRAIN_LOG}"
