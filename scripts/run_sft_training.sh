#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must identify the allocated GPUs" >&2
  exit 2
fi

DATASET_PATH=${SFT_DATASET_PATH:-data/intermediate/sft/ms_swift_agent_881.jsonl}
OUTPUT_DIR=${SFT_OUTPUT_DIR:-outputs/train/sft_qwen3_4b_lora_r128}
MAX_STEPS=${SFT_MAX_STEPS:--1}
EPOCHS=${SFT_EPOCHS:-2}
GRAD_ACCUM=${SFT_GRAD_ACCUM:-16}
LEARNING_RATE=${SFT_LEARNING_RATE:-1e-4}
SAVE_STEPS=${SFT_SAVE_STEPS:-25}
DEEPSPEED_CONFIG=${SFT_DEEPSPEED_CONFIG:-configs/sft/zero3_cpu_offload.json}
RESUME_CHECKPOINT=${SFT_RESUME_CHECKPOINT:-}
NPROC=${NPROC_PER_NODE:-4}
SEQUENCE_PARALLEL_SIZE=${SFT_SEQUENCE_PARALLEL_SIZE:-4}

extra_args=()
if [[ "${DEEPSPEED_CONFIG}" == "none" ]]; then
  DEEPSPEED_CONFIG=""
elif [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  extra_args+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  extra_args+=(--resume_from_checkpoint "${RESUME_CHECKPOINT}")
fi

PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
NPROC_PER_NODE="${NPROC}" \
PYTHONPATH=src \
.venv-train/bin/swift sft \
  --model models/Qwen3-4B-Instruct-2507 \
  --dataset "${DATASET_PATH}" \
  --split_dataset_ratio 0 \
  --train_type lora \
  --lora_rank 128 \
  --lora_alpha 256 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --num_train_epochs "${EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --max_length 32768 \
  --agent_template hermes \
  --loss_scale default \
  --attn_impl flash_attn \
  --padding_free true \
  --use_liger_kernel true \
  --sequence_parallel_size "${SEQUENCE_PARALLEL_SIZE}" \
  --gradient_checkpointing true \
  --dataset_num_proc 8 \
  --dataloader_num_workers 4 \
  --logging_steps 1 \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 10 \
  --save_only_model false \
  --report_to tensorboard \
  --skip_memory_metrics false \
  --seed 20260819 \
  --data_seed 20260819 \
  --output_dir "${OUTPUT_DIR}" \
  "${extra_args[@]}"
