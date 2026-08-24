#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

STAGE=${GRPO_STAGE:-smoke}
PORT=${GRPO_ROLLOUT_PORT:-8008}
DATASET=${GRPO_DATASET:-data/intermediate/grpo/smoke_12.jsonl}
OUTPUT_DIR=${GRPO_OUTPUT_DIR:-outputs/grpo/smoke_4gpu_v1}
NUM_GENERATIONS=${GRPO_NUM_GENERATIONS:-2}
MAX_TURNS=${GRPO_MAX_TURNS:-3}
MAX_STEPS=${GRPO_MAX_STEPS:-5}
EPOCHS=${GRPO_EPOCHS:-1}
GRAD_ACCUM=${GRPO_GRAD_ACCUM:-2}
SAVE_STEPS=${GRPO_SAVE_STEPS:-1}
RESUME_CHECKPOINT=${GRPO_RESUME_CHECKPOINT:-}
TRAINER_GPUS=${GRPO_TRAINER_GPUS:-1,2,3}
NPROC=${GRPO_NPROC_PER_NODE:-$(awk -F, '{print NF}' <<<"${TRAINER_GPUS}")}
SNAPSHOT_VERSION=${WEB_SNAPSHOT_VERSION:-4}
MODEL=${GRPO_MODEL:-outputs/models/qwen3_4b_sft_merged}

extra_args=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  extra_args+=(--resume_from_checkpoint "${RESUME_CHECKPOINT}")
fi

exec env \
  CUDA_VISIBLE_DEVICES="${TRAINER_GPUS}" \
  NPROC_PER_NODE="${NPROC}" \
  PYTHONPATH=src \
  WEB_MODE=replay_only \
  WEB_BACKEND=selfhosted \
  WEB_SNAPSHOT_VERSION="${SNAPSHOT_VERSION}" \
  FIXED_JUDGE_ENDPOINT=http://127.0.0.1:8019/judge \
  GRPO_SCHEDULE_MAX_STEPS="${GRPO_SCHEDULE_MAX_STEPS:-${MAX_STEPS}}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv-train/bin/swift rlhf \
    --rlhf_type grpo \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --split_dataset_ratio 0 \
    --train_type full \
    --torch_dtype bfloat16 \
    --template qwen3_nothinking \
    --agent_template hermes \
    --loss_scale default \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port "${PORT}" \
    --vllm_server_timeout 600 \
    --vllm_server_pass_dataset true \
    --external_plugins scripts/grpo_swift_plugin.py \
    --multi_turn_scheduler travel_agent_replay \
    --max_turns "${MAX_TURNS}" \
    --completion_length_limit_scope per_round \
    --max_completion_length 512 \
    --num_generations "${NUM_GENERATIONS}" \
    --reward_funcs \
      travel_valid_agent_step \
      travel_valid_tool_call \
      travel_final_boundary \
      travel_stage_awareness \
      travel_tool_efficiency \
      travel_fixed_judge \
    --reward_weights 1 1 1 0.5 0.5 1 \
    --scale_rewards true \
    --beta 0.04 \
    --epsilon 0.2 \
    --temperature 0.9 \
    --learning_rate 5e-7 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.05 \
    --weight_decay 0.0 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --num_train_epochs "${EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --deepspeed configs/sft/zero3_cpu_offload.json \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --max_length 32768 \
    --logging_steps 1 \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 10 \
    --save_only_model false \
    --log_completions true \
    --report_to tensorboard \
    --skip_memory_metrics false \
    --seed 20260819 \
    --data_seed 20260819 \
    --output_dir "${OUTPUT_DIR}" \
    "${extra_args[@]}"
