#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_revision=0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
model_path=/data/home/zhourunfeng/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/$model_revision
visible_gpus=${TEACHER_VISIBLE_GPUS:-1,2,5,6}
port=${TEACHER_PORT:-8010}

if [[ ! -f "$model_path/config.json" ]]; then
  printf 'teacher snapshot is incomplete: %s\n' "$model_path" >&2
  exit 1
fi

IFS=, read -r -a gpu_ids <<< "$visible_gpus"
if [[ ${#gpu_ids[@]} -ne 4 ]]; then
  printf 'TEACHER_VISIBLE_GPUS must contain exactly four GPU indices\n' >&2
  exit 2
fi
for gpu_id in "${gpu_ids[@]}"; do
  used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" | tr -d ' ')
  if (( used_mib > 1024 )); then
    printf 'refusing occupied GPU %s (%s MiB used)\n' "$gpu_id" "$used_mib" >&2
    exit 3
  fi
done

export CUDA_VISIBLE_DEVICES=$visible_gpus
exec "$repo_root/.venv-train/bin/vllm" serve "$model_path" \
  --host 127.0.0.1 \
  --port "$port" \
  --served-model-name qwen3-30b-teacher \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --disable-log-requests
