# Qwen3-4B Agent SFT Production Training

Generated: 2026-08-23T19:53:13.555008+00:00

## Frozen inputs

- Training data: `data/intermediate/sft/ms_swift_agent_881.jsonl`
- Training data SHA-256: `f9b8b2be0289f48bb89ce769a351bf652d2b74de61706d8a57d3b6644a346111`
- Configuration: `configs/sft/production.yaml`
- Configuration SHA-256: `fb605ba86bf227ace2ea196c7852fc3f9f9779f6f0c81b6abc2b990f2d70952d`
- Base model revision: `cdbee75f17c01a7cc42f958dc650907174af0554`

## Configuration

- Model: `models/Qwen3-4B-Instruct-2507`
- LoRA rank/alpha/dropout: 128/256/0.05
- Epochs: 2
- Max length: 32768
- Precision: `bfloat16`
- GPUs: 4
- Per-device batch / gradient accumulation / global batch: 1 / 4 / 16
- Learning rate: 2e-05
- Attention/kernel: `flash_attn` / Liger=True
- Gradient checkpointing: True

## Results

- Optimizer steps: 110
- Wall-clock training runtime: 1034.1570 seconds
- Aggregate train loss: 0.27801540
- Last-step train loss: 0.22089405
- Exact full-corpus train token accuracy: 92.97% (610305/656433)
- Supervised validation loss: N/A; immutable dev has no assistant labels
- Supervised validation token accuracy: N/A; immutable dev has no assistant labels
- Peak trainer-reported GPU memory: 16.81 GiB
- Final gradient norm: 0.41995385
- Final learning rate: 0.0
- NaN/OOM/NCCL errors: 0

## Artifacts

- LoRA adapter: `outputs/train/sft_qwen3_4b_lora_r128_production/v0-20260824-032206/checkpoint-110`
- Merged model: `outputs/models/qwen3_4b_sft_merged`
- Merged model load verification: `outputs/models/qwen3_4b_sft_merged/verification.json` (`loaded=True`)
- Training log: `outputs/train/sft_qwen3_4b_lora_r128_production/v0-20260824-032206/logging.jsonl`
- Token-accuracy audit: `outputs/eval/sft_train_token_accuracy.json`

The synthetic dev is kept immutable and query-only. Creating assistant labels for it would change the frozen evaluation set, so validation loss and validation token accuracy are reported as N/A rather than fabricated.
