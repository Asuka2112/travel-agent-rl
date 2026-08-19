# Compatibility record

Checked on 2026-08-19. Sources are linked so the assessment can be audited.

## Host constraint

The server has NVIDIA driver 550.144.03, CUDA runtime/toolkit 12.4, and RTX
4090 GPUs. The system driver must not be upgraded without explicit approval.

## Upstream facts

- Qwen3-4B-Instruct-2507 is Apache-2.0 and the model card recommends a recent
  Transformers release: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- Qwen's official training guide documents ms-swift, DeepSpeed ZeRO-3, GRPO,
  and RTX support: https://github.com/QwenLM/Qwen3/blob/main/docs/source/training/ms_swift.md
- ms-swift documents server-mode asynchronous multi-turn schedulers and custom
  environment steps: https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/DeveloperGuide/multi_turn.md
- Current ms-swift full-install guidance targets Python 3.10/3.11 and CUDA 12,
  but its newest example image uses a newer CUDA stack than this host:
  https://github.com/modelscope/ms-swift/blob/main/requirements/install_all.sh

## Candidate CUDA 12.4 stack

The initial candidate is pinned in `requirements/train-cu124.txt`: PyTorch
2.6.0 cu124, Transformers 4.53.3, ms-swift 3.8.3, vLLM 0.8.5.post1,
DeepSpeed 0.16.9, and FlashAttention 2.7.4.post1.

This is deliberately labeled **candidate**, not **working**. It becomes the
lock only after all imports, bf16 matmul, Qwen config/tokenizer loading,
vLLM startup, DeepSpeed extension checks, and a minimal multi-turn GRPO update
are executed and saved. Failures and the smallest required changes are logged
here; no base-model or full-parameter-GRPO substitution is implicit.

