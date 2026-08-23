# Qwen3-4B Agent SFT Smoke Test

Generated: 2026-08-24

## Fixed subset and configuration

- Source: frozen 881-row native Agent SFT dataset
- Source SHA-256: `f9b8b2be0289f48bb89ce769a351bf652d2b74de61706d8a57d3b6644a346111`
- Fixed smoke subset: `data/intermediate/sft/smoke_320.jsonl`
- Smoke subset SHA-256: `1fc3bed9e9bed6e9d2e4399315bc1a4c126130a9b8aa5a34749d9994c1e00ade`
- Distribution: Easy 152, Medium 113, Hard 55
- Model: Qwen3-4B-Instruct-2507
- LoRA: rank 128, alpha 256, dropout 0.05, all linear modules
- Precision/kernel: bf16, FlashAttention 2, Liger, gradient checkpointing, padding-free
- GPUs: 4, per-device batch 1, gradient accumulation 4, global batch 16
- Learning rate: `2e-5`, selected after a controlled `1e-4` comparison produced poorer
  tool-call/final-answer balance
- Optimizer steps: 40 (within the required 20–50 range)

The smoke subset is engineering-only and is not used as a final model-quality result.

## Training and recovery checks

| Check | Result | Evidence |
|---|---|---|
| Finite loss | PASS | step 40 loss `0.28020304`; no NaN/Inf |
| Gradient norm | PASS | step 40 `0.56648821`; finite throughout |
| CUDA OOM | PASS | none; peak trainer memory `16.81 GiB` |
| NCCL/runtime error | PASS | none |
| Checkpoint save | PASS | checkpoint-10/20/30/40 saved |
| Checkpoint resume | PASS | checkpoint-20 resumed through step 40 |
| Adapter load | PASS | checkpoint-40 loaded by both inference evaluators |
| Tool template/schema | PASS | native `qwen3_nothinking` + `hermes`, all eight tools |

The resumed segment completed in 198.22 seconds. The warning emitted at process exit states
that `destroy_process_group()` was not explicitly called; it did not affect training,
checkpoint integrity, or NCCL collectives.

## Ten-query Agent behavior

Two complementary diagnostics were preserved:

1. Free-running frozen Train Search dev: every generated call was schema-valid and all ten
   selected Train Search. The model then requested POI/Web evidence that this deliberately
   minimal backend does not implement; 2/10 reached a final answer before the six-turn cap.
2. Immutable trajectory-context diagnostic: it supplies the real teacher observation at three
   checkpoints to isolate the behaviors required by the smoke gate.

| Trajectory-context check | Protocol pass | Reference action match | Invalid |
|---|---:|---:|---:|
| Initial tool selection | 10/10 | 8/10 | 0 |
| Continue after first observation | 10/10 | 6/10 | 0 |
| Final answer after complete evidence | 5/10 | 5/10 | 0 |
| Overall | 25/30 | 19/30 | 0 |

This gate checks pipeline integrity rather than claiming final quality. It demonstrates legal
tool JSON, schema parsing, observation-conditioned continuation, final-answer generation,
checkpoint recovery, and adapter loading without template corruption. The longer formal run
subsequently improved the same merged-model trajectory-context final-answer check to 9/10.

## Decision

**PASS.** All engineering gate requirements passed; the non-representative free-running dev
limitation is retained in the audit rather than hidden or used to modify the frozen evaluator.
