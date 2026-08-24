# GRPO Smoke Replay Gate

Generated: 2026-08-24T06:33:08Z

## Status

`waiting_for_snapshot_decision`

The real 4-GPU full-parameter pipeline feasibility checks passed, but the strict
`replay_miss = 0` acceptance condition did not: one of 30 rollouts emitted a new
Web Search query that is absent from immutable snapshot v4. Pilot training was
therefore not started.

This is neither a hardware feasibility failure nor a HARD STOP. The completed
checkpoint is safe to resume after the frozen-snapshot policy is resolved.

## Completed smoke evidence

- Topology: GPU0 TP=1 vLLM server; GPU1-3 DeepSpeed ZeRO-3 trainer
- Training: Qwen3-4B, 4.022B/4.022B parameters trainable (100%)
- Frozen RL engineering subset: 12 hard queries
- Subset SHA-256: `844bb65ec77efcf33c79e56d8a086a1b7feac25fb5f96fa019b232248cf2f437`
- Optimizer steps: 5
- Resume: checkpoint-3 to checkpoint-5 across a new trainer process
- Rollouts: 30
- Tool calls / observations: 88 / 88
- Tool-schema count: 8 in every rollout
- Reward mean / mean group std: 2.891676282 / 0.16214967
- Advantage non-zero ratio: 71.43%
- Mean KL: 0.000132312
- Mean rollout latency: 4.874 seconds
- Unknown exceptions: 0
- Invalid tool errors: 0
- Public-network requests: 0
- NaN / Inf / OOM / NCCL failures: 0
- GPU0 peak total VRAM: 15,401 MiB, including the untouched external ~4.5 GiB
- GPU1/2/3 peak VRAM: 7,330 / 7,404 / 7,344 MiB
- Checkpoint: `outputs/grpo/smoke_valid_4gpu_v1/v1-20260824-141304/checkpoint-5`

Three sampled full-parameter tensors changed. Their changed-element counts were
126,742; 889,148; and 128,875, respectively. The detailed before/after hashes
are in `outputs/grpo/smoke_valid_4gpu_v1/parameter_change.json`.

## Replay miss

- Query ID: `agent_star:hard_trsd60`
- Tool: `web_search`
- Generated request: `flights from Fort Lauderdale to North Carolina cities March 2022`
- Snapshot request key: `211403b5e37c8bc74367373e370254865f621d95255c61ff584ac5f8bda5e12d`
- Result: deterministic `snapshot_miss`
- Live request: false
- Public request count: 0
- Frozen snapshot v4 content SHA-256: `dbd160b6de284208c3906e9da0768cabfe3a77315e64280ea59685eefff15f92`

The exact query does not exist in v4. Treating it as a hit, returning fabricated
content, silently returning empty results, disabling Web Search, or excluding
the rollout would violate the frozen protocol or the audit requirements.

## Engineering bugs fixed before this gate

1. ms-swift 3.8.3 moved the dataset `tools` column into `data_dict` in server
   mode. The rollout scheduler now restores the native request field and fails
   fast unless exactly eight tool schemas are visible.
2. The deterministic zero-loss turn-limit marker was initially parsed as a real
   final answer. It is now excluded from final-boundary and tool-call rewards.
3. Rollout telemetry now records schema visibility, tool/observation alignment,
   latency, repeated actions, replay misses, and public-network counts.

All GRPO and reward tests pass (17 tests), and ruff passes for the changed GRPO,
reward, test, and reporting code.

## Decision required

Either option changes a frozen experimental condition and therefore requires
explicit authorization:

1. Create immutable snapshot v5 using one real self-hosted online capture,
   preserving v4 and recording both hashes, then rerun the strict smoke gate.
2. Keep v4 unchanged and explicitly relax the smoke/pilot criterion to permit
   deterministic offline snapshot misses while continuing to require zero
   public-network requests during training.

Until one option is authorized, the 48-query pilot and formal GRPO remain
unstarted. Project rollout, judge, and resource-monitor services were stopped;
GPU0 returned to the untouched external 4,520 MiB occupancy and GPU1-3 are free.
