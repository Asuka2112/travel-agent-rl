# Replay v2 handoff

Replay Policy v2 is implemented and committed at
`f83bd733478602f8f194fe1817fabe43e621c411`. The full-parameter regression
resumed checkpoint-5, completed real optimizer step 6, and saved checkpoint-6.
All 6 deterministic snapshot misses became structured observations; all 6
continued to another tool, with zero public-network requests. The v2 Gate,
parameter-change check, 84 tests, and ruff all pass. No GRPO/Judge/monitor tmux
session is running.

Do not modify snapshot v4 (`dbd160b6de284208c3906e9da0768cabfe3a77315e64280ea59685eefff15f92`),
capture public web data, change frozen datasets, kill/reset other users' GPU
processes, or use the synthetic regression set as Pilot/Formal data.

Work is intentionally paused before the 48-query Pilot. On explicit instruction,
start with the exact GPU-audit command in `orchestration/STATE.json`, then select
the safe dynamic topology and run the frozen `pilot_48.jsonl` using
`configs/grpo/pilot_4gpu.yaml`. Do not start the G=6 Probe or Formal GRPO unless
the user expands the boundary again.

Key artifacts:

- checkpoint: `outputs/grpo/replay_v2_regression_4gpu_v1/v0-20260824-150953/checkpoint-6`
- metrics: `outputs/grpo/replay_v2_regression_4gpu_v1/metrics.json`
- telemetry: `outputs/grpo/rollout_telemetry/replay_v2_regression.jsonl`
- parameter check: `outputs/grpo/replay_v2_regression_4gpu_v1/parameter_change.json`
- report: `reports/GRPO_REPLAY_V2_REGRESSION.md`
- policy: `docs/replay_policy_v2.md`
- final GPU snapshot: `outputs/grpo/resource_snapshots/replay_v2_regression_final.json`
- run scripts: `scripts/run_grpo_rollout_server.sh`, `scripts/run_grpo_training.sh`
