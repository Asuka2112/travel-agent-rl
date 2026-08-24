# Current task

Run the Formal Parameter Probe with Replay Policy v2:

- dataset: 8–16 frozen RL queries (currently `data/intermediate/grpo/smoke_tool_12.jsonl`)
- num_generations: 6
- max_turns: 13
- optimizer steps: at least 3
- topology: choose from a fresh GPU0–7 audit; include idle-safe GPU6
- fixed Judge: immutable Base model, GPU-backed and proven non-fallback before trainer launch
- web mode: replay_only
- snapshot: immutable v4

On success: verify OOM/reward variance/advantage/multi-turn/weight sync/checkpoint/resource gates, stop Probe services, commit the milestone, then re-audit GPU0–7 before the one-time 200-query Formal Full-Parameter GRPO run.
