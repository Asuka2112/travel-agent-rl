# Current task

Run the one-time 200-query Formal Full-Parameter GRPO experiment with Replay Policy v2:

- dataset: `data/processed/rl_queries.jsonl` (200 frozen RL queries)
- num_generations: 6
- max_turns: 13
- optimizer steps: enough to cover all 200 unique queries exactly once at G=6
- topology: choose from a fresh GPU0–7 audit; include idle-safe GPU6
- initialization: `outputs/models/qwen3_4b_sft_merged`
- training: 100% full parameters; no LoRA
- fixed Judge: immutable Base model, GPU-backed and already validated non-fallback in Probe
- web mode: replay_only
- snapshot: immutable v4

This is the single formal main run. Do not tune rewards, seeds, model, dataset, G, max turns, replay semantics, or evaluation protocol. On completion, stop project services, validate the final checkpoint and training artifacts, commit the milestone, and proceed directly to unified Base/SFT/RL evaluation.
