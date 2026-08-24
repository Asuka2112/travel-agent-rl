# Current task

Run the parity-locked Base/SFT/RL unified evaluation after the successful Formal GRPO run:

- frozen dev: `data/processed/dev.jsonl` (16 rows, SHA-256 `66d8d8f5ce4ac9e42a265da3f2c28634a54d14a3628150e36b008bb1496c5712`)
- models: immutable Base, merged full-parameter SFT, and Formal RL `checkpoint-200`
- protocol: same `full_replay` backend, seed, template, max turns, generation limit, and fixed local Base Judge
- selected resource: idle-safe GPU6, with Base/SFT/RL evaluated sequentially
- public network: zero
- outputs: immutable per-query records, summaries, Judge dimensions, unified comparison, pairwise evaluation, and failure analysis

Do not modify the frozen dev set, replay snapshot, Judge model/prompt, or evaluation settings between models. After evaluation, generate the final metrics and reproduction manifest, run tests/ruff and repository audits, then create the final commit.
