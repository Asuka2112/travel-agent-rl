# Frozen Evaluation: Base vs SFT

Generated: 2026-08-23T19:53:13.555008+00:00

- Frozen dev SHA-256: `66d8d8f5ce4ac9e42a265da3f2c28634a54d14a3628150e36b008bb1496c5712`
- Identical template, tool schemas, seed, backend, parser, and scoring pipeline
- Public web requests: 0
- Judge: no LLM judge exists for this synthetic dev; deterministic task score is shown

| Metric | Base | SFT | Delta |
|---|---:|---:|---:|
| Valid tool-call rate | 1.000000 | 1.000000 | +0.000000 |
| Invalid tool-call rate | 0.000000 | 0.000000 | +0.000000 |
| Final-answer rate | 1.000000 | 0.000000 | -1.000000 |
| Task completion | 1.000000 | 0.000000 | -1.000000 |
| Mean task score | 1.000000 | 0.500000 | -0.500000 |
| Average turns | 2.000000 | 6.000000 | +4.000000 |
| Average tool calls | 1.000000 | 6.000000 | +5.000000 |
| Loop rate | 0.000000 | 0.000000 | +0.000000 |

This dev consists of 16 explicit Train Search coverage prompts and is not a representative end-to-end travel-planning validation distribution. Results are reported without changing the frozen set or scoring logic.
