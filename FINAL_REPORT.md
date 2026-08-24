# Travel Agentic RL Final Report

Generated from immutable project artifacts on 2026-08-25. Machine-readable JSON under
`outputs/final/` is the source of truth for all reported values.

## 1. Hardware

The run used a shared Ubuntu 22.04 server with 2 x AMD EPYC 7763 CPUs, 503 GiB RAM,
and 8 x NVIDIA RTX 4090 24 GiB GPUs (driver 550.144.03, CUDA 12.4). The training
environment used Python 3.10.12, PyTorch 2.6.0+cu124, Transformers 4.53.3,
ms-swift 3.8.3, vLLM 0.8.5.post1, and DeepSpeed 0.16.9.

## 2. GPU topology

GPU availability was re-audited before each expensive stage. The Formal GRPO run used
GPU6 for vLLM rollout, GPUs1/2/3 for the ZeRO-3 full-parameter trainer, and a fixed Base
Judge co-located on GPU1. External jobs on GPUs0/4/5/7 were never killed, reset,
preempted, or shared without proof of non-interference. Unified and pairwise evaluation
ran sequentially on idle-safe GPU6.

## 3. Dataset

The frozen production set contains 1,835 queries selected from an audited 17,486-query
catalog. After trajectory filtering, the immutable split contains 881 SFT trajectories,
200 RL queries, and 16 synthetic engineering dev queries. SFT/RL query-ID overlap,
dev/training overlap, protected benchmark overlap, and frozen benchmark overlap are all
zero. Key hashes are recorded in `orchestration/STATE.json` and
`outputs/final/repro_manifest.json`.

## 4. Distillation

Teacher distillation generated 1,820/1,835 trajectories (99.18%). Fifteen generation
failures were retained: 11 model-call-limit failures and 4 timeouts. Average generated
trajectory length was 10.824 turns and 10.889 tool calls. The production replay recorded
4,278/4,278 snapshot hits and zero public or cloud requests.

## 5. Filtering

The Rule gate accepted 1,699 trajectories and rejected 121 repeated-action cases. All
1,699 then passed format/schema validation. The fixed Judge accepted 1,463 and rejected
236 below-threshold trajectories. Judge retry repaired only exception rows; prompts and
thresholds were unchanged, and all original attempt artifacts were retained.

## 6. SFT

Qwen3-4B-Instruct-2507 was trained for 2 epochs and 110 optimizer steps using LoRA
rank/alpha 128/256, then merged into a 4,022,468,096-parameter standalone model with no
remaining LoRA parameter names. Runtime was 1,034.157 seconds. Aggregate/last loss was
0.2780154/0.22089405, exact train token accuracy was 92.9729%, and the frozen 64-row
held-out validation loss/token accuracy was 0.2238471/92.7896%.

## 7. Full-parameter GRPO

Replay Policy v2 regression, the 48-query G=4/max_turns=8 Pilot, and the three-step
G=6/max_turns=13 parameter Probe all passed before the single Formal run. Formal GRPO
started from the merged SFT model and kept Qwen3-4B, 200 frozen RL queries, G=6,
max_turns=13, eight tools, full-parameter updates, snapshot v4, and replay-only semantics
fixed. It completed 200/200 optimizer steps and exactly 200 unique queries x 6 rollouts
(1,200 telemetry records), producing checkpoint-200. Sampled tensors changed, and no
OOM, NCCL, NaN, or Inf failure occurred.

Formal aggregate metrics: reward 4.511838, mean group reward std 0.089953,
advantage non-zero ratio 0.970149, mean KL 0.008048, mean turns 9.170833, mean tool
calls 8.1925, final-answer rate 0.9275, and average rollout latency 25.2268 seconds.

## 8. Reward

The six explicit reward components remained fixed. Formal component means were:

| Component | Mean | Mean logged std |
| --- | ---: | ---: |
| Valid agent step | 0.999226 | 0.001535 |
| Valid tool call | 1.000000 | 0.000000 |
| Final-answer boundary | 0.978333 | 0.042988 |
| Stage awareness | 1.000000 | 0.000000 |
| Tool efficiency | 0.974810 | 0.044732 |
| Fixed local Judge | 0.546873 | 0.028231 |

Tool observations were verified outside the policy response-loss mask, response tokens
and masks remained aligned, and all eight tool schemas were visible in every rollout.

## 9. Base / SFT / RL unified evaluation

All three models used the identical 16-query dev hash, `full_replay` backend,
qwen3_nothinking/Hermes format, seed 20260819, max_turns=13, generation limit 768,
and fixed unmodified Base Judge with the same prompt hash.

| Metric | Base | SFT | RL |
| --- | ---: | ---: | ---: |
| Task success | 0.8750 | 0.6875 | 1.0000 |
| Constraint satisfaction | 0.8750 | 0.6875 | 1.0000 |
| Valid tool-call rate | 0.8889 | 1.0000 | 1.0000 |
| Invalid tool-call rate | 0.1111 | 0.0000 | 0.0000 |
| Final-answer rate | 0.8750 | 0.6875 | 1.0000 |
| Average turns | 2.0000 | 7.8750 | 2.0000 |
| Average tool calls | 1.0000 | 7.1875 | 1.0000 |
| Loop rate | 0.0000 | 0.3125 | 0.0000 |
| Tool efficiency | 0.9375 | 0.6563 | 1.0000 |
| Fixed Judge / 5 | 4.7768 | 4.0714 | 4.9911 |

No model made a public-network request or produced an unknown exception.

## 10. Pairwise evaluation

| Left vs right | Left wins | Ties | Left losses | Position disagreement | Pairwise score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base vs SFT | 10 | 4 | 2 | 4 | 0.7500 |
| Base vs RL | 10 | 4 | 2 | 4 | 0.7500 |
| SFT vs RL | 0 | 10 | 6 | 10 | 0.3125 |

Pairwise judging uses the fixed Base model, a frozen pairwise rubric, Path = 30%, Final
Answer = 70%, and 96 real Judge calls covering A/B and B/A orientations for every query
and model pair. The raw orientation records and position disagreements are retained in
`outputs/final/pairwise_metrics.json`.

The direct pairwise outcome does not reproduce the absolute-score ordering: the Base
Judge assigned RL the highest independent mean score, but preferred Base to RL on 10
queries in consensus pairwise comparison. Position disagreement affected 4/16 Base/RL
queries and 10/16 SFT/RL queries. This is evidence of prompt/context and position
sensitivity in the small fixed Judge, so neither view is suppressed or treated as the
sole scientific conclusion.

## 11. Failure analysis

Base failed 2/16 queries due to invalid native tool-call formatting. SFT failed 6/16,
with 30 deterministic snapshot misses, 5 loop cases, and no invalid tool calls or unknown
exceptions. RL failed 0/16. These outcomes are reported without threshold, seed, or
checkpoint cherry-picking.

## 12. Snapshot misses

Snapshot v4 is immutable with content SHA-256
`dbd160b6de284208c3906e9da0768cabfe3a77315e64280ea59685eefff15f92`.
Formal training logged 1,398 structured deterministic misses: 1,350 recovered into
continued agent execution and 48 remained explicit final failures. No miss fabricated a
result or fell through to the public network. Unified evaluation recorded 30 misses for
SFT and zero for Base/RL.

## 13. Resource usage

Formal peak GPU memory was 22,388 MiB on GPU1, 11,004 MiB on GPU2, 11,044 MiB on
GPU3, and 15,352 MiB on GPU6. Peak project RSS was 175.304 GiB and peak host RAM use
was 255.167 GiB. All selected GPUs reached 100% measured utilization while external GPU
processes remained untouched.

## 14. Training time

Production SFT took 1,034.157 seconds (17m 14s). Formal GRPO took 25,796.069 seconds
(7h 9m 58s). Pilot, Probe, evaluation, and Judge timings are retained in their logs and
machine-readable artifacts.

## 15. Reproducibility

`outputs/final/repro_manifest.json` records source/data/config/prompt/artifact hashes,
model lineage, fixed protocol settings, topology logs, environment versions, and exact
entry-point commands. `outputs/final/scientific_audit.json` validates dataset overlap,
contamination, snapshot integrity, masks, reward definition, evaluation/Judge parity,
public-network isolation, and checkpoint provenance.

## 16. Limitations

The interactive dev set has only 16 synthetic engineering queries focused on tool
coverage, so the perfect RL point estimate is not a broad travel-planning benchmark
claim. The same Base model acts as the fixed Judge, which can introduce model-family
bias. Replay-only evaluation cannot answer unseen public-Web requests and intentionally
returns deterministic misses. SFT used a LoRA training stage before merge, while GRPO
itself was a verified 100% full-parameter update. No confidence interval is claimed for
the small dev set.

## 17. Final conclusion

The end-to-end Data -> Distillation -> Filtering -> SFT -> Full-Parameter Agentic GRPO
-> Evaluation pipeline completed with immutable provenance and zero public-network
requests during replay-only RL/evaluation. The single Formal run passed its technical
and Replay v2 gates. On the frozen 16-query engineering evaluation, RL exceeded both
Base and SFT on task success, constraint satisfaction, tool efficiency, and independent
fixed-Judge score. True pairwise judging preferred RL over SFT but preferred Base over
RL, with material position disagreement. The behavioral result is positive, while the
conflicting Judge views and small synthetic dev set prevent a broad superiority claim.
