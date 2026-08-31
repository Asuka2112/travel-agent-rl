# Resume Evidence Audit — Travel Agentic RL

Generated from frozen repository artifacts only. No training, inference, validation, checkpoint mutation, reward/Judge change, or dataset mutation was performed. Tags mean: `[DIRECT]` is present in a primary artifact; `[DERIVED]` is deterministically recomputed; `[MISSING]` is absent; `[UNVERIFIED]` is text without primary support.

## Executive summary: TOP 10 strongest resume claims

| # | Claim | Value | Why it matters | Evidence | Confidence |
|---:|---|---|---|---|---|
| 1 | Frozen 17.6K-candidate data pipeline | [DIRECT] 17,594 raw → 17,386 train candidates → 1,835 frozen queries | Shows reproducible selection and contamination exclusion | `data/processed/data_stats.json` | A |
| 2 | High-yield offline teacher distillation | [DIRECT] 1,820/1,835 trajectories (99.1826%) | Demonstrates production-scale agent trajectory generation | `reports/distillation_1835.json` | A |
| 3 | Eight-tool trajectory corpus | [DERIVED] 19,818 tool calls across 1,820 trajectories | Demonstrates broad agent/tool orchestration | `reports/distillation_1835.json:tool_calls` | B |
| 4 | Replay-only reproducibility | [DIRECT] 4,278/4,278 snapshot hits; 0 public-network requests | Eliminates live-web drift during distillation | `reports/distillation_1835.json:replay` | A |
| 5 | Quality-filtered data | [DIRECT] 1,820 raw → 1,699 rule/format-valid → 1,463 Judge-accepted → 881 SFT + 200 RL | Shows auditable data gates | `reports/distillation_1835.json:cleaning/split` | A |
| 6 | Long-context LoRA engineering | [DIRECT] rank 128; max_length 32,768; verified 32K real train step | Shows memory-aware fine-tuning | `configs/sft/production.yaml; reports/sft_memory_probe.md` | A |
| 7 | Fast production SFT | [DIRECT] 881 rows × 2 epochs, 110 steps in 1,034.157 s; peak 16.81 GiB | Quantifies training efficiency | `outputs/final/training_metrics.json:sft` | A |
| 8 | Held-out SFT quality | [DIRECT] loss 0.223847; 43,844/47,251 token accuracy = 92.7896% | Separates held-out metrics from train metrics | `outputs/eval/sft_heldout_validation_64/metrics.json` | A |
| 9 | Full-parameter agentic GRPO | [DIRECT] 200 steps × G=6 = 1,200 rollouts; max_turns=13 | Shows end-to-end multi-turn RL | `outputs/grpo/formal_4gpu_v1/feasibility.json` | A |
| 10 | GRPO stability | [DIRECT] 97.0149% non-zero advantages; KL mean 0.00805; no NaN/OOM/NCCL | Shows usable learning signal and stable optimization | `outputs/final/training_metrics.json:formal_grpo_200` | A |

## TOP 5 claims that should NOT be written

| Claim | Verdict | Reason |
|---|---|---|
| 306 artificial seeds | [MISSING] | No artificial seed set exists; queries are selected from Agent-STAR TravelTotal. |
| 1,831 generated trajectories | DIFFERENT | Primary report says 1,820 successes and 15 failures. |
| 2 rollout + 6 trainer GPUs | DIFFERENT | Actual formal topology is GPU6 rollout + GPUs1–3 trainer (1+3). |
| Judge weight 0.60→0.75 / reward 0.43→0.61 | DIFFERENT | Actual Judge curriculum is 0.2→1.0; total reward early/late is 4.0489→4.8925. |
| TravelPlanner success | DO NOT CLAIM | Corrected Final Pass remains 0/180 for Base/SFT/RL; hard constraints never executed. |

## 1. Project identity and immutable lineage

- [DIRECT] Project: Travel Agentic RL; audit baseline HEAD `18dc679d7b4a53baf6c46f2dd18e3cabb1f63a83`; worktree was clean before this audit (`git status --porcelain` empty).
- [DIRECT] Final training commit: `5b81c3318a4305e2981e78ccedff00b3214fc539`; frozen external-eval commit: `3a7f633ad1ad20066ab361b173bf3de703a7d44e`; final adapter-repair commit: `18dc679d7b4a53baf6c46f2dd18e3cabb1f63a83`.
- [DIRECT] Base: `models/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`, tree hash `d12ae104099c24b2c3a8131d677043f4cbd8efcdf6de074b437203f619271405`.
- [DIRECT] SFT adapter: `outputs/train/sft_qwen3_4b_lora_r128_production/v0-20260824-032206/checkpoint-110`; merged model: `outputs/models/qwen3_4b_sft_merged`, tree hash `d0bc741cadea98fd392e3e1931229c81f61b0816773e24e0240df718729e6f06`.
- [DIRECT] RL: `outputs/grpo/formal_4gpu_v1/v1-20260824-191232/checkpoint-200`, tree hash `6af8b25ff92c0b131b18f0c0c91efb8f15e6e15bc831168536044c633d1c8c13`; size 52.46 GiB [DERIVED].
- [DIRECT] Frozen-query/SFT/RL hashes: `cf2f694419f7a96e868d07507a0a26d648cfd885ce69cda7daae36b3d3d82e6e` / `b63a8464d58191cd5825cdd54899f443d1f90f57cece678a89fff23839de8008` / `ca9411657993bb3cc4c754ec4a47583d3207ab6dc9cb4bfa8451e934148e9c80`.
- [DIRECT] Snapshot-content/reward/formal-GRPO/SFT/adapter hashes: `dbd160b6de284208c3906e9da0768cabfe3a77315e64280ea59685eefff15f92` / `f980d6a47045f7a622eadc726e0009a47838eceae8e3ed3f727fd7a90761e57d` / `af96f0672c588b48971f4011692050a84f2851260262645adf5139d6551d1e5f` / `fb605ba86bf227ace2ea196c7852fc3f9f9779f6f0c81b6abc2b990f2d70952d` / `73c4f17184ea0d50793418fb244b83562474f56051386d4b7b0182ca6050a270`.

## 2. Data and query audit

- [DIRECT] Source is Agent-STAR `TravelTotal_17K.jsonl` revision `d644daaffe2c6a1ec63d86c70fe4c639215f28ee`: 17,594 raw candidates; 17,486 unique accepted; 17,386 train candidates after 108 exclusions (86 duplicate constraints, 2 duplicate queries, 22 protected fingerprints). TravelPlanner contributes protected validation/test references and database protocol, not training queries. No artificial/manual seed set is recorded [MISSING].
- [DIRECT] Frozen selection: 1,835; Easy/Medium/Hard = 804/650/381; single/multi-city = 1,053/782; 3/5/7-day = 1,053/392/390; 238 unique origins and 235 unique destinations.
- [DIRECT] Constraints: budget 1,835; accommodation 717; dietary 504; transportation 381; multi-constraint 381. Max origin/destination shares are 2.6158% / 2.3978%.
- [DIRECT] Frozen test=100; protected references=1,360; protected comparisons=2,495,600; protected overlap=0; dev-training overlap=0; SFT/RL overlap=0.

## 3. Teacher distillation and filtering

- [DIRECT] Teacher Qwen3-30B-A3B-Instruct-2507 revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, temperature 0, TP=4, max model length 32,768, max_turns=13.
- [DIRECT] 1,820/1,835 successes (99.1826%), 15 failures: 11 model-call-limit and 4 timeout. Mean latency 33.172s.
- [DERIVED] Turns over 1,820 raw trajectories: mean=10.8242, std=2.7922, min=4.0000, p50=12.0000, p90=13.0000, max=13.0000; reported mean tool calls=10.8890; total calls=19818.
| Tool | Calls | Share |
|---|---:|---:|
| `poi_search` | 8467 | 42.72% [DERIVED] |
| `web_search` | 5943 | 29.99% [DERIVED] |
| `flight_search` | 3091 | 15.60% [DERIVED] |
| `route_planning` | 1338 | 6.75% [DERIVED] |
| `web_scraping` | 374 | 1.89% [DERIVED] |
| `nearby_search` | 261 | 1.32% [DERIVED] |
| `train_search` | 213 | 1.07% [DERIVED] |
| `weather` | 131 | 0.66% [DERIVED] |
- [DIRECT] Tool errors: repeated_action=198, invalid_arguments=4, timeout=74, provider client/server=23/37, tool-budget-exceeded=886, answer_required=118. Malformed-output count is not separately recorded [MISSING].
- [DIRECT] Replay v4: 4,278 unique requests, 4,278 hits, 0 misses, 0 public/live requests; provider errors replayed=89.
- [DIRECT] Filter funnel: 1,820 raw → 1,699 rule accepted (121 repeated-action rejects; 93.35% retained) → 1,699 format accepted (0 rejects) → 1,463 Judge accepted (236 below-threshold; 86.11% of Judge input) → 881 SFT + 200 RL. Judge threshold mean≥4.0 and every dimension≥3.0. Separate incomplete/tool-misuse/factual-safety/format rejection counts are not present [MISSING].

## 4. SFT data, token lengths, configuration, and metrics

- [DIRECT] SFT/RL/held-out counts = 881/200/64. SFT/RL overlap=0; protected benchmark overlap=0; dev-training overlap=0.
- [DIRECT] Token lengths (881 rows): min 2,865; mean 5,477.25; median 5,714; p75 6,161; p90 6,479; p95 6,726; p99 7,106; max 7,401; >8K/>16K/>24K/>32K all 0. By difficulty: Easy n=419 mean=5,619.98 max=7,401; Medium n=310 mean=5,397.77 max=7,144; Hard n=152 mean=5,245.90 max=7,363.
- [DIRECT] Production: Qwen3-4B; LoRA rank/alpha/dropout=128/256/0.05; all-linear; bf16; max_length=32,768; 2 epochs; LR=2e-5 cosine; warmup=0.05; weight decay=0.01; per-device batch=1; grad accumulation=4; 4-rank global batch=16; gradient checkpointing, FlashAttention, padding_free, Liger enabled; packing disabled; SP=1; no DeepSpeed/offload; seed/data_seed=20260819. Optimizer name is not explicitly frozen in `configs/sft/production.yaml` [MISSING].
- [DIRECT] GPUs 0–3, four-rank DDP, RTX 4090 24,564 MiB each. GPU0 carried ~4.5 GiB pre-existing external allocation; no external process was killed. The separately verified 32K fallback used SP4 + ZeRO-3 CPU offload, but production used DDP because max row=7,401.
- [DIRECT] 110 optimizer steps; 881 samples/epoch; 1,762 sample-presentations [DERIVED]; runtime 1,034.157s; mean 9.401s/step [DERIVED]; 1.704 samples/s and 0.106 steps/s (trainer log); peak reported GPU memory 16.81 GiB. Total processed tokens and exact tokens/s are absent [MISSING]; peak CPU RAM is absent [MISSING].
- TRAIN [DIRECT]: aggregate loss=0.2780154, last loss=0.22089405, exact full-corpus token accuracy=610,305/656,433=92.9729%.
- VALIDATION [DIRECT]: immutable held-out 64 rows, loss=0.223847093; accuracy=43,844/47,251=92.7896%. This later artifact supersedes the earlier training report's N/A statement.

## 5. Formal GRPO architecture and parameters

- [DIRECT] Actual topology is **1+3**, not 2+6: GPU6 rollout; GPUs1,2,3 full-parameter trainer; GPU1 also fixed Judge. RTX 4090 24,564 MiB. GPUs0/4/5/7 stayed unused due external workloads (4,498/5,940/19,392/11,474 MiB).
- [DIRECT] Rollout: vLLM 0.8.5.post1, server mode, TP=1, memory utilization=0.60, max_model_len=8192, max_num_seqs=16, async engine requested, eager mode, logprobs enabled; weight sync through `trl.scripts.vllm_serve.WeightSyncWorkerExtension`; mean rollout latency=25.2268s.
- [DIRECT] Trainer: world size=3, full-parameter bf16, ZeRO-3 with CPU optimizer offload, no model offload, SP=1, gradient checkpointing, FlashAttention.
- [DIRECT] 200 RL queries; G=6; max_turns=13; temperature=0.9; max completion per round=512; LR=5e-7; beta/KL=0.04; clip epsilon=0.2; one epoch; batch/device=1; grad accumulation=2; seed=20260819. top_p/top_k and explicit generation batch are absent [MISSING].
- [DIRECT] 200/200 optimizer steps completed and 200×6=1,200 rollouts [DERIVED]. Checkpoint every 10 steps, retention 10 (checkpoint-110…200). Attempt 1 stopped before rollout on Arrow datetime schema; one successful restart. `completions.jsonl` contains one duplicate shutdown row for step 200, explicitly de-duplicated by step in this audit.
- [DIRECT] Runtime=25,796.0687s (7h09m58s); rollout+trainer nominal GPU-hours=28.66 [DERIVED]; peak VRAM MiB GPU1/2/3/6=22,388/11,004/11,044/15,352; host RAM peak=255.17 GiB; project RSS peak=175.30 GiB.

## 6. Six-reward audit

Formal sum is `1*valid_agent + 1*valid_tool + 1*final_boundary + 0.5*stage + 0.5*efficiency + 1*weighted_judge`; every component's raw output is [0,1]. Judge ORM internally applies its curriculum before the trainer weight.

| Reward | Definition / inputs | Trainer weight | Dynamic | Raw-output statistics over 1,200 rollouts |
|---|---|---:|---|---|
| `ValidAgentStepORM` | fraction of legal user/assistant/tool transitions; trajectory messages | 1 | no | mean=0.999226; std=0.005200; min/max=0.964286/1.000000; nonzero=100.0000% [DERIVED] |
| `ValidToolCallORM` | fraction of emitted calls with registered name + JSON-schema-valid args; messages/tool schemas | 1 | no | mean=1.000000; std=0.000000; min/max=1.000000/1.000000; nonzero=100.0000% [DERIVED] |
| `FinalAnswerBoundaryORM` | exactly one final answer and it is terminal; messages | 1 | no | mean=0.978333; std=0.145593; min/max=0.000000/1.000000; nonzero=97.8333% [DERIVED] |
| `StageAwarenessORM` | tool→observation progression plus evidence-aware terminal answer; messages | 0.5 | no | mean=1.000000; std=0.000000; min/max=1.000000/1.000000; nonzero=100.0000% [DERIVED] |
| `ToolEfficiencyORM` | penalizes duplicate/repeated actions and missing final; messages | 0.5 | no | mean=0.974810; std=0.102185; min/max=0.000000/1.000000; nonzero=99.9167% [DERIVED] |
| `FixedLocalJudgeORM` | normalized fixed-local-Judge score × linear step curriculum; trajectories/trainer_state | 1 | yes, 0.2→1.0 by global_step/200 | mean=0.546873; std=0.228052; min/max=0.108571/0.992000; nonzero=100.0000% [DERIVED] |

## 7. Judge and reward-curve audit

- [DIRECT] Judge model is local Base Qwen3-4B-Instruct-2507, Transformers/uvicorn service on GPU1; deterministic generation (temperature flag ignored under greedy generation); prompt `prompts/judge/v1.txt`, hash `a67adfec206097df6e39c75b92c6c74db8b5877fe569906ccbde6d5950626e38`; timeout=300s; no retry loop in reward ORM; exceptions fall back to 0.
- [DIRECT] Curriculum is **0.2→1.0**, not 0.60→0.75: `weight = 0.2 + 0.8*clip(global_step/max_steps,0,1)`, keyed to global_step.
- [DIRECT] 600 batched HTTP Judge calls returned 200; they yielded 1,200 rollout scores [DERIVED]; non-200=0, fallback-zero=0, score success=1,200/1,200=100%. Timeout/retry counts are 0/0.
- [DERIVED] Raw normalized Judge score: full mean=0.8969, std=0.0854, min=0.4778, p50=0.9088, p90=0.9940, max=0.9960; first 20 steps mean/std/p50/p90=0.8211/0.0978/0.8408/0.9552; last 20=0.9304/0.0696/0.9391/0.9958.
| Step segment | Total reward mean | Std |
|---|---:|---:|
| 1-20 | 4.048950 | 0.409763 [DERIVED] |
| 21-40 | 4.185966 | 0.338748 [DERIVED] |
| 41-80 | 4.351511 | 0.224316 [DERIVED] |
| 81-120 | 4.550075 | 0.064753 [DERIVED] |
| 121-160 | 4.681817 | 0.094851 [DERIVED] |
| 161-200 | 4.858328 | 0.074095 [DERIVED] |
- [DERIVED] First 10%→last 10% total reward: 4.048950→4.892479; actual all-run mean=4.511838. This is not 0.43→0.61.

## 8. GRPO learning signal and tool behavior

- [DERIVED] Advantages: mean=-0.0001, std=0.9022, min=-2.0408, p50=0.3315, p90=0.9400, max=2.0128; non-zero=97.0000%. KL: mean=0.008048, p95=0.023628, max=0.046709. Policy loss mean=0.00037557; grad norm mean=2.597905; mean completion length=636.57 tokens.
| Signal | Early (steps 1–20) | Mid (91–110) | Late (181–200) |
|---|---:|---:|---:|
| reward mean | 4.026370 | 4.151608 | 4.886450 [DERIVED] |
| advantages mean | 0.026678 | -0.000006 | -0.058603 [DERIVED] |
| kl mean | 0.000285 | 0.005496 | 0.021906 [DERIVED] |
| loss mean | 0.000020 | 0.000222 | 0.001119 [DERIVED] |
| grad_norm mean | 2.360831 | 2.469034 | 3.178758 [DERIVED] |
| completion_length mean | 706.375031 | 638.608357 | 519.466687 [DERIVED] |
- [DIRECT] Formal rollout behavior: mean turns=9.1708, mean tool calls=8.1925, final-answer=1,113/1,200=92.75%, loop/repeated-action=70/1,200=5.8333%, invalid tool errors=0. Snapshot misses=1,398; 1,350 recovered (96.5665% [DERIVED]); 48 final failures; public requests=0.
- [DIRECT] Formal tool calls: POI 4,950; flight 2,230; web search 1,669; route 670; nearby 153; web scrape 91; train 45; weather 23. The training artifacts do not contain a strictly comparable Base/SFT rollout on the same 200 queries [MISSING], so no causal before/after claim is made from these counts.

## 9. Internal frozen evaluation (n=16)

| Metric | Base | SFT | RL |
|---|---:|---:|---:|
| Task success | 14/16=87.5% | 11/16=68.75% | 16/16=100% |
| Constraint satisfaction | 14/16=87.5% | 11/16=68.75% | 16/16=100% |
| Valid tool calls | 16/18=88.8889% | 115/115=100% | 16/16=100% |
| Invalid tool calls | 2/18=11.1111% | 0/115=0% | 0/16=0% |
| Final answer | 14/16=87.5% | 11/16=68.75% | 16/16=100% |
| Loop | 0/16=0% | 5/16=31.25% | 0/16=0% |
| Avg turns / calls | 2.0 / 1.0 | 7.875 / 7.1875 | 2.0 / 1.0 |
| Pointwise Judge (1–5) | 4.7768 | 4.0714 | 4.9911 |
All rows above are [DIRECT], except tool-call numerators/denominators reconstructed exactly from rate and call totals [DERIVED]. SFT underperformed Base on success, final-answer, loops, efficiency, and Judge; this negative result must remain visible.

## 10. Pairwise Judge

- [DIRECT] Fixed Base Judge; prompt hash `59584211aa357e8aa57096b66dedc3ffb6e7c866c5b1839ac70bd13aea8ca6ec`; 16 queries/pair, both A/B and B/A; 96 calls, 96 successes, 0 failures. Timeout and retry counters are not separate fields [MISSING], but no error record is present. Path weight=30% over 3 dimensions; final answer=70% over 4 dimensions.
| Pair (left perspective) | Win/Tie/Loss | Score | Position disagreement |
|---|---:|---:|---:|
| base vs sft | 10/4/2 | 0.7500 | 4/16 |
| base vs rl | 10/4/2 | 0.7500 | 4/16 |
| sft vs rl | 0/10/6 | 0.3125 | 10/16 |
- [DIRECT] Total position disagreement=18. Pairwise metrics are Judge preferences and must not replace deterministic benchmark results.

## 11. TravelPlanner external validation

### Invalid / adapter-contaminated evaluation (preserved)

| Model | CS micro | CS macro | Hard micro | Hard macro | Final pass |
|---|---:|---:|---:|---:|---:|
| base | 40/1440=2.7778% | 0/180 | 0/420 | 0/180 | 0/180 |
| sft | 27/1440=1.8750% | 0/180 | 0/420 | 0/180 | 0/180 |
| rl | 411/1440=28.5417% | 0/180 | 0/420 | 0/180 | 0/180 |
### Adapter-corrected frozen-output evaluation (valid postprocess attempt)

| Model | CS micro | CS macro | Hard micro | Hard macro | Final | Parsed/schema overall | Schema valid given parsed | Complete | Parser fail | Sandbox | Hard executed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 30/1440=2.0833% | 0/180 | 0/420 | 0/180 | 0/180 | 8/180=4.4444% | 8/8=100.00% | 0/180 | 172/180=95.56% | 2/180 | 0/180 |
| sft | 21/1440=1.4583% | 0/180 | 0/420 | 0/180 | 0/180 | 6/180=3.3333% | 6/6=100.00% | 0/180 | 174/180=96.67% | 0/180 | 0/180 |
| rl | 429/1440=29.7917% | 0/180 | 0/420 | 0/180 | 0/180 | 106/180=58.8889% | 106/106=100.00% | 0/180 | 74/180=41.11% | 13/180 | 0/180 |
All values in the two tables are [DIRECT]. Corrected RL CS micro improved 411→429 passes, but completeness remained 0/180, hard constraints executed 0/180, and Final Pass 0/180. Therefore external generalization is not demonstrated.

## 12. Adapter scientific audit

- [DIRECT] 540 raw answers (180×3), all byte hashes unchanged; ordered hashes Base/SFT/RL `d80c5cfab41b4d5adba4af878ab1dc7ca7c412a2acdb8c3aa2dacd48317d1bd8` / `3b2df6e9d86f9eb887c9db0c77c1d846d53c6ef1c1c76689624b437424b44683` / `2d9be2cbb64d74b07f7bff27a08868bd8d285075fc592c8b34edeb259fe04d97`.
- [DIRECT] Model inference runs=0; formal postprocess attempts=1; deterministic no-LLM adapter hash `73c4f171...a270`; parser prompt/model absent by design.
- [DIRECT] Fixed manual audit n=60: faithful extraction 40, field loss 20, hallucinated field 0, wrong field assignment 0, entity mutation 0. Official schema-valid given parsed=100% for all models, but most prose could not be faithfully extracted and no task passed completeness.

## 13. Failures and negative results

| Problem | Root cause | Fixed? | Final impact |
|---|---|---|---|
| SFT < Base internally | SFT produced long loops and fewer terminal answers | No retraining (frozen) | 68.75% vs 87.5% task success |
| TravelPlanner Final Pass 0 | Incomplete official daily fields; sandbox/entity and planning failures | No | 0/180 for all models |
| Adapter v1 contamination | Itinerary prose collapsed into attraction field | Adapter-only v2 repair | Old results retained invalid; corrected results still 0 final |
| Corrected parser failures | Generic parser could not faithfully map unstructured prose | Partly | Base/SFT/RL fail 172/174/74 |
| Hard constraints not executed | Completeness gate never passed | No | 0/420 hard micro |
| GRPO snapshot misses | Frozen snapshot lacks generated request variants | Replay-v2 recovery | 1,350/1,398 recovered; 48 failed |
| Pilot Judge=0 | CPU Judge exceeded 300s timeout | GPU fixed Judge for probe/formal | Formal fallback=0 and 100% scoring success |
| Formal restart | Arrow datetime in raw source schema | Generated frozen JSON-safe formal view | One pre-rollout restart; no lost optimization |
| GPU constraints | Four GPUs occupied by other users | Safe 1+3 topology | No preemption; 200 steps completed |

## 14. Engineering, tests, and reproducibility

- [DIRECT] Final gate: pytest 110/110, Ruff passed, scientific audit passed, adapter scientific checks passed, contamination overlap=0, frozen hashes recorded, clean worktree before audit.
- [DERIVED] Git-tracked engineering scope before these reports: 111 non-test Python files / 15,621 LOC; 33 test files / 2,081 LOC; 33 config JSON/YAML files; 75 tracked scripts; 52 tracked report files; 32 commits. LOC counts include tracked vendored official evaluator code where present and should not be used as personal-authorship claims.
- [DIRECT] Distillation replay public requests=0; formal GRPO public requests=0; all model/data/config hashes listed above. Repository-wide disk usage is ~2.19 TiB [DERIVED] but includes models, checkpoints, databases, and outputs, so it is an operational footprint—not authored code size.

## 15. Time and resource cost

| Stage | Wall time / resource evidence | Status |
|---|---|---|
| Distillation | mean latency 33.172s/trajectory; exact end-to-end wall clock absent | [DIRECT]/[MISSING] |
| SFT production | 1,034.157s; 4 GPUs; nominal 1.149 GPU-h [DERIVED] | [DIRECT] |
| GRPO pilot | 16 steps/192 rollouts; exact wall time absent | [DIRECT]/[MISSING] |
| Formal probe | 3 steps/18 rollouts; exact wall time absent | [DIRECT]/[MISSING] |
| Formal GRPO | 25,796.0687s; four project GPUs; nominal 28.66 GPU-h | [DIRECT]/[DERIVED] |
| Internal/external evaluation | timestamps and per-query latency exist, but unified end-to-end wall time is not frozen | [MISSING] |

## 16. Reference-claim cross-check

| Reference claim | Our actual result | Match? | Can write? | Evidence |
|---|---|---|---|---|
| 306 artificial seeds | No artificial seeds recorded | MISSING | No | data/manifests/data_snapshot.json |
| 1,835 queries | 1,835 frozen | MATCH | Yes | train_queries_1835_manifest.json |
| 1,831 trajectories | 1,820 successful / 15 failed | DIFFERENT | Write actual | distillation_1835.json |
| 1,081 filtered | 1,463 Judge accepted | DIFFERENT | Write actual | cleaning/stats.json |
| 881 SFT | 881 | MATCH | Yes | sft_agent_881_manifest.json |
| 200 RL | 200 | MATCH | Yes | processed_split_manifest.json |
| 32K | max_length=32,768; real data max=7,401; 32K probe passed | QUALIFIED MATCH | Yes with caveat | sft_memory_probe.md |
| rank128 | 128 | MATCH | Yes | configs/sft/production.yaml |
| lr=8e-6 | 2e-5 SFT; 5e-7 GRPO | DIFFERENT | Write actual | configs/sft/production.yaml; formal_4gpu.yaml |
| 2 epochs | 2 | MATCH | Yes | training_metrics.json |
| 420 steps | 110 SFT; 200 GRPO | DIFFERENT | Write actual | training_metrics.json |
| val loss=.6143 | 0.223847 | DIFFERENT | Write actual | sft_heldout_validation_64/metrics.json |
| val acc=83.29% | 92.7896% | DIFFERENT | Write actual | same |
| 2 rollout + 6 trainer | 1 rollout + 3 trainer | DIFFERENT | Write actual | formal_grpo_topology.json |
| G=6 | 6 | MATCH | Yes | formal_4gpu.yaml |
| max_turns=13 | 13 | MATCH | Yes | formal_4gpu.yaml |
| Judge 0.60→0.75 | 0.2→1.0 | DIFFERENT | Write actual | swift_plugin.py |
| reward 0.43→0.61 | 4.04895→4.89248 | DIFFERENT | Write actual with scale | completions.jsonl [DERIVED] |
| GPT Judge +13.46% | No GPT experiment | NOT REPRODUCED | No | [MISSING] |
| Gemini 8.341 vs 7.802 | No Gemini experiment | NOT REPRODUCED | No | [MISSING] |

## 17. Evidence grades and additional defensible highlights

- Grade A: counts/hashes/configs/final metrics, 110/110 tests, 1,200 rollouts, 0 public requests, exact validation numerators, adapter audit.
- Grade B: 19,818 summed tool calls, 1,762 processed sample-presentations, 28.66 nominal formal GPU-hours, reward segment statistics, recovery percentage, tracked LOC.
- Grade C: narrative root-cause statements in reports without a machine counter; use only with path citation.
- Grade D: all reference-only numbers with no artifact: 306 seeds, GPT/Gemini gains, 0.60→0.75, 420 steps.
- Additional A/B highlights: 4,278/4,278 replay hit gate; 100% formal Judge score coverage; 97.0149% non-zero advantages; RL reduced internal loop 31.25%→0% vs SFT; adapter repair raised RL commonsense passes 411→429 while preserving 540 raw answers; checkpoint-200 verified parameter changes; zero protected overlap across 2,495,600 comparisons.
