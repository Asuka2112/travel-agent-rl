# Milestones

## M0 - Foundation (completed)

- [x] Audit OS, GPUs, CUDA, CPU, RAM, disk, Python, and Git.
- [x] Establish repository structure and reproducibility documents.
- [x] Implement unified schemas and deterministic eight-tool mock registry.
- [x] Add minimal CI and test suites.
- [x] Validate pinned CUDA 12.4 training dependencies and real vLLM inference.
- [x] Record exact package lock and model revision.

## M1 - Licensed data and frozen splits (completed)

- [x] Download license-cleared source revisions with checksums.
- [x] Normalize and deduplicate at least 1,500 candidate queries.
- [x] Exclude all official test examples from training candidates.
- [x] Create and freeze an independent test split of at least 80 examples.
- [x] Generate provenance, statistics, and contamination reports.

## M2 - Production tool environment (completed)

- [x] Add AMap, weather, Firecrawl/search adapters behind interfaces.
- [x] Add timeout, retry, rate-limit normalization, disk cache, and fixtures.
- [x] Build deterministic flight and train databases/simulators.
- [x] Implement parser, validator, repeat detection, and force-answer state machine.
- [x] Run credentialed live-tool smoke (`AMAP_API_KEY`, `FIRECRAWL_API_KEY`).

## M3 - Teacher distillation and filtering (running: dataset production)

- [x] Version teacher and judge prompts.
- [x] Implement concurrent resumable trajectory generation.
- [x] Implement rule, canonical-format, and structured judge stages.
- [ ] Produce SFT/RL datasets and full stage-by-stage statistics.

## M4 - LoRA SFT (planned)

- [ ] Run 100-sample smoke, then 100-step sanity validation.
- [ ] Run 2-epoch rank-128 32K LoRA SFT with evidence-based batching.
- [ ] Merge adapter and validate tool calls and answer termination.

## M5 - Online multi-turn GRPO (planned)

- [ ] Implement six separately logged/configurable reward components.
- [ ] Validate rollout -> tool -> reward -> backward -> sync -> next rollout.
- [ ] Scale only after the 3-turn/2-generation gate passes.
- [ ] Attempt full-parameter ZeRO-3 target with memory evidence.

## M6 - Frozen evaluation and report (planned)

- [ ] Evaluate Base, SFT, and RL on identical frozen queries.
- [ ] Run bidirectional pairwise judging to reduce position bias.
- [ ] Generate final metrics, failure analysis, costs, and report.
