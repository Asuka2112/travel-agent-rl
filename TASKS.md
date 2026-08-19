# Milestones

## M0 - Foundation (running)

- [x] Audit OS, GPUs, CUDA, CPU, RAM, disk, Python, and Git.
- [x] Establish repository structure and reproducibility documents.
- [x] Implement unified schemas and deterministic eight-tool mock registry.
- [x] Add minimal CI and test suites.
- [ ] Validate pinned CUDA 12.4 training dependency candidate on the server.
- [ ] Record exact package lock and model revision.

## M1 - Licensed data and frozen splits (planned)

- [ ] Download license-cleared source revisions with checksums.
- [ ] Normalize and deduplicate at least 1,500 candidate queries.
- [ ] Exclude all official test examples from training candidates.
- [ ] Create and freeze an independent test split of at least 80 examples.
- [ ] Generate provenance, statistics, and contamination reports.

## M2 - Production tool environment (planned)

- [ ] Add AMap, weather, Firecrawl/search adapters behind interfaces.
- [ ] Add timeout, retry, rate-limit normalization, disk cache, and fixtures.
- [ ] Build deterministic flight and train databases/simulators.
- [ ] Implement parser, validator, loop detection, and force-answer state machine.

## M3 - Teacher distillation and filtering (planned)

- [ ] Version teacher and judge prompts.
- [ ] Implement concurrent resumable trajectory generation.
- [ ] Implement rule, canonical-format, and structured judge stages.
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

