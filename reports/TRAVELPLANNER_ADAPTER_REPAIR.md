# TravelPlanner Adapter Repair

Generated: 2026-08-25T18:20:43.006440+00:00

This was an adapter-only experiment over 540 frozen final answers. No model inference, training, prompt change, tool call, plan repair, entity replacement, or constraint solving occurred.

## Root cause and contract

Adapter v1 recognized day headings but, when prose lacked explicit `Field:` labels, placed the entire day body in `attraction`. It therefore discarded `current_city`, transportation, meals, and accommodation as separate official fields. The exact official contract and field mapping are frozen in `docs/TRAVELPLANNER_ADAPTER_SPEC.md`.

Adapter v2 is deterministic and receives only the raw final answer. It uses no query, task ID, database, evaluator verdict, or parser LLM. Missing content remains `-` or a parser failure; alternatives are not resolved and absent entities are never supplied.

## Independent tests and audit

The pre-validation suite covered the official 180-line `example_evaluation.jsonl`, an unrelated synthetic itinerary, and a project-native non-TravelPlanner trajectory. The unmodified official `format_check.py` completed on the official example and synthetic submission.

The fixed-seed audit reviewed 20 Base, 20 SFT, and 20 RL answers. Candidate history 001-008 is retained. Final counts were:

| Audit label | Count |
|---|---:|
| faithful extraction | 40 |
| field loss | 20 |
| hallucinated field | 0 |
| wrong field assignment | 0 |
| entity mutation | 0 |

Frozen adapter SHA-256: `73c4f17184ea0d50793418fb244b83562474f56051386d4b7b0182ca6050a270`. Parser model and parser prompt hash are `null` because no LLM parser is used.

## One-shot execution

The adapter was committed and hash-frozen before the only formal 540-answer postprocess. Each model then received exactly one unmodified official `eval.py --set_type validation` run. The lock records one generation attempt and zero model-inference runs. All exact raw-answer hashes matched before and after the run.

No post-score adapter edit or evaluator rerun was performed.
