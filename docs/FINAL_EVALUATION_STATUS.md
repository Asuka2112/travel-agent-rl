# Evaluation scope and archived provenance

Updated 2026-10-20 (Asia/Shanghai). The main project presentation uses **Base**,
**SFT**, and **GRPO** only. Its workflow is data construction, supervised
adaptation, agentic reinforcement learning, and frozen evaluation. GRPO denotes
the selected internal delivery model; SFT denotes the frozen supervised comparison.
These concise labels do not alter the underlying checkpoint identities or imply
that the displayed SFT checkpoint directly initialized the displayed GRPO model.

## Current evidence

- Frozen internal set: 80 queries, unchanged eight-tool protocol and candidate outputs.
- Judge comparisons: Base/SFT, Base/GRPO, and SFT/GRPO, each in AB and BA order.
- GPT: 480 usable slots, consisting of 473 original valid scores and 7 manual supplements.
- Gemini 3.1 Pro: 480 slots, consisting of 451 original valid scores and 29 manual supplements.
- Shared scoring rule: 30% path quality and 70% final-answer quality, using the original seven dimensions.
- The selected GRPO model has 68/80 internal structural successes; SFT has 55/80.
- Against SFT, GRPO receives GPT W/T/L 55/9/16 and Gemini W/T/L 57/15/8.

## What the labels do not claim

Structural success is not factual correctness or independently verified task
success. The 36 manual supplements came from shared-context batches; their actual
transmitted inputs and provider identities/settings were not independently
observed. Earlier shared/manual protocol exceptions remain in the provenance.
960 complete score slots do not establish 960 independent successful calls.

Excluding the manual supplements and earlier protocol exceptions still supports
the selected GRPO model in the strict earlier-stage sensitivity analysis. Its
coverage is incomplete; it must not be presented as a fully independent full-80
experiment. Query bootstrap intervals do not model within-batch correlation.

## Archive policy

This change simplifies the current presentation, **not the experimental record**.
The original reports, exact checkpoint/version mapping, unsuccessful experiments,
source metrics, frozen request queue, and original failed Judge records remain
unchanged. Machine-readable presentation labels are separate from archived source
keys; no frozen metric is rewritten. Older reports can still be inspected for
the exact technical lineage, but are not used as the primary resume narrative.

The original completion report and integrity summaries are retained in the
existing repository reports. Raw attachments and the derived completion overlay
remain in the server archive under
`outputs/experiments/final_evaluation_20261016/manual_completion_063/`, with the
original primary records in the adjacent `browser_judges/` directory. Full raw
browser/session captures, checkpoints, and additional training/test data are not
uploaded by this documentation update.

| Integrity object | SHA256 |
| --- | --- |
| Frozen 480-request queue | `ede327adfae4423f4916cfa34ce05ea1897135a2c687fc527fa5889f2ccee380` |
| Original 960-record primary inventory | `7a01d3384afaf6126a2c3b8da5951c2175293549041b36145793f22444913ccc` |
| Completion audit archive | `2c0809efb750179cf2bb0a10b0eae504dfb71a33aa8e24bbe05c59e5421e84dc` |

Human-review materials are not completed human ratings. Historical external
evaluation is retained separately and is not an internal success metric or an
additional resume-completion task. No external-generalization improvement is claimed.

The current update changes presentation only: no retraining, new inference,
checkpoint alteration, paid API calls, data deletion, or Git history rewrite.
Experiments and automatic continuation remain stopped.
