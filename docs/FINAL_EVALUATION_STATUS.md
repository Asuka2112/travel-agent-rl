# Current evaluation status and publication scope

Updated 2026-10-19 (Asia/Shanghai). The internal dual-Judge completion was accepted
on 2026-10-18. This update publishes the latest progress and auditable result
summaries; it does not merge the server's entire experimental code/data branch
into the GitHub default branch, rerun training, or upload checkpoints.

## Current result

- Selected internal delivery recommendation: **GRPO-v1**.
- Frozen test set: 80 queries; four-model structural comparison.
- Judge pairs: Base/SFT-v2, Base/GRPO-v1, SFT-v2/GRPO-v1; AB and BA each.
- GPT: 480 usable score slots = 473 original valid + 7 manual supplements.
- Unified Gemini 3.1 Pro: 480 = 451 original valid + 29 manual supplements.
- All six Judge/pair combinations have 80 bidirectionally scored queries.
- Original primary failures are preserved. Supplements are explicitly labeled in
  a derived overlay, not silently substituted into the immutable primary run.
- Training, new judging, and automatic continuation remain stopped. No new paid
  API calls or checkpoint changes were made during completion/publication.

## Claims and limits

960 scored slots do not establish 960 independent calls. Manual batches shared
context; exact provider identity/settings and transmitted bytes were not
independently verified. The 30/70 weighting and reference input contents are
unchanged. The machine-readable metrics include position disagreement, Judge
agreement, weighted differences, Wilson intervals and query bootstrap intervals.
Those intervals do not model same-batch correlation.

The strict sensitivity summary excludes all 36 new manual supplements and the
7 earlier Gemini shared/manual protocol exceptions. Its preference conclusion
still supports GRPO-v1; full independent coverage is not claimed. Human-review
materials do not constitute completed human ratings. Historical external
evaluation remains separately documented, not an internal metric or a resume
completion requirement. There is no external-generalization success claim.

## Published evidence versus server archive

GitHub includes the unchanged final report 063, compact aggregate metrics,
strict sensitivity metrics, an integrity-audit summary, and a short resume
project summary. Per-query rationales, user attachments, browser account/session
captures, model outputs, checkpoints, and frozen training/test data are not added
by this documentation update. Their omission is not deletion from the archive.

The source implementation/evidence branch on the server is `experiments/sft-v2`,
at completion commit `75c081ca03d968da54c57c0bb31733428b1cd709`.
The full supplement delivery is retained under
`outputs/experiments/final_evaluation_20261016/manual_completion_063/`, with the
original primary browser records in the adjacent `browser_judges/` directory.
Paths beginning `results/` in report 063 refer to that archive, not files newly
published on GitHub. The four-model source outputs and original training hashes
remain in their original server locations.

| Integrity object | SHA256 |
| --- | --- |
| Frozen 480-request queue | `ede327adfae4423f4916cfa34ce05ea1897135a2c687fc527fa5889f2ccee380` |
| Original 960-record primary inventory | `7a01d3384afaf6126a2c3b8da5951c2175293549041b36145793f22444913ccc` |
| Final lightweight audit ZIP 063 | `2c0809efb750179cf2bb0a10b0eae504dfb71a33aa8e24bbe05c59e5421e84dc` |

Both original primary and derived-overlay inventories, as well as archive
integrity, were checked on the server during delivery. Reference hashes verify
the preserved artifacts, not unobserved manual browser transmission.

## Historical material

The previous README is retained verbatim in
[docs/history/README_before_final80.md](history/README_before_final80.md).
The original `FINAL_REPORT.md`, existing training reports, and external benchmark
audits remain unchanged. Earlier 16-query results and snapshot-v4 narratives must
not be relabeled as the latest 80-query evidence: the actual saved final-80
trajectories were audited against snapshot v1, with 2,185 normalized observations
replay-consistent.

The latest completion-stage validation passed 17 offline Python regression tests.
The public-facing documentation update itself is checked for metric agreement,
relative-link integrity, accidental secrets/session data, and preserved history.
