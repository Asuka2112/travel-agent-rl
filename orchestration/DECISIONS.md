# Orchestration decisions

## 2026-08-24 takeover

- `orchestration/STATE.json` and `orchestration/HANDOFF.md` are the system of record.
- Replay Policy v2, frozen snapshot v4, completed data/SFT work, and earlier smoke runs will not be repeated.
- The requested `gpt-5.6-luna` routing probe self-identified only as `GPT-5 Codex`, not Luna. No tools or writes occurred. Bulk delegation is disabled to avoid unintended Sol usage.
- Pilot uses the four currently idle GPUs: GPU6 for rollout and GPU1/2/3 for the full-parameter ZeRO-3 trainer. GPUs 0/4/5/7 remain untouched because they contain external workloads.
- The first Pilot launch with gradient accumulation 2 was rejected before model loading because generation batch 6 is not divisible by G=4. Attempt 1 is preserved; gradient accumulation 4 gives generation batch 12 and passes the invariant.
- Real telemetry proves each Pilot optimizer step produces 12 rollouts from 3 unique queries (3 queries x G=4). Full coverage of all 48 frozen queries therefore requires 16 optimizer steps. The already-running four-step segment will finish naturally, then resume from checkpoint-4 to step 16 with identical scientific settings.
- Before every Probe/Formal launch, audit all GPU0-7 again. Prefer 2 rollout + 6 trainer when eight are idle-safe, then 1 + 6 with seven, else 1 rollout + all remaining idle-safe trainer GPUs. Log each GPU's external VRAM, selected role, and any unused reason. Never preempt external workloads.
- Pilot completed all 16 optimizer steps and exactly 48 unique queries x G=4 at checkpoint-16. The feasibility gate passed with public network count zero and real parameter changes.
- Pilot's fixed Base Judge was CPU-bound and every request exceeded its 300-second reward timeout, so its component deterministically fell back to zero. This limitation is retained in the Pilot record. Before the Formal Probe, the same immutable Base Judge will be hosted on a safe GPU and must return real structured scores without changing prompt, model, threshold, or reward semantics.
