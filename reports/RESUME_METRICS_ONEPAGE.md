# Resume Metrics One-Page

All claims below are artifact-backed; prefer A/B claims and preserve denominators.

1. [A] 17,594 raw Agent-STAR candidates → 17,386 train candidates → 1,835 frozen queries; 0 protected overlap across 2,495,600 comparisons.
2. [A] 1,820/1,835 teacher trajectories succeeded (99.1826%); 15 failures.
3. [B] 19,818 eight-tool calls; mean 10.889 calls/trajectory.
4. [A] Replay-only distillation: 4,278/4,278 snapshot hits and 0 public requests.
5. [A] Filter funnel: 1,820 → 1,699 rule/format valid → 1,463 Judge accepted → 881 SFT + 200 RL.
6. [A] SFT LoRA rank 128/alpha 256, bf16, 32K max_length, 2 epochs, global batch 16.
7. [A] Verified a real 32,768-token training step using SP4 + ZeRO-3 CPU offload; production data max was 7,401 tokens.
8. [A] Production SFT: 110 steps in 1,034.157s; peak 16.81 GiB.
9. [A] Train token accuracy 610,305/656,433=92.9729%.
10. [A] Held-out n=64: loss 0.223847; accuracy 43,844/47,251=92.7896%.
11. [A] Formal full-parameter GRPO: 200 queries × G=6 = 1,200 rollouts, max_turns=13.
12. [A] Actual topology: 1 rollout GPU + 3 trainer GPUs; completed in 7h09m58s without preempting external jobs.
13. [A] 97.0149% non-zero advantages; KL mean 0.00805; no NaN/OOM/NCCL.
14. [B] Total reward first/last 10%: 4.0489→4.8925.
15. [A/B] 600 successful batched Judge HTTP calls yielded 1,200/1,200 scores; 0 fallbacks.
16. [A] Snapshot miss recovery 1,350/1,398=96.5665%; 0 public requests.
17. [A] Internal n=16 RL: 16/16 task success, 16/16 valid tool calls, 0/16 loops; Base 14/16 and SFT 11/16.
18. [A] Pairwise: 96 Judge calls, 0 failures; 18 position disagreements explicitly reported.
19. [A] Corrected TravelPlanner RL: 429/1,440 commonsense micro=29.7917%, parsed 106/180, schema-valid 106/106.
20. [A] External limitation: completeness 0/180, hard executed 0/180, Final Pass 0/180—do not claim benchmark success.
21. [A] Adapter audit n=60: hallucinated fields=0, wrong assignments=0, entity mutations=0; 540 raw answers unchanged.
22. [A] Final verification: pytest 110/110, Ruff/scientific/contamination audits passed.
