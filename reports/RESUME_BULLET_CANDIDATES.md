# Resume Bullet Candidates

Every bullet includes an evidence locator. A versions emphasize implementation, B versions emphasize measured outcome, and C versions are interview-friendly. Claims intentionally avoid external benchmark success.

## 数据准备

- A版：构建 Agent-STAR 旅行 Query 的固定哈希分层采样与污染审计管线，将 17,594 条原始候选清洗为 17,386 条训练候选并冻结 1,835 条 Easy/Medium/Hard Query。 [EVIDENCE: data/processed/data_stats.json; data/manifests/train_queries_1835_manifest.json]
- B版：在 2,495,600 次 protected-reference 比对中实现 0 overlap，并冻结 1,835 条可复现训练 Query。 [EVIDENCE: data/manifests/processed_split_manifest.json:parent_contamination]
- C版：把“数据够不够多”转化为可核验的 selection manifest、hash、difficulty/day/city 分布与 contamination gate。 [EVIDENCE: data/manifests/train_queries_1835_manifest.json]

## SFT

- A版：基于 Qwen3-4B 实现 rank-128 LoRA Agent SFT，集成 8-tool Hermes schema、response loss mask、bf16/FlashAttention/Liger/gradient checkpointing，并验证 SP4+ZeRO-3 的 32K fallback。 [EVIDENCE: configs/sft/production.yaml; reports/sft_memory_probe.md]
- B版：在 881 条轨迹上完成 2 epochs/110 steps（17.24 min，峰值 16.81 GiB），held-out 64 条 loss=0.223847、token accuracy=43,844/47,251=92.79%。 [EVIDENCE: outputs/final/training_metrics.json:sft; outputs/eval/sft_heldout_validation_64/metrics.json]
- C版：根据实测 token max=7,401 选择四卡 DDP 生产训练，同时保留通过真实 32K step 的 SP4+ZeRO-3 CPU-offload 路径。 [EVIDENCE: reports/sft_token_length_report.md; reports/sft_memory_probe.md]

## Agentic RL

- A版：实现 vLLM rollout server + 三卡 ZeRO-3 full-parameter trainer 的多轮 GRPO 系统，支持 G=6、13 turns、rollout logprobs/info 与 checkpoint/resume。 [EVIDENCE: configs/grpo/formal_4gpu.yaml; outputs/grpo/formal_4gpu_v1/rollout.log]
- B版：在共享 4×RTX 4090 安全拓扑上完成 200 steps/1,200 rollouts，97.01% advantage non-zero、KL mean=0.00805、无 NaN/OOM/NCCL。 [EVIDENCE: outputs/final/training_metrics.json:formal_grpo_200]
- C版：面对 4 张被他人占用的 GPU，动态选择 1 rollout+3 trainer，不抢占外部进程并完成 7h09m58s 正式训练。 [EVIDENCE: outputs/grpo/resource_snapshots/formal_grpo_topology.json; outputs/final/training_metrics.json]

## Reward / Judge

- A版：设计六路 trajectory reward（协议、tool schema、terminal boundary、stage、efficiency、local Judge），Judge 权重按 global_step 从 0.2 线性升至 1.0。 [EVIDENCE: src/travel_agent_rl/rewards/components.py; src/travel_agent_rl/grpo/swift_plugin.py:FixedLocalJudgeORM]
- B版：600 次本地 Judge batch 请求为 1,200 条 rollout 全量打分，成功率 100%、fallback=0；总 reward 前/后 10% 从 4.0489 升至 4.8925。 [EVIDENCE: outputs/grpo/formal_4gpu_v1/judge.log; completions.jsonl; outputs/final/resume_evidence.json:grpo]
- C版：把 LLM-as-Judge 固定为本地 Base 模型和 hash-pinned prompt，并用 timeout→0 的确定性 fallback 控制在线奖励失效模式。 [EVIDENCE: src/travel_agent_rl/grpo/swift_plugin.py; outputs/final/final_metrics.json:judge_prompt_sha256]

## 模型评测

- A版：建立 Base/SFT/RL 同协议 16-query rule-based + pointwise + 双向 pairwise 评测，并单独接入官方 TravelPlanner evaluator。 [EVIDENCE: outputs/final/final_metrics.json; outputs/final/pairwise_metrics.json; outputs/final/travelplanner_corrected_metrics.json]
- B版：内部冻结集 RL 达到 16/16 task success、16/16 valid tool calls、0/16 loops；pairwise 执行 96 次 Judge 调用且 0 failures。 [EVIDENCE: outputs/final/final_metrics.json; outputs/final/pairwise_metrics.json]
- C版：主动保留 TravelPlanner 0/180 Final Pass 负结果，并区分 rule-based、pointwise 与 position-sensitive pairwise 指标，避免用 Judge 偏好覆盖确定性结果。 [EVIDENCE: reports/TRAVELPLANNER_CORRECTED_VALIDATION.md; outputs/final/pairwise_metrics.json]

## 工程系统

- A版：实现冻结 snapshot replay、deterministic miss recovery、科学性审计、hash manifest 和 adapter-only postprocess freeze。 [EVIDENCE: reports/GRPO_REPLAY_V2_REGRESSION.md; outputs/final/travelplanner_adapter_v2/reproduction_manifest.json]
- B版：蒸馏阶段达到 4,278/4,278 snapshot hits；正式 GRPO 恢复 1,350/1,398 misses；全程 public network=0；最终测试 110/110。 [EVIDENCE: reports/distillation_1835.json:replay; outputs/final/training_metrics.json; outputs/final/scientific_audit.json]
- C版：对 adapter 修复执行非 validation 单测→60 样本 faithfulness audit→冻结 hash→一次正式 postprocess，确认 540 条 raw answers 未变化。 [EVIDENCE: outputs/final/travelplanner_corrected_metrics.json:scientific_checks]
