# Interview Knowledge Map

每个术语均绑定本项目事实；不要把可选/探针拓扑说成正式训练拓扑。

## LoRA

1. 一句话：低秩增量参数微调。
2. 本项目：Qwen3-4B all-linear rank128/alpha256。
3. 设计理由：在较低显存下做领域行为适配。
4. 替代方案：full fine-tune、QLoRA、adapter。
5. 可能追问：为何 rank128；trainable 参数比例？
6. 证据：`configs/sft/production.yaml; checkpoint-110/adapter_config.json`。

## 32K context

1. 一句话：单序列最多 32,768 token。
2. 本项目：配置保留 32K，真实数据 max=7,401；另做真 32K step。
3. 设计理由：确保不截断并验证未来长轨迹路径。
4. 替代方案：truncate/packing/rope scaling。
5. 可能追问：配置 32K为何数据不到8K？
6. 证据：`reports/sft_token_length_report.md; reports/sft_memory_probe.md`。

## Sequence Parallel

1. 一句话：沿序列维切分激活。
2. 本项目：32K probe 使用 SP4；生产 SP1。
3. 设计理由：降低单卡长序列激活内存。
4. 替代方案：tensor/pipeline/context parallel。
5. 可能追问：SP 与 DDP/TP 区别？
6. 证据：`configs/sft/probe_32768_sp4.yaml; sft_memory_probe.md`。

## ZeRO-3

1. 一句话：分片参数、梯度和优化器状态。
2. 本项目：GRPO 三卡 full-parameter；32K SFT fallback。
3. 设计理由：4B 全参 RL 适配 24GB 卡。
4. 替代方案：FSDP、ZeRO-2。
5. 可能追问：为什么 SFT生产不用、GRPO用？
6. 证据：`configs/sft/zero3_cpu_offload.json; configs/grpo/formal_4gpu.yaml`。

## CPU offload

1. 一句话：把优化器/状态移到主存。
2. 本项目：GRPO optimizer offload；峰值项目 RSS 175.3 GiB。
3. 设计理由：以主存换显存。
4. 替代方案：NVMe offload/FSDP CPU offload。
5. 可能追问：速度瓶颈和内存峰值？
6. 证据：`configs/sft/zero3_cpu_offload.json; training_metrics.json`。

## ReAct

1. 一句话：交错推理动作与工具观察。
2. 本项目：assistant tool call→tool response→next action。
3. 设计理由：支持多轮证据收集。
4. 替代方案：plan-and-execute、function calling。
5. 可能追问：如何防循环？
6. 证据：`src/travel_agent_rl/grpo/scheduler.py; rewards/components.py`。

## Tool Calling

1. 一句话：结构化选择工具和 JSON 参数。
2. 本项目：Hermes 8-tool schema + JSON Schema 验证。
3. 设计理由：可执行且可计分的 agent 行为。
4. 替代方案：纯文本 action、MCP。
5. 可能追问：invalid call 如何处理？
6. 证据：`src/travel_agent_rl/sft/data.py; valid_tool_call_reward`。

## MultiTurn Scheduler

1. 一句话：管理多轮生成、工具执行和终止。
2. 本项目：travel_agent_replay，max_turns=13。
3. 设计理由：把环境反馈纳入 rollout。
4. 替代方案：单轮生成、外部 agent loop。
5. 可能追问：forced terminal 如何记分？
6. 证据：`src/travel_agent_rl/grpo/scheduler.py`。

## GRPO

1. 一句话：同 prompt 多生成并做组内相对优化。
2. 本项目：G=6、200 query/step、1,200 rollout。
3. 设计理由：无需独立 value model。
4. 替代方案：PPO、RLOO、DPO。
5. 可能追问：reward scale、组内零方差？
6. 证据：`configs/grpo/formal_4gpu.yaml; trainer_state.json`。

## Group Advantage

1. 一句话：同组六个 reward 标准化形成 advantage。
2. 本项目：97.0149% non-zero。
3. 设计理由：提供相对学习信号。
4. 替代方案：value-function GAE、leave-one-out。
5. 可能追问：零方差组如何处理？
6. 证据：`completions.jsonl:advantages; feasibility.json`。

## KL

1. 一句话：约束策略偏离 reference。
2. 本项目：beta=.04，mean=.00805，p95=.02363。
3. 设计理由：抑制训练漂移。
4. 替代方案：adaptive KL、no-KL clipping。
5. 可能追问：为什么 KL 很低？
6. 证据：`configs/grpo/formal_4gpu.yaml; logging.jsonl:kl`。

## Reward Design

1. 一句话：把目标拆成六路可审计信号。
2. 本项目：协议/tool/final/stage/efficiency/Judge。
3. 设计理由：兼顾可执行性和答案质量。
4. 替代方案：单 reward、rule-only、Judge-only。
5. 可能追问：哪些 reward 饱和？
6. 证据：`src/travel_agent_rl/rewards/components.py; resume_evidence.json`。

## LLM-as-Judge

1. 一句话：固定模型评估轨迹/答案质量。
2. 本项目：本地 Base Judge，0.2→1.0 curriculum。
3. 设计理由：提供规则难覆盖的质量信号。
4. 替代方案：human labels、reward model、rules。
5. 可能追问：偏差、position bias、fallback？
6. 证据：`src/travel_agent_rl/grpo/swift_plugin.py; pairwise_metrics.json`。

## response_loss_mask

1. 一句话：只对 assistant 目标 token 计 SFT loss。
2. 本项目：ms-swift agent template/loss scale。
3. 设计理由：避免学习 user/tool observation token。
4. 替代方案：全 token LM loss。
5. 可能追问：mask 边界如何验证？
6. 证据：`data/manifests/sft_agent_881_manifest.json; tests/sft`。

## rollout_logprobs

1. 一句话：保存采样策略 token logprob。
2. 本项目：vLLM `--logprobs true --max_logprobs 1`。
3. 设计理由：支持 GRPO ratio/KL。
4. 替代方案：trainer 端重算。
5. 可能追问：权重同步后如何对齐？
6. 证据：`outputs/grpo/formal_4gpu_v1/rollout.log`。

## rollout_infos

1. 一句话：随 rollout 传递请求/轨迹元数据。
2. 本项目：scheduler 保存 messages/request ids/replay status。
3. 设计理由：reward 可访问完整轨迹。
4. 替代方案：仅 completion text。
5. 可能追问：schema/序列化 bug？
6. 证据：`src/travel_agent_rl/grpo/scheduler.py; trainer_attempt1_invalid_raw_schema.log`。

## vLLM

1. 一句话：高吞吐 GPU 推理引擎。
2. 本项目：GPU6、TP1、8192、max_num_seqs16。
3. 设计理由：与三卡 trainer 解耦 rollout。
4. 替代方案：Transformers、SGLang、TGI。
5. 可能追问：为什么 eager/0.60？
6. 证据：`outputs/grpo/formal_4gpu_v1/rollout.log`。

## checkpoint/resume

1. 一句话：持久化模型/优化器/RNG/状态。
2. 本项目：每10步，保留10；正式前滚动验证 resume。
3. 设计理由：支持长实验恢复与审计。
4. 替代方案：model-only save、stateless restart。
5. 可能追问：为何仅110–200保留？
6. 证据：`checkpoint-200/trainer_state.json; feasibility.json`。

## TravelPlanner evaluator

1. 一句话：官方规则检查 daily plan 的常识/硬约束。
2. 本项目：官方 commit e52c87f；adapter-only frozen-output 评测。
3. 设计理由：外部可比的确定性基准。
4. 替代方案：内部 Judge、自定义 success。
5. 可能追问：为何 CS micro>0 但 Final=0？
6. 证据：`outputs/final/travelplanner_corrected_metrics.json`。
