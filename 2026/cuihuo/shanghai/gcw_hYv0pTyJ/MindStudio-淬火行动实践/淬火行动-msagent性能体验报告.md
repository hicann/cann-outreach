# MSAgent 体验报告

## 一、报告基本信息

| 项目 | 内容 |
|------|------|
| 报告编号 | 淬火行动-20260819-001 |
| 体验日期 | 2026-08-19 |
| Agent版本 | msAgent v26.1.0a2（mindstudio-agent==26.1.0a2） |
| LLM配置 | provider=openai（兼容DeepSeek），base_url=https://api.deepseek.com，model=deepseek-v4-flash |
| Profiling数据来源 | Qwen3-32B.zip：4卡（rank 0-3）单机 ascend_pt Profiling 数据 + cluster_analysis.db 集群分析库 |
| 数据路径 | /root/user_data/cluster_analysis_output/、/root/user_data/df536040f370_651/654/657/664_*_ascend_pt |
| 测试目标 | 验证 msagent 在 Profiling 下发/调度类问题（集群快慢卡）分析中的能力 |

---

## 二、交互记录

### 第 1 轮交互（UC003 能力自述）

| 项目 | 内容 |
|------|------|
| 输入 Prompt | 你好，我是淬火行动的测试者。请简单介绍一下你自己，以及你当前有哪些能力、能帮我做什么？ |
| Agent 输出（文字摘要） | Agent 自述为 Profiler（Ascend NPU 性能分析助手），工作原则为"数据驱动、证据闭环、结论简洁"；能力覆盖：单卡性能分析（Timeline→算子热点→通信→采集配置）、多卡/集群分析（快慢卡识别、负载不均衡、慢节点定位）、通信分析（HCCL 耗时/带宽/Notify Wait/小包重传）、计算分析（算子热点/动态shape/低效Kernel）、调度分析（Host 下发瓶颈/launch 延迟/同步阻塞）、算子 MFU 计算、数据完整性校验；工具能力：msprof-mcp 工具集（trace_view.json、kernel_details.csv、op_statistic.csv、communication_matrix.json、analysis.db）、SQL 查询 profiler DB、专业技能库按需加载。 |
| **输出截图** | 见附件 UC003-能力自述.txt（文本记录） |
| 是否符合预期 | 是 |
| 评价 | 回答结构清晰、能力覆盖与任务手册描述一致，能正确引导用户提供数据路径进入分析流程 |

### 第 2 轮交互（UC004-1 快慢卡判断）

| 项目 | 内容 |
|------|------|
| 输入 Prompt | 从当前Profiling数据来看，有无集群快慢卡，有什么关键证据？数据在 /root/user_data/cluster_analysis_output/cluster_analysis.db（集群分析库），以及 /root/user_data/ 下 df536040f370_651/654/657/664 四个 *_ascend_pt 目录（4卡单机 Profiling 数据）。请基于真实数据给出结论。 |
| Agent 输出（文字摘要） | 结论：存在快慢卡现象。慢卡 = Rank 1（主因）+ Rank 2（次因），瓶颈类型为「Host 下发瓶颈/调度瓶颈」（伪快卡）；Rank 0/3 是被拖累的正常卡。关键证据：1) cluster_time_summary：Rank1/2 Free 占比 61.5%/58.2%（约 869.5ms/823.6ms），Rank0/3 仅 44%（625-628ms），且差值恰等于 Rank0/3 多出的通信时间——典型伪快卡特征；2) slow_rank：Rank1 影响集群 137 次、Rank2 129 次（rank0/3 未上榜）；3) cann_api_sum：Rank1 Host 侧 Attention 下发 API 耗时 130.1ms（1.55x）、aclrtRecordEvent 62.7ms（1.58x），而 aclrtSynchronizeEvent 仅 19.6ms（不阻塞，说明 CPU 下发进度永远落后）；4) free_analysis：Rank1 最大 CANN 下发间隙 19.92ms；5) hccl_sum：allReduce 平均 Rank0/3=1.65ms（几乎全为 idle 等待）、Rank1/2=0.78ms；6) 排除项：HCCS 带宽 16.5GB/s 无慢链路（offsetRatio~3%）、各卡计算算子 count/耗时一致（排除计算型慢卡）；7) advisor：Rank1 Host 瓶颈占比 0.9、Rank2 0.881。优化建议（P0-P2）：CPU 绑核/NUMA 排查、小包通信合并（allReduce 单包仅 164KB）、字节对齐（17 个算子）、关闭动态 shape、算子融合。 |
| **输出截图** | 见附件 UC004-第1轮-快慢卡判断.txt（文本记录） |
| 是否符合预期 | 是 |
| 评价 | 结论与 msprof-analyze 官方 slow_rank 判定完全一致；证据链完整（宏观→微观→排除项→建议），量化到位，符合“数据驱动、证据闭环”的定位 |

### 第 3 轮交互（UC004-2 快慢卡根因）

| 项目 | 内容 |
|------|------|
| 输入 Prompt | 造成快慢卡的原因是什么？请结合通信耗时、带宽、等待/同步时间等维度分析根因。 |
| Agent 输出（文字摘要） | 根因定性：通信型慢卡（等待/同步链主导）叠加 Host 下发间隙，带宽与计算均不是原因。核心结论：1) 慢卡（拖慢集群的迟到方）= Rank 1 → Rank 2（分阶段轮换：allReduce 0-130 → allReduce 131-258+allGather 259 → allGather 262-274），slow_rank 影响 137/129 次；2) 同一逻辑 allReduce 启动时刻跨卡差 8.9-12.4ms 但结束时刻仅差 0.05ms——迟到方一到即完成、早到方全部空等（最长 12.47ms）；3) 通信耗时差异 99% 为等待/同步时间：Rank0 通信 447.9ms 中 idle 442.6ms，真实传输仅 24-30µs；4) 根因链：Rank1/2 Host 下发存在异常空洞（free_analysis：最大 node@launch 间隙 19.7ms）→ NPU 早段被饿死（Timeline：Rank1 Free 358ms vs Rank0 131ms）→ 258 个 allReduce 每次迟到 4-12ms → Rank0/3 在算子内空等并产生 286-330ms aclrtSynchronizeEvent 阻塞；5) 排除项：计算型慢卡（count 一致、耗时差异 ≤3%）、负载不均衡、带宽/链路故障（HCCS 16.6GB/s 正常、无 RDMA）；6) 待验证项：早段（0-600ms）是否为 warmup；建议 P0 排查 Rank1/2 下发线程绑核/NUMA/锁竞争，P1 合并梯度桶小包、核对 HCCL 配置。已生成 Artifact：timing_evidence.csv。 |
| **输出截图** | 见附件 UC004-第2轮-根因分析.txt（文本记录） |
| 是否符合预期 | 是 |
| 评价 | 根因链条完整（现象→证据→因果链→排除项→待验证项→建议），能区分“表象通信耗时高”与“真正迟到方”，分析深度超出预期；还自动生成了证据文件 |

### 第 4 轮交互（UC004-3 影响量化）

| 项目 | 内容 |
|------|------|
| 输入 Prompt | 评估快慢卡问题造成的影响：拖慢了整体训练/推理多少时间？请量化到具体毫秒并说明计算依据。 |
| Agent 输出（文字摘要） | 结论：快慢卡问题每个 step 拖慢 441.9 ms，占 step 总耗时 1414.0 ms 的 31.3%；消除后理论 step 约 972.1 ms（吞吐可提升约 1.45x）。量化依据：1) Σ 275 个集合通信算子的跨卡偏差 ≈ 441.9ms（平均每个集合通信被拉长 1.61ms）；2) 交叉验证：Rank0 通信 447.9ms 中 442.6ms 为 idle 空等（98.8%），≈ 441.9ms skew + 6.8ms 真实搬运（SlowLinkSum allReduce 均值 24.4µs×258 + allGather 30.6µs×17）；3) 通信与计算几乎零重叠（overlapped 仅 79.9µs），空等全部落在关键路径；4) 275 个算子中 272 个最后到达者为 Rank1（138 次）或 Rank2（134 次），与 SlowRank 表 137/129 次一致；5) skew 前后半段均摊（224.6ms/217.3ms），为贯穿性系统现象；6) 最严重单次 12.45ms（hcom_allReduce__503_1_1 被拉伸到 12.47ms，min 仅 0.02ms）。外推公式：总损失 ≈ 441.9ms × step 总数。P0 建议：Rank1/2 Host 下发瓶颈（CPU 绑核/NUMA）、收敛 attention/recordEvent Host 侧 1.5x 差异；P1：合并 80KB 小包 allReduce；复测目标 skew<50ms。给出了可复现的验证 SQL。 |
| **输出截图** | 见附件 UC004-第3轮-影响量化.txt（文本记录） |
| 是否符合预期 | 是（数字经复核：按 start_timestamp 跨卡跨度 Σ=443.4ms，step 级通信差额 441.3ms，与 441.9ms 吻合；迟到方分布 rank1=137/rank2=131 与官方 SlowRank 一致） |
| 评价 | 量化结果可信且给了可复现验证方法；小瑕疵：文中“Σ(max−min elapsed_time)”的 SQL 描述与实际口径（start_timestamp 跨度）不完全一致，建议优化表述精确性 |

*（根据实际对话轮次自行增删）*

---

## 二·附、必测用例记录表（手册 3.1）

| 用例ID | 测试场景 | 详细实操步骤 | 输入参数/配置 | 预期结果 | 实际结果 | 测试结论 |
|--------|---------|-------------|--------------|---------|---------|---------|
| UC001 | 环境搭建验证 | msagent --version | / | msAgent v26.1.0a2 | 返回 msAgent v26.1.0a2 | 通过 |
| UC002 | 配置模型与API | export OPENAI_API_KEY；msagent config --llm-provider openai --llm-base-url https://api.deepseek.com --llm-model deepseek-v4-flash；msagent config --show | openai/deepseek-v4-flash | config --show 显示配置信息 | Provider=openai、Model=deepseek-v4-flash、Base URL=api.deepseek.com、MCP=msprof-mcp 已启用 | 通过 |
| UC003 | 启动msagent会话 | msagent "你好…介绍能力"（one-shot 模式） | / | 正常对话 | 完整自我介绍，覆盖单卡/多卡分析、通信/计算/调度、MFU、数据校验、msprof-mcp 工具链 | 通过 |
| UC004 | msagent辅助性能分析（下发调度类） | 基于 Qwen3-32B 4卡 Profiling 数据三轮追问：①有无快慢卡及证据 ②根因 ③影响量化；每轮记录输出并判断 | 数据：/root/user_data/cluster_analysis_output + 4× ascend_pt；-a Profiler -am aggressive | 得到符合预期的分析结论 | ①慢卡=Rank1/2（Host下发瓶颈型伪快卡），与 msprof-analyze 官方 slow_rank 一致 ②根因链完整（19.7ms 下发空洞→NPU 饿死→集合通信迟到 4-12ms→Rank0/3 空等）③每 step 拖慢 441.9ms（31.3%），复核 441.3-443.4ms，迟到方分布与官方一致 | 通过（附1处表述精度小瑕疵） |

---

## 三、Insight 验证（手册 3.2，CLI 环境等效验证）

> 说明：本环境为 CLI 容器，无 MindStudio Insight GUI；以下使用与 Insight 同源的官方 msprof-analyze 产物与 trace_view.json 时间线数据完成等效验证。

**概览页比对（计算/空闲/通信时间，cluster_time_summary，µs）**

| Rank | 计算 | 通信 | 空闲(Free) | Stage | Free占比 |
|------|------|------|-----------|-------|---------|
| 0 | 333,768 | 447,903 | 625,006 | 1,414,033 | 44.2% |
| 1 | 332,579 | 203,812 | 869,529 | 1,413,914 | 61.5% |
| 2 | 335,418 | 247,768 | 823,581 | 1,414,580 | 58.2% |
| 3 | 334,224 | 444,982 | 628,357 | 1,415,349 | 44.4% |

→ 计算四卡均衡（极差<1%）；Rank1/2 Free 占比显著偏高，与 msagent 结论一致。

**时间线页 Overlap_analysis 最大 Free 切片（trace_view.json 提取，µs）**

| Rank | 最大 Free 切片 | 总 Free | 总 Computing |
|------|--------------|---------|-------------|
| 0 | 64,974 | 640,330 | 333,768 |
| 1 | 46,619 | 886,376 | 332,579 |
| 2 | **68,634** | 839,306 | 335,415 |
| 3 | 66,315 | 641,885 | 334,217 |

→ 时间线 Free 切片与概览统计一致，Rank1/2 空闲总量最大；Rank2 存在最大单段 Free（68.6ms），与 free_analysis 的 8.87ms 下发间隙同源。

**官方判定交叉验证**：msprof-analyze -m slow_rank → Rank1 影响 137 次、Rank2 影响 129 次；-m slow_link → 无异常链路（HCCS 16.5GB/s，offsetRatio≈3%）；-m hccl_sum → allReduce 单包 81KB 小包通信。均与 msagent 结论一致。

---

## 三、多轮交互整体评价

| 评价维度 | 评分（1~5） | 说明 |
|----------|-------------|------|
| 问题理解准确性 | 5 | 三轮追问均准确理解意图，能自动定位数据并选择对应技能（快慢卡检测/DB探索/msprof-analyze） |
| 数据分析深度 | 5 | 宏观（cluster_time_summary）→微观（cann_api/free_analysis/hccl_sum）→排除项（计算/带宽/负载）三层证据链，根因链清晰 |
| 证据链条完整性 | 5 | 每条结论附量化证据，关键数字与官方 slow_rank/slow_link/hccl_sum 交叉一致；自带验证SQL |
| 优化建议实用性 | 4 | 建议分 P0-P2 且含验证方法（绑核后复测指标）；部分建议（如 CPU 绑核）需环境信息才能落地 |
| 多轮对话连贯性 | 4 | 三轮独立执行（one-shot 模式），结论口径一致无矛盾；未测试连续会话内多轮（受环境限制） |
| 响应速度与稳定性 | 3 | 每轮约 5-10 分钟（工具调用多）；首次配置时因配置模板 bug 挂起、网络波动时偶发连接错误，稳定性待提升 |
| 整体满意度 | 4 | 分析质量高，达到“可直接沉淀为 Issue/优化建议”的标准；稳定性与配置体验拖分 |

---

## 四、问题与改进建议

### 发现的问题

1. 【高】使用 `-w <工作目录>` 时，msagent 在目标目录自动生成的 config.llms.yml 为未替换模板（version=__APP_VERSION__、model=gpt-4o-mini、缺 base_url），导致非默认工作目录下所有 LLM 调用指向错误模型/端点，表现为长时间挂起或连接错误（已手动修复）。
2. 【高】非交互（无 TTY）模式下默认审批模式（active）遇到工具审批需求时无限挂起，无超时/降级/报错提示；必须显式 -am aggressive 才能自动继续。
3. 【中】LLM 连接错误时输出 “Error processing message: Connection error” 但进程退出码为 0，且不自动重试，脚本难以可靠判断成败。
4. 【低】影响量化轮中，结论数字（441.9ms）正确（复核 441.3-443.4ms），但文中所述 SQL 口径（max−min elapsed_time）与实际依据（start_timestamp 跨度）不一致，建议修正表述并给出可复现 SQL。

### 改进建议

1. 配置初始化时校验模板替换完整性（__APP_VERSION__ 等占位符）并打印配置校验警告；默认模型回退到全局配置。
2. 非交互模式自动切换审批策略或对审批请求设置超时并给出明确报错；文档补充 headless 使用指引（-am aggressive）。
3. 网络类错误区分退出码（如非零）+ 内置有限重试（指数退避），并在 stderr 输出结构化错误。
4. 量化类输出附上可执行的验证 SQL 与数据口径说明，方便用户复现。
