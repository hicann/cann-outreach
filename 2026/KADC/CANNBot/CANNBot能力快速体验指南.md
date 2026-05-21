# CANNBot 能力快速体验指南

CANNBot 是昇腾原生 AI 智能开发助手，内置 40 个技能模块、7 大开发插件，覆盖 AscendC / PyPTO / Triton / PyTorch 全场景算子开发与模型推理优化。**自然语言描述需求，AI 自动完成设计→编码→调试→调优全流程。**

以下所有体验在展台即可完成——无需安装，无需登录，直接对话。

---

## 30 秒感受：随便问一个昇腾问题

CANNBot 不是通用 chatbot。它内部加载了 1022 个 API 文档、587 个示例代码、NPU 硬件参数表、昇腾编码规范等结构化知识，能直接定位到你要的具体信息。试着问一句——你会发现它真的懂。

### 按你的开发阶段，挑一个问题试试

每个昇腾开发者都会经历的旅程：搭环境 → 做设计 → 写代码 → 调 bug → 查精度 → 优性能 → 写测试 → 做审查。CANNBot 在每个环节都能帮上忙。

**🔧 环境搭建——环境对不对？**

> - 怎么检查我的 CANN 环境是否正确安装了？
> - 我的 npu-smi 显示设备 but CANN 找不到设备，怎么排查？

**📐 方案设计——用哪个 API？怎么切 Tiling？**

> - Atlas A2 和 A3 的 NPU 架构有什么区别？
> - DataCopy 和 DataCopyPad 的区别是什么，什么时候用哪个？
> - 帮我设计 Softmax 的 Tiling 方案，float16，shape [32,4096]，在 910B 上跑

**💻 编码实现——怎么写是对的？**

> - 双缓冲加流水线的标准实现哪里有参考代码？我想直接看示例
> - Ascend C 里的 repeatTimes 参数有什么限制？容易踩什么坑？
> - 多芯片架构的条件编译宏怎么写？我的算子要同时支持 910B 和 950PR

**🐛 运行调试——报错什么意思？卡死怎么办？**

> - 我的算子报了 aclnn 错误码 161001，是什么意思，怎么排查？
> - Kernel 跑到一半卡死不返回，从哪里开始排查？
> - 编译时报 "undefined reference to AscendC::..." 是什么原因？

**🎯 精度调试——结果不对，从哪里查？**

> - 我的算子输出全是 0，可能是什么原因？
> - FP16 精度差于预期，和 PyTorch 参考实现对比 rtol 不达标，怎么定位？
> - Cast 后数据出现随机错误，是不是流水线同步有问题？

**⚡ 性能调优——速度不达标，怎么优化？**

> - 我的 MatMul 算子带宽利用率只有 40%，从哪些方向优化？
> - Profiling 数据显示 UB 访问冲突严重，怎么解决？
> - 怎样判断应该用双缓冲还是多缓冲？对性能影响有多大？

**🧪 测试验证——测试用例怎么写？**

> - AscendC 算子的 ST 测试用例应该覆盖哪些测试因子？
> - 帮我为这个算子生成白盒测试用例，覆盖 low/medium/high 三档
> - 我的 UT 覆盖率不够，帮我分析哪些代码路径还没测到

**✅ 代码审查——代码有没有隐患？**

> - 帮我检视这段算子代码，全量覆盖安全规范和功能正确性

**做 PyTorch 模型推理？**

> - torch.compile 在昇腾 NPU 上怎么配置？我应该用 npugraph_ex 还是 GE 模式？
> - reduce-overhead 和 max-autotune 在昇腾上应该选哪个？
> - 我的 ResNet 用 torch.compile 报了 BackendCompilerFailed，怎么排查？
> - 推理时 OOM 了，怎么用 KVCache 优化显存？PagedAttention 怎么接入？
> - 模型从 GPU 迁移到 NPU 推理，输出精度有偏差，怎么定位？

**做 PyPTO 算子开发？**

> - 我想用 PyPTO 实现 GELU 算子，PyPTO 支持哪些操作？
> - 帮我生成这个算子的 Golden 参考实现，用于精度对比
> - PyPTO 算子的性能泳道图怎么看？红色段代表什么？

**做 Triton 算子开发？**

> - 帮我设计 Flash Attention 的 Triton-Ascend 算法草图，fp16，[bs=32, heads=32, seq=4096, dim=128]，910B
> - MatMul [M=1024, K=512, N=2048] fp16，在 910B 上怎么设计 Triton kernel？
> - 我有一个 GPU 上的 Triton kernel，怎么迁移到 Ascend NPU？

---

## 2 分钟体验：让它帮你干活

除了回答问题，CANNBot 还能在 2 分钟内完成实际的开发任务。挑一个试试。

### 代码检视

贴一段你的 AscendC 代码，CANNBot 会自动完成全量检视——并行调度多个检视 Agent，覆盖安全规范和功能正确性，输出逐条问题列表（含行号、代码片段、规范条款、修复建议）。

> 试试：找工作人员要一段示例代码，输入"请对这段 VecAdd 算子代码进行全量检视，覆盖安全规范和功能正确性。"

### Tiling 策略设计

描述你的算子需求，CANNBot 在 2 分钟内给出完整 Tiling 方案——多核切分策略、UB Buffer 规划、双缓冲建议、分支覆盖分析。

> 试试："帮我设计 Softmax 的 Tiling 方案，float16，shape [32,4096]，在 910B 上跑。"

### Tiling 自动求解

CANNBot 可以自动调用专业的 TilingSolver CLI 工具，为你的 MatMul 或 Vector 算子求解最优 Tiling 参数。

> 试试："帮我求解 MatMul [M=4096, K=4096, N=4096] fp16 在 910B 上的最优 Tiling 参数。"

### Torch Compile 配置

不确定 torch.compile 在昇腾上怎么配？直接把你的场景告诉 CANNBot。

> 试试："帮我生成一段 npugraph_ex 的标准推理代码模板，我要在 910B 上跑推理。"

---

## 30 分钟完整走通：从自然语言到可运行算子

以上都是秒级/分钟级的能力展示。如果你想完整体验 CANNBot 从零生成一个 NPU 算子的全过程——包括编译、运行、性能验证——请参考：

→ **[CANNBot 新用户快速上手指南](./CANNBot新用户快速上手指南.md)**

从注册 GitCode 到在 CANNLab 云端 NPU 环境跑通第一个算子，全程约 30-40 分钟（其中大部分时间在等编译）。免费算力，浏览器即用，无需本地安装。

---

## CANNBot 能做什么：能力全景

| 技术栈 | 支持的开发场景 | 一句话描述 |
|--------|--------------|-----------|
| **Ascend C 直调** | Kernel 原型开发与验证 | 自然语言→Kernel 代码→编译→运行→性能报告 |
| **Ascend C 注册调用** | 完整自定义算子工程 | 从需求到 aclnn/GEIR 接口的标准化算子开发 |
| **PyPTO** | Python 原生算子开发 | PyTorch 风格的昇腾算子端到端开发 |
| **Triton-Ascend** | Triton DSL 算子开发 | 从算法草图到性能优化的 6 阶段流水线 |
| **Torch Compile** | 图模式编译优化 | npugraph_ex 配置、诊断、自定义算子入图 |
| **模型推理优化** | LLM 推理全链路优化 | 并行策略/KVCache/融合算子/图模式/多流/SuperKernel |
| **代码检视** | AscendC 算子代码审查 | 主 Agent 调度多子 Agent 并行检视，100% 条例覆盖 |

---

## 更多资源

- **CANNBot 仓库**：[https://gitcode.com/cann/cannbot-skills](https://gitcode.com/cann/cannbot-skills)
- **CANN 社区**：[https://gitcode.com/cann](https://gitcode.com/cann)
