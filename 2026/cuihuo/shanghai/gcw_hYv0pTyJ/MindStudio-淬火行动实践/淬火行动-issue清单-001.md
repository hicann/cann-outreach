# 淬火行动 Issue 清单 - 001

## 基本信息

- 开发者ID：（待填写）
- 体验日期：2026-08-19
- Agent版本：msAgent v26.1.0a2（mindstudio-agent==26.1.0a2）
- 关联仓库：https://gitcode.com/Ascend/msagent
- 体验场景：4卡（DP=4）Qwen3-32B 集群 Profiling 快慢卡分析（UC001-UC004）

## Issue 清单

| 序号 | Issue 标题 | Issue 链接 | 问题类型 | 严重程度 | 状态 |
|------|-----------|-----------|---------|---------|------|
| 1 | [配置] 使用 -w 工作目录时自动生成的 config.llms.yml 为未替换模板（version=__APP_VERSION__、model=gpt-4o-mini、缺少 base_url），导致非默认目录下 LLM 调用指向错误模型/端点并长时间挂起 | 待提交 | 配置/缺陷 | 高 | 待提交 |
| 2 | [CLI] 非交互（无TTY）模式下默认审批模式(active)遇到工具审批请求时无限挂起，无超时/降级/报错 | 待提交 | CLI/交互 | 高 | 待提交 |
| 3 | [CLI] LLM 连接错误时输出 "Error processing message: Connection error" 但进程退出码为 0 且不自动重试，脚本无法可靠判断成败 | 待提交 | CLI/错误处理 | 中 | 待提交 |
| 4 | [文档/精度] 影响量化回答中量化 SQL 口径（max−min elapsed_time）与实际依据（start_timestamp 跨度）不一致，建议输出可复现 SQL 与口径说明 | 待提交 | 文档/精度 | 低 | 待提交 |

## 说明

- Issue 提交地址：https://gitcode.com/Ascend/msagent/issues
- 标签规范：淬火行动 / document / 对应模块 / 版本标签（26.1.0）
- 每个 Issue 需写清：问题发生上下文、预期结果、实际结果、相关截图/日志
- 详细复现上下文见《淬火行动-msagent性能体验报告.md》第四节

## 问题 1 复现要点（示例）

```
环境：Ubuntu 22.04 (aarch64)，mindstudio-agent==26.1.0a2，OPENAI_API_KEY 已配置（DeepSeek）
复现：msagent -a Profiler -w /root/user_data --no-stream "请分析数据"
现象：进程挂起（阻塞在 LLM 调用重试）；检查 /root/user_data/.msagent/config.llms.yml 发现：
  version: __APP_VERSION__   # 模板占位符未替换
  model: gpt-4o-mini          # 非全局配置的 deepseek-v4-flash
  缺 base_url                  # 指向默认 openai 端点
预期：工作目录配置应继承全局配置（模型/端点/版本），或提示初始化失败
```
