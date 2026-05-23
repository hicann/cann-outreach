# CANN 环境检查报告

> 生成时间：2026-05-22

## 使用的 Skill

**Skill 名称**：`ascendc-env-check`（Ascend C 算子开发环境检查技能）

**Skill 位置**：`.opencode/skills/ascendc-env-check/SKILL.md`

**调用的脚本**：
- `scripts/npu_info.sh` — NPU 设备信息综合查询
- `scripts/check_env.sh` — CANN 环境配置检查

---

## 1. NPU 设备信息

| 项目 | 值 |
|------|-----|
| npu-smi 版本 | 25.5.1 |
| 设备数量 | 2 |
| 芯片型号 | Ascend910 |
| HBM 容量 | 65536 MB (每卡) |

| 设备 | Phy-ID | Health | 温度(C) | HBM 使用 | 状态 |
|------|--------|--------|---------|----------|------|
| NPU 5 (Chip 0) | 10 | OK | 45 | 2908 / 65536 MB | **不可用** |
| NPU 5 (Chip 1) | 11 | OK | 46 | 2870 / 65536 MB | **不可用** |

> **警告**：2 个 NPU 设备均标记为**不可用**，无法执行算子上板运行和性能采集。

---

## 2. CANN 环境配置

| 检查项 | 结果 | 详情 |
|--------|------|------|
| ASCEND_HOME_PATH | ✓ 通过 | `/home/developer/Ascend/cann-8.5.2` |
| CANN 版本 | ✓ 通过 | 8.5.2（运行时基线 8.5） |
| ASCEND_OPP_PATH | ✓ 通过 | `/home/developer/Ascend/cann-8.5.2/opp` |
| 自定义算子包 | ⚠ 未安装 | 仅运行自定义算子时需要 |
| msprof 工具 | ✓ 可用 | — |
| cannsim 工具 | ⚠ 不可用 | 仅 ascend950 需要，可忽略 |
| 日志目录 | ⚠ 不存在 | 首次运行后自动创建 |
| 日志打屏 | ⚠ 未开启 | 建议 `export ASCEND_SLOG_PRINT_TO_STDOUT=1` |

---

## 3. 总结

- **CANN 环境配置**：通过，核心依赖（Toolkit 8.5.2、Ops 路径、msprof）均正常
- **NPU 设备**：2 张 Ascend910 被检测到但**均不可用**，上板运行和性能验收将受限

## 4. 建议操作

1. 排查 NPU 不可用原因（驱动/固件/权限）
2. 开启日志打屏：`export ASCEND_SLOG_PRINT_TO_STDOUT=1`
3. 安装自定义算子包（如需运行自定义算子）
