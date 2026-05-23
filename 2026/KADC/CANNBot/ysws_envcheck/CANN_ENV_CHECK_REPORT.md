# CANN 环境检查报告

## 使用的 Skill

**Skill 名称**：`ascendc-env-check`

**Skill 描述**：Ascend C 算子开发环境检查技能。用于：(1) 通过 npu-smi 查询 NPU 设备信息（设备列表、状态、资源使用），(2) 检查 CANN 环境配置（CANN Toolkit、Ops、自定义算子包），(3) 验证开发依赖是否完整。

**Skill 路径**：`.opencode/skills/ascendc-env-check/SKILL.md`

**Skill 工作流程**：

```
环境检查
    │
    ├─ NPU 设备检查
    │   └─ npu-smi list / scripts/npu_info.sh
    │
    └─ CANN 环境检查
        └─ scripts/check_env.sh
```

**使用的诊断脚本**：

| 脚本 | 用途 |
|------|------|
| `scripts/npu_info.sh` | NPU 设备信息综合查询 |
| `scripts/check_env.sh` | CANN 环境配置检查 |

---

## 检查结果

### 一、NPU 设备检查

执行命令：`bash scripts/npu_info.sh`

| 检查项 | 结果 | 详情 |
|--------|------|------|
| 设备检测工具 | ✅ 通过 | npu-smi 可用（版本: 25.5.0） |
| 芯片型号 | ✅ 通过 | 910B3 |
| 设备数量 | ✅ 通过 | 检测到 1 个设备 |
| 可用设备列表 | ❌ 不可用 | Device 7（910B3）不可用 |

**⚠️ 关键问题**：检测到的 NPU 设备（Device 7, 910B3）当前状态为不可用，无法用于算子开发。需排查设备占用或驱动问题。

### 二、CANN 环境检查

执行命令：`bash scripts/check_env.sh`

| # | 检查项 | 结果 | 详情 |
|---|--------|------|------|
| 1 | CANN Toolkit 环境 | ✅ 通过 | `ASCEND_HOME_PATH = /home/developer/Ascend/cann-8.5.2`，set_env.sh 存在 |
| 2 | CANN 版本 | ✅ 通过 | CANN 版本: 8.5.2，运行时基线: 8.5 |
| 3 | CANN Ops 环境 | ✅ 通过 | `ASCEND_OPP_PATH = /home/developer/Ascend/cann-8.5.2/opp`，已安装（0 个 vendors） |
| 4 | 自定义算子包 | ⚠️ 警告 | 未安装自定义算子包（仅运行自定义算子时需要） |
| 5 | CANN 工具 | ✅/⚠️ | msprof 可用；cannsim 不可用（仅 ascend950 需要） |
| 6 | 日志目录 | ⚠️ 警告 | 日志目录不存在（将在首次运行后创建） |
| 7 | 调试配置 | ⚠️ 警告 | 日志打屏未开启，建议 `export ASCEND_SLOG_PRINT_TO_STDOUT=1` |

**总体结论**：✅ CANN 环境检查通过

---

## 综合评估

| 类别 | 状态 | 说明 |
|------|------|------|
| CANN 软件环境 | ✅ 通过 | CANN 8.5.2 已正确安装，关键环境变量配置正常 |
| NPU 硬件状态 | ❌ 不可用 | Device 7 (910B3) 当前不可用 |

### 建议操作

1. **NPU 设备不可用**：运行 `npu-smi info` 查看设备详细状态，排查是否被占用或驱动异常
2. **日志打屏**：建议执行 `export ASCEND_SLOG_PRINT_TO_STDOUT=1` 方便调试
3. **自定义算子包**：如需运行自定义算子，需安装 op_api 库
4. **版本配套**：CANN 8.5.2 与 npu-smi 25.5.0 的配套关系请查阅 CANN 官方 Release Notes

---

## 附录：Skill 检查项说明

| 检查项 | 说明 | 必需性 |
|--------|------|--------|
| ASCEND_HOME_PATH | CANN Toolkit 路径 | 必需 |
| CANN 版本 | 检测版本号及运行时依赖基线 | 建议确认 |
| ASCEND_OPP_PATH | CANN Ops 路径 | 运行时必需 |
| 自定义算子包 | op_api 库 | 运行自定义算子必需 |
| CANN 工具 | msprof/cannsim | 可选 |