# 环境检查报告

**算子**: truncate_mod   **状态**: ✅ 通过   **时间**: 2026-07-31T17:48:35+08:00

## 硬件

| 项 | 值 |
|----|----|
| 芯片型号 | Ascend 910B3 |
| SocVersion | ASCEND910B |
| 设备数 | 1 |

> NPU Arch / `--npu-arch` 编译参数由 Architect 在 Step 2 通过 `/npu-arch` skill 查得后写入 DESIGN.md，本章节不填。

## CANN

- ASCEND_HOME_PATH: `/home/developer/Ascend/cann-8.5.2`
- 版本: `8.5.2`
- CPU 架构目录: `aarch64-linux`

## 编译器与库

> 标 ✅ 的前提是 `test -x <path>` 通过（对可执行文件）或 `test -f <path>` 通过（对头文件/.so）；存在但不可执行 → ❌。

- ✅ bisheng: `/home/developer/Ascend/cann-8.5.2/bin/ccec`
- ✅ kernel_operator.h: `/home/developer/Ascend/cann-8.5.2/aarch64-linux/include/ascendc/basic_api/kernel_operator.h`
- ✅ libregister.so: `/home/developer/Ascend/cann-8.5.2/aarch64-linux/devlib/linux/aarch64/libregister.so`
- ✅ libascendcl.so: `/home/developer/Ascend/cann-8.5.2/aarch64-linux/devlib/linux/aarch64/libascendcl.so`

## asc-devkit

- ✅ 路径: `/mnt/workspace/code/asc-devkit`
- API 文档: 2801 个
- 示例: 324 个
- CMake 配置: ✅

## 检查汇总

- 错误: 0
- 警告: 3

### 警告明细（如 0 则省略本节）
- ⚠ 未安装自定义算子包（仅运行自定义算子时需要）
- ⚠ cannsim 不可用（仅 ascend950 需要）
- ⚠ 日志打屏未开启（建议 `export ASCEND_SLOG_PRINT_TO_STDOUT=1`）
