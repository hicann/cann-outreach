# SoftshrinkGrad 自测报告

**CANN SUMMER CAMPS 2026 · XJTU · 第16组**

## 1. 交付件检查

| 项目 | 状态 |
|---|---|
| 算子设计文档 | 已完成 |
| 项目设计题目 | 已完成 |
| Host/Tiling代码 | 已完成 |
| Ascend C Kernel | 已完成 |
| FLOAT/FLOAT16/BFLOAT16设计 | 已完成 |
| 动态shape与尾块设计 | 已完成 |
| CPU参考测试 | 已完成 |
| ACLNN端到端样例 | 已完成 |
| 编译安装运行脚本 | 已完成 |

## 2. CPU 参考测试

执行：

```bash
python3 test/test_reference.py
```

覆盖：

1. 同 shape 基本计算。
2. `self == ±lambd` 边界。
3. 两输入 broadcast。
4. 标量。
5. 空 Tensor。
6. NaN。
7. 负 `lambd` 参数校验。
8. 不兼容 shape 校验。

实测结果：**8/8 全部通过**。

## 3. NPU 端到端测试

实测环境：

| 项目 | 配置 |
|---|---|
| 测试日期 | 2026-07-30 |
| CPU架构 | aarch64 |
| NPU | Ascend 910B3 |
| NPU驱动/npu-smi | 25.5.0 |
| CANN Toolkit | 9.0.0 |
| CMake | 3.22.1 |
| GCC/G++ | 11.4.0 |

执行：

```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
bash run.sh
```

判定标准：

- 自定义算子编译和安装成功。
- `aclnnSoftshrinkGradGetWorkspaceSize` 返回成功。
- Kernel 执行成功。
- FLOAT 输出与 CPU golden 逐元素一致。
- 终端输出 `SoftshrinkGrad ACLNN test: PASS`。

实测结果：

| 测试阶段 | 结果 |
|---|---|
| Host/Tiling编译 | 通过 |
| Ascend C Kernel编译 | 通过 |
| 910B算子二进制生成 | 通过 |
| 自定义算子打包与安装 | 通过 |
| ACLNN样例编译 | 通过 |
| NPU Kernel执行 | 通过 |
| 输出与CPU golden比对 | 通过 |

最终输出：

```text
SoftshrinkGrad ACLNN test: PASS
```

测试容器已经将物理NPU映射为逻辑设备0，因此测试程序使用
`aclrtSetDevice(0)`；未额外设置 `ASCEND_RT_VISIBLE_DEVICES`。

## 4. 精度标准

SoftshrinkGrad 只在输入梯度与零之间选择，不包含累加或近似函数：

- FLOAT：绝对误差 0。
- FLOAT16/BFLOAT16：输出应等于输入梯度的原始可表示值或零。
- 边界 `self == ±lambd`：严格输出零。

## 5. 当前验证结论

仓库提交前已完成本地Python参考测试、静态检查以及Ascend 910B3设备端编译和
ACLNN端到端运行，功能结果正确。CANNJudge尚未提交，因此不填写CANNJudge提交号
或性能排名。

| 验证项 | 结果 |
|---|---|
| Python CPU参考测试 | 8/8 通过 |
| Git差异格式检查 | 通过 |
| CANN Toolkit设备端编译 | CANN 9.0.0 + Ascend 910B3通过 |
| NPU端到端运行 | `SoftshrinkGrad ACLNN test: PASS` |
| CANNJudge | 尚未提交，不填写虚假提交号或性能结果 |
