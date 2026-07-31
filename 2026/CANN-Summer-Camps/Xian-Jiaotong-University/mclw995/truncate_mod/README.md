# TruncateMod

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A2训练系列产品/Atlas 800I A2推理产品 | √ |

## 功能说明

- 算子功能：计算输入张量 `x1` 对 `x2` 的截断取余结果，支持符合广播规则的逐元素计算。
- 计算公式：

$$
y = x1 - \operatorname{truncate}(x1 / x2) \times x2
$$

其中：

- `x1`：被除数输入张量。
- `x2`：除数输入张量。
- `truncate`：表示向 0 方向截断取整。
- `y`：截断取余输出张量。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| x1 | 输入 | TruncateMod计算的第一个输入张量，公式中的 `x1`。 | float16、float32、bfloat16、int32、int8、uint8 | ND |
| x2 | 输入 | TruncateMod计算的第二个输入张量，公式中的 `x2`。 | float16、float32、bfloat16、int32、int8、uint8 | ND |
| y | 输出 | TruncateMod计算结果。 | float16、float32、bfloat16、int32、int8、uint8 | ND |

## 约束说明

- `x1`、`x2` 和 `y` 的数据类型需保持一致，实际支持类型以算子注册中的 `float16`、`float32`、`bfloat16`、`int32`、`int8`、`uint8` 为准。
- `x1` 与 `x2` 支持按标准广播规则推导输出shape；当前Kernel中shape信息按最多8维保存和寻址。
- 输入和输出format均为ND，算子注册支持动态rank和动态shape。
- 输入数据应满足对应数据类型下除法操作的合法性要求；实现未对 `x2` 为0的场景提供额外语义保证。
- 整型和低精度类型在Kernel中存在中间类型转换路径，其中 `int32` 的除法和截断商计算会先转换为 `float32`，超出 `float32` 精确整数表示范围的输入可能受中间精度影响。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| aclnn调用 | [test_aclnn_truncate_mod](examples/test_aclnn_truncate_mod.cpp) | 通过 `aclnnTruncateModGetWorkspaceSize` 和 `aclnnTruncateMod` 接口调用 TruncateMod 算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| --- | --- | --- | --- | --- |
| 12组 | CANN训练营 | TruncateMod | 2026.7.15 | 新增 TruncateMod Ascend C算子实现、Host侧注册/shape推导/tiling、Kernel实现和ACLNN调用样例。 |
