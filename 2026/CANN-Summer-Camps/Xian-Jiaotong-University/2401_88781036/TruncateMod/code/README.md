# TruncateMod

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| Atlas A2 训练系列产品/Atlas A3 系列产品 | √ |

## 功能说明

TruncateMod 对两个输入逐元素执行向零截断取模：

$$
y = x1 - \operatorname{trunc}(x1 / x2) \times x2
$$

其中 `trunc` 表示向零截断。输入支持 NumPy 风格广播，最大维数为 8。

示例：

```text
truncate_mod( 7,  3) =  1
truncate_mod(-7,  3) = -1
truncate_mod( 7, -3) =  1
truncate_mod(-7, -3) = -1
```

## 数据类型

输入和输出支持 `bfloat16`、`float16`、`float32`、`int32`、`int8` 和 `uint8`。两个输入经过类型推导后由同一 TruncateMod kernel 计算，输出类型需能由推导类型安全转换。

## 目录结构

```text
truncate_mod/
├── docs/       aclnn 接口说明
├── examples/   aclnn 调用示例
├── op_api/     aclnn 与 Level-0 实现
├── op_host/    算子原型、InferShape 和 Tiling
├── op_kernel/  AscendC kernel
└── tests/      API、Host 和 Kernel 单元测试
```

## 接口

- `aclnnTruncateModTensorGetWorkspaceSize`
- `aclnnTruncateModTensor`
- `aclnnInplaceTruncateModTensorGetWorkspaceSize`
- `aclnnInplaceTruncateModTensor`

详细参数、约束和调用方法参见 [aclnnTruncateModTensor](docs/aclnnTruncateModTensor.md)。

## 构建与测试

在 `ops-math` 仓库根目录按工程统一方式构建和执行 UT。提交前应至少完成：

1. `op_api`、`op_host`、`op_kernel` 单元测试；
2. Ascend 910B 功能测试；
3. 50 个性能测试点与 Baseline 对比；
4. README、设计文档及自验证报告检查。
