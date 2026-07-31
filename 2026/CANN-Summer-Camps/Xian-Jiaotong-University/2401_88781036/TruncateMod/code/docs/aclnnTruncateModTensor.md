# aclnnTruncateModTensor

[查看源码](https://gitcode.com/cann/ops-math/tree/master/experimental/math/truncate_mod)

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| Atlas A2 训练系列产品/Atlas A3 系列产品 | √ |

## 功能说明

对 `self` 和 `other` 逐元素执行向零截断取模：

$$
out = self - \operatorname{trunc}(self / other) \times other
$$

`trunc` 表示向零截断，因此结果符号与 `self` 一致。两个输入支持广播。

## 函数原型

Tensor-Out 接口：

```cpp
aclnnStatus aclnnTruncateModTensorGetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* other,
    aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnTruncateModTensor(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

原地接口：

```cpp
aclnnStatus aclnnInplaceTruncateModTensorGetWorkspaceSize(
    const aclTensor* selfRef,
    const aclTensor* other,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnInplaceTruncateModTensor(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

每个算子均为两段式接口：先获取 workspace 大小和执行器，再执行计算。

## 参数说明

### aclnnTruncateModTensorGetWorkspaceSize

- `self`：Device 侧输入 Tensor。
  - 数据类型支持 `BF16`、`FLOAT16`、`FLOAT`、`INT32`、`INT8`、`UINT8`；
  - 支持非连续 Tensor；
  - 数据格式支持 `ND`；
  - 维数不超过 8。
- `other`：Device 侧输入 Tensor。类型约束与 `self` 相同，shape 需与 `self` 满足广播关系。
- `out`：Device 侧输出 Tensor。
  - shape 必须等于 `self` 与 `other` 广播后的 shape；
  - 数据类型必须能由两个输入的推导类型转换；
  - 支持非连续 Tensor及 `ND` 格式。
- `workspaceSize`：返回执行所需的 Device workspace 字节数。
- `executor`：返回包含计算流程的执行器。

### aclnnTruncateModTensor

- `workspace`：Device workspace 地址；当 `workspaceSize` 为 0 时可传入空指针。
- `workspaceSize`：第一段接口返回的 workspace 大小。
- `executor`：第一段接口返回的执行器。
- `stream`：执行任务的 ACL stream。

### 原地接口

`selfRef` 同时作为输入和输出。广播结果 shape 必须与 `selfRef` 相同，其余约束与 Tensor-Out 接口一致。

## 返回值

- `ACLNN_SUCCESS`：执行成功；
- `ACLNN_ERR_PARAM_NULLPTR`：必要参数为空指针；
- `ACLNN_ERR_PARAM_INVALID`：数据类型、格式、维数、广播关系或输出 shape 不合法；
- 其他返回值参见 aclnn 返回码说明。

## 约束说明

- 最大支持 8 维输入；
- 不支持私有格式；
- 整数除数为 0 时，kernel 输出 0；
- 浮点输入应避免除数为 0；
- 默认采用确定性实现。

## 调用示例

下面展示两段式接口的核心调用流程。Tensor 创建、数据搬运及资源释放按 ACL 标准接口处理。

```cpp
#include "acl/acl.h"
#include "aclnn_truncate_mod_tensor.h"

int RunTruncateMod(
    const aclTensor* self,
    const aclTensor* other,
    aclTensor* out,
    aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnTruncateModTensorGetWorkspaceSize(
        self, other, out, &workspaceSize, &executor);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
    }

    ret = aclnnTruncateModTensor(workspace, workspaceSize, executor, stream);
    if (ret == ACLNN_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return ret;
}
```
