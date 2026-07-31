# aclnnTruncateMod

## 支持的产品型号

| 产品型号 | 是否支持 |
| --- | --- |
| Atlas A2训练系列产品/Atlas 800I A2推理产品 | √ |

## 功能说明

`aclnnTruncateMod`用于计算输入张量`x1`对输入张量`x2`的截断取余结果。该接口支持按照广播规则对两个输入张量进行逐元素计算，输出张量`out`的shape为`x1`与`x2`广播后的shape。

计算公式如下：

$$
out = x1 - \operatorname{trunc}(x1 / x2) \times x2
$$

其中，`trunc`表示向0方向截断取整。

## 函数原型

每个算子分为两段式接口，必须先调用`aclnnTruncateModGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnTruncateMod`接口执行计算。

```cpp
aclnnStatus aclnnTruncateModGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnTruncateMod(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnTruncateModGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x1 | 输入 | aclTensor类型，TruncateMod计算的第一个输入张量。支持数据类型为FLOAT16、FLOAT、BFLOAT16、INT32、INT8、UINT8，数据格式为FORMAT_ND。 |
| x2 | 输入 | aclTensor类型，TruncateMod计算的第二个输入张量。支持数据类型为FLOAT16、FLOAT、BFLOAT16、INT32、INT8、UINT8，数据格式为FORMAT_ND。 |
| out | 输出 | aclTensor类型，TruncateMod计算结果。支持数据类型为FLOAT16、FLOAT、BFLOAT16、INT32、INT8、UINT8，数据格式为FORMAT_ND。 |
| workspaceSize | 输出 | uint64_t指针，返回用户需要在Device侧申请的workspace大小。 |
| executor | 输出 | aclOpExecutor指针的地址，返回包含算子计算流程的执行器。 |

### 返回值说明

| 返回值 | 描述 |
| --- | --- |
| ACLNN_SUCCESS | 接口调用成功。 |
| ACLNN_ERR_PARAM_NULLPTR | 参数`x1`、`x2`、`out`或`executor`为空指针。 |
| 其他非0返回值 | 接口调用失败。 |

## aclnnTruncateMod

### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| workspace | 输入 | Device侧workspace内存地址。 |
| workspaceSize | 输入 | workspace内存大小，单位为Byte。 |
| executor | 输入 | `aclnnTruncateModGetWorkspaceSize`接口返回的执行器。 |
| stream | 输入 | aclrtStream类型，指定执行任务的AscendCL Stream。 |

### 返回值说明

| 返回值 | 描述 |
| --- | --- |
| ACLNN_SUCCESS | 接口调用成功。 |
| ACLNN_ERR_PARAM_INVALID | `workspaceSize`小于接口内部所需的输入连续化workspace大小。 |
| 其他非0返回值 | 接口调用失败。 |

## 约束说明

- `x1`、`x2`和`out`的数据类型必须一致，支持的数据类型为FLOAT16、FLOAT、BFLOAT16、INT32、INT8、UINT8。
- `x1`、`x2`和`out`的数据格式必须为FORMAT_ND。
- `out`的shape为`x1`与`x2`按照广播规则推导得到的shape；当两个输入shape在某一维度上不相等且均不为1时，shape推导失败。
- 当前实现中tiling数据和Kernel侧shape寻址数组最多保存8维shape信息，因此输入与输出Tensor维度不应超过8维。
- 当前实现支持动态rank和动态shape。
- 输入数据应满足对应数据类型下除法操作的合法性要求；当前实现未定义`x2`元素为0时的特殊处理语义。

## 调用示例

示例代码如下，仅用于展示接口调用流程。

```cpp
#include <cstdint>
#include <vector>
#include "acl/acl.h"
#include "aclnn_truncate_mod.h"

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    std::vector<float> hostX1 = {5.5f, 7.0f, -5.5f, 10.0f, 3.0f};
    std::vector<float> hostX2 = {2.0f, 3.0f, 2.0f, 4.0f, 2.0f};
    std::vector<float> hostOut(5, 0.0f);
    std::vector<int64_t> shape = {5};
    std::vector<int64_t> strides = {1};

    void *devX1 = nullptr;
    void *devX2 = nullptr;
    void *devOut = nullptr;
    aclrtMalloc(&devX1, hostX1.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devX2, hostX2.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devOut, hostOut.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(devX1, hostX1.size() * sizeof(float), hostX1.data(),
        hostX1.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devX2, hostX2.size() * sizeof(float), hostX2.data(),
        hostX2.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    aclTensor *x1 = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT,
        strides.data(), 0, ACL_FORMAT_ND, nullptr, 0, devX1);
    aclTensor *x2 = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT,
        strides.data(), 0, ACL_FORMAT_ND, nullptr, 0, devX2);
    aclTensor *out = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT,
        strides.data(), 0, ACL_FORMAT_ND, nullptr, 0, devOut);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnTruncateModGetWorkspaceSize(x1, x2, out, &workspaceSize, &executor);

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    aclnnTruncateMod(workspace, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(hostOut.data(), hostOut.size() * sizeof(float), devOut,
        hostOut.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    // hostOut期望结果为：{1.5, 1.0, -1.5, 2.0, 1.0}

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(out);
    aclrtFree(devX1);
    aclrtFree(devX2);
    aclrtFree(devOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```
