# Iamax算子

## 算子概述

Iamax 算子实现了查找向量中"模"最大元素的索引，核心运算为遍历向量计算"模"并比较大小。该算子返回 1-based 索引，遵循 BLAS 惯例，常用于主元选择和迭代算法中。实数（FP32）场景取绝对值 |x[i]|；复数（COMPLEX64）场景取 1-范数口径的"模" |Re(x[i])| + |Im(x[i])|（Netlib SCABS1 语义，非欧氏模）。

数学表达式：

```
result = argmax_i |x[i]|              # aclblasIsamax（实数绝对值）
result = argmax_i (|Re(x[i])| + |Im(x[i])|)   # aclblasIcamax（复数1-范数"模"）
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasIsamax | 查找 FP32 向量中绝对值最大元素的 1-based 索引 |
| aclblasIcamax | 查找 COMPLEX64 向量中"模"（1-范数 |Re|+|Im|）最大元素的 1-based 索引 |

## 算子执行接口

### aclblasIsamax

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasIsamax(aclblasHandle_t handle, int n, const float *x, int incx, int *result)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量元素个数，Host 内存 |
| x | 输入 | const float* | 指向 float 向量的 device 指针，Device 内存 |
| incx | 输入 | int | 向量 x 中相邻元素之间的步长，Host 内存 |
| result | 输出 | int* | 绝对值最大元素的 1-based 索引，Device 内存 |

#### 约束说明

- n < 0 时返回 ACLBLAS_STATUS_INVALID_VALUE
- n = 0 或 incx < 1 时 result 写 0，返回 ACLBLAS_STATUS_SUCCESS
- handle 不可为 nullptr
- x、result 不可为 nullptr
- 当多个元素绝对值相同时，返回索引最小的元素

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

class AclContext {
public:
    explicit AclContext(int deviceId) : deviceId_(deviceId) {}

    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

struct AclrtMemDeleter {
    void operator()(void* ptr) const
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

struct AclblasHandleDeleter {
    void operator()(aclblasHandle_t handle) const
    {
        if (handle != nullptr) {
            aclblasDestroy(handle);
        }
    }
};

int aclblasIsamaxTest(AclContext& ctx)
{
    constexpr int n = 5;
    constexpr int incx = 1;
    constexpr size_t xBytes = n * sizeof(float);

    float hX[n] = {1.0f, -5.0f, 3.0f, -4.0f, 2.0f};

    void *rawX = nullptr;
    auto aclRet = aclrtMalloc(&rawX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dX(rawX);

    void *rawResult = nullptr;
    aclRet = aclrtMalloc(&rawResult, sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dResult(rawResult);

    aclRet = aclrtMemcpy(dX.get(), xBytes, hX, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasIsamax(
        static_cast<aclblasHandle_t>(handle.get()), n,
        static_cast<const float*>(dX.get()), incx,
        static_cast<int*>(dResult.get()));
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    int result = 0;
    aclRet = aclrtMemcpy(&result, sizeof(int), dResult.get(), sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 预期结果：2（1-based 索引，|-5| 最大）
    printf("isamax result = %d\n", result);

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasIsamaxTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```

### aclblasIcamax

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasIcamax(aclblasHandle_t handle, int n, const aclblasComplex *x, int incx, int *result)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 复数向量元素个数，Host 内存，n ≥ 0 |
| x | 输入 | const aclblasComplex* | 指向 COMPLEX64（单精度复数，实部/虚部各 float32 交错存储）向量的 device 指针，Device 内存 |
| incx | 输入 | int | 向量 x 中相邻复数元素之间的步长，Host 内存 |
| result | 输出 | int* | “模”最大元素的 1-based 索引，Device 内存 |

#### 约束说明

- 复数“模”为 1-范数口径：|x[k]| = |Re(x[k])| + |Im(x[k])|（Netlib SCABS1 语义），非欧氏模
- n < 0 时返回 ACLBLAS_STATUS_INVALID_VALUE
- n = 0 或 incx ≤ 0（含 incx = 0 与负步长）为合法 quick return：不触发归约计算，result 写 0，返回 ACLBLAS_STATUS_SUCCESS；本算子不做负步长反向遍历
- handle 不可为 nullptr（返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR）
- n > 0 时 x、result 不可为 nullptr（返回 ACLBLAS_STATUS_INVALID_VALUE）
- 输出为 1-based 整数索引；当多个元素取得相同最大“模”时，返回索引最小的元素（对齐 Netlib 严格大于才更新语义）
- 接口声明位于 include/cann_ops_blas.h，供各产品线共用
