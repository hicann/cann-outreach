# SyncBatchNormBackwardReduce

## 产品支持情况

| 产品 | 是否支持 |
|------|---------|
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

- **算子功能**：完成 SyncBatchNorm 反向传播的归约计算。对多路输入执行乘法、减法、乘法三步运算，输出中间梯度和最终梯度。

- **计算公式**：

```
dy_mean      = mean * sum_dy              (Mul)
sum_dy_xmu   = sum_dy_dx_pad - dy_mean    (Sub)
y            = sum_dy_xmu * invert_std    (Mul)
```

其中 `sum_dy`、`sum_dy_dx_pad`、`mean`、`invert_std` 为输入，`sum_dy_xmu`、`y` 为输出，所有张量 shape 相同。

对于 FLOAT16/BF16 类型，输入先 Cast 到 FLOAT32 进行计算，再 Cast 回原数据类型（RoundMode: CAST_RINT）。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|--------------|------|---------|---------|
| sum_dy | 输入 | dy 的归约和。 | BF16, FLOAT16, FLOAT | ND |
| sum_dy_dx_pad | 输入 | dy*dx 的归约和（已 pad 对齐）。 | BF16, FLOAT16, FLOAT | ND |
| mean | 输入 | 前向传播保存的均值。 | BF16, FLOAT16, FLOAT | ND |
| invert_std | 输入 | 标准差的倒数。 | BF16, FLOAT16, FLOAT | ND |
| sum_dy_xmu | 输出 | sum_dy_dx_pad - mean*sum_dy 的结果。 | BF16, FLOAT16, FLOAT | ND |
| y | 输出 | sum_dy_xmu * invert_std 的结果。 | BF16, FLOAT16, FLOAT | ND |

## 约束说明

- 四路输入的 shape 必须完全相同。
- 输出 shape 与输入 shape 相同。
- 支持 ND 格式。
- 数据类型支持：BF16, FLOAT16, FLOAT。
- 当 `totalNum < 1024`（MIN_SPLIT_THRESHOLD）时使用单核处理，否则多核并行。
- 元素数量需对齐到 BLOCK_SIZE（32 字节），不足部分通过 pad 补齐。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| aclnn调用 | [test_aclnn_sync_batch_norm_backward_reduce](examples/test_aclnn_sync_batch_norm_backward_reduce.cpp) | 参见算子调用完成算子编译和验证。 |

### aclnn 调用示例

```cpp
#include "aclnn_sync_batch_norm_backward_reduce.h"

// 1. 获取 workspace 大小
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnSyncBatchNormBackwardReduceGetWorkspaceSize(
    sum_dy, sum_dy_dx_pad, mean, invert_std,
    sum_dy_xmu, y, &workspaceSize, &executor);

// 2. 申请 workspace 内存
void* workspaceAddr = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

// 3. 执行算子
aclnnSyncBatchNormBackwardReduce(workspaceAddr, workspaceSize, executor, stream);
```

### 编译与运行

```bash
# 编译算子包
bash build.sh

# 运行 UT 测试
bash tests/ut/run.sh

# 运行 aclnn 调用示例（需要 NPU 环境）
bash examples/run.sh              # 测试所有 dtype
bash examples/run.sh float16      # 仅测试 float16
bash examples/run.sh float32      # 仅测试 float32
bash examples/run.sh bfloat16     # 仅测试 bfloat16
```

## 目录结构

```
code/
├── op_host/          # 算子 host 侧实现 (def, infershape, tiling)
├── op_kernel/        # 算子 kernel 侧实现
├── examples/         # aclnn 调用示例
│   ├── test_aclnn_sync_batch_norm_backward_reduce.cpp
│   ├── CMakeLists.txt
│   └── run.sh
├── tests/ut/         # 单元测试
│   ├── op_host/      # Tiling UT (16个用例)
│   ├── op_kernel/    # Kernel UT (14个用例)
│   │   └── sync_batch_norm_backward_reduce_data/
│   │       ├── gen_data.py       # 测试数据生成
│   │       └── compare_data.py   # 精度比对
│   ├── common/       # 测试公共工具
│   ├── CMakeLists.txt
│   └── run.sh
├── CMakeLists.txt
├── build.sh
└── README.md
```
