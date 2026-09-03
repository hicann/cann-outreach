# aclblasIcamax 算子测试工程说明

> 适配硬件：Ascend 950PR　|　CANN 版本：9.1.0　|　数据类型：COMPLEX64 输入 → INT32 索引标量输出（1-based）

## 1. 算子与接口

`aclblasIcamax`：查找单精度复数向量 x 中"模"最大元素的索引。

```
result = argmax_i ( |Re(x[k])| + |Im(x[k])| ) ，i = 1..n，k = 1+(i-1)*incx（1-based）
```

- **复数"模"为 1-范数口径** `|Re| + |Im|`（Netlib SCABS1），**非欧氏模**；
- 相同最大值取**最小索引**（严格大于才更新，对齐 Netlib `icamax`）；
- 输出为离散整数索引，精度判定**精确相等（bit-exact）**；
- `n = 0` 或 `incx ≤ 0`（含 0 与负步长）为合法 quick return：result = 0，返回 `ACLBLAS_STATUS_SUCCESS`，不反向遍历；
- `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；handle 空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；n > 0 时 x/result 空指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。

接口声明位于仓根 `include/cann_ops_blas.h`（与同族接口 `aclblasIsamax` 逐参数对齐），算子实现位于 `blas/iamax/arch35/`。

## 2. 目录结构

```
test/iamax/icamax/
├── CMakeLists.txt            # arch35 下注册 icamax_test 目标
├── icamax_param.h            # CSV 参数解析（含 result 列 NULLPTR 标记、复数填充辅助）
├── icamax_golden.h           # CPU golden：cblas_icamax（Netlib icamax）+1 转 1-based
├── icamax_npu_wrapper.h      # NPU 包装：DeviceBuffer 分配/拷贝/同步，result 空指针负向入口
└── arch35/
    ├── icamax_test.cpp       # GTest 用例（NullHandle + CSV 驱动）
    └── icamax_test.csv       # 1200 条用例（1000 精度 + 200 性能）
```

## 3. 测试用例（icamax_test.csv）

共 1200 条，由 `gen_csv.py`（随任务提供）以固定种子确定性生成：

| 类别 | 前缀 | 条数 | 说明 |
|------|------|------|------|
| L0 基础 | TC_L0 | 10 | 小尺寸 × 步长（含 quick return 语义） |
| L1 尺寸 | TC_SQ | 38 | 1→1048576，质数/2 的幂及 ±1/非对齐 |
| L2 步长 | TC_INC | 21 | incx ∈ ±1/±2/±3/0 |
| L5 填充 | TC_FL | 18 | 均匀随机/全零/常量/交替/极端/Inf/NaN + tie 专项 |
| L6 边界 | TC_ED | 13 | n=0、incx≤0、负 n、空指针 x/result |
| EX 扩展 | TC_EX | 900 | 尺寸×步长×填充确定性采样 |
| PF 性能 | TC_PF | 200 | 3 条任务书典型 case（n=1048576/2097152/4194304，incx=1）等，连续访存 |

CSV 列：`case_name,description,n,incx,x,result,random_seed,expect_result`。`x` 填充模式由仓测试框架 `test/frame/fill.h` 解析；`result` 列标记 `NULLPTR` 表示 result 空指针负向用例。

精度比对：golden 由 cblas（Netlib BLAS 复数 `icamax`）生成，0-based +1 转 1-based 后与 NPU 输出**精确相等**比对；性能用例不参与 golden 比对。

## 4. 复现步骤

### 4.1 构建算子测试

```bash
# 在 ops-blas 仓根目录，指定目标 SoC 与测试名
./build.sh --test=icamax --soc=ascend950    # 具体参数以仓根 build.sh 说明为准
```

arch35 编译产物目标名为 `icamax_test`。

### 4.2 运行精度测试

```bash
# 方式一：随任务脚本（构建 + 运行 + 解析 PASS/FAIL，自动排除 TC_PF）
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend950 --csv ./icamax_test.csv

# 方式二：直接运行 GTest 二进制
./icamax_test --gtest_filter='*Icamax*' --gtest_color=no
```

判定：任一用例 NPU 输出索引 ≠ golden 索引即失败（精确相等，matched_ratio = 1 才通过）。

### 4.3 运行性能测试

```bash
python verify_performance.py --repo /path/to/ops-blas --soc ascend950 --timeout 3600
```

- 判定口径：warmup 后有效采样 >50 次取平均单次耗时，≤ `gpu_baseline.csv` 标杆耗时；
- 任务书 §3.3 三档标杆（us）：n=1048576→24.52，n=4194304→25.81，n=16777216→31.44。

### 4.4 重新生成用例（可选）

```bash
python gen_csv.py                            # 默认 1000 精度 + 200 性能
python gen_csv.py --accuracy 1500 --perf 300 # 扩展条数
python gen_csv.py --seed 12345               # 更换种子
```

## 5. 开发期 CPU 侧仿真自验

无法上板的环境下，可用随交付的 `verify/icamax_sim_verify`（CPU 仿真，复刻 host tiling + SIMT 归约 + reduce 全流程）预先校验算法语义：

```bash
g++ -std=c++17 -O2 -I verify -I test/frame -I test/iamax/icamax -I include \
    -I blas/common/helper -I /usr/include/x86_64-linux-gnu \
    verify/icamax_sim_verify.cpp -o icamax_sim_verify -lblas
./icamax_sim_verify test/iamax/icamax/arch35/icamax_test.csv 20 24 40 48
```

以多种核数（20/24/40/48）对全部精度用例与 cblas golden 比对，预期 1000/1000 精确一致。
