# 东南大学

## 团队信息

- 提交者: RuiRin
- 身份: 学生
- 单位: 东南大学

## 成员

- RuiRin (gcw_rlZOWc6b): 提交者

## 算子: op_04_square

概述
Square（平方）算子是逐元素（Element-wise）运算的激活函数，计算输入张量的平方值：y = x²

技术规格
项目	说明
输入	input_x: tensor, 任意形状 (..., N)
输出	output: tensor, 与输入形状和类型相同
数据类型	float16, float32
数据格式	ND (N-dimensional)
维度范围	N ∈ [1, 10240]
架构设计

┌─────────────────────────────────────────────────────────┐
│                      Square 算子                         │
├─────────────────────────────────────────────────────────┤
│  Tiling (op_host)                                       │
│  ├── 获取平台信息 (coreNum, ubSize)                     │
│  ├── 多核切分: 每核 ≤ 32K 元素                          │
│  ├── UB 切分: 预留 8KB, 按 alignNum 对齐               │
│  └── 设置 BlockDim, TilingKey                          │
├─────────────────────────────────────────────────────────┤
│  Kernel (op_kernel)                                     │
│  ├── CopyIn:  DoubleBuffer 预取 GM → UB                │
│  ├── Compute: AscendC::Mul 向量指令 y = x * x          │
│  └── CopyOut: UB → GM                                  │
└─────────────────────────────────────────────────────────┘
核心代码
Tiling (square_tiling.cpp):


// 多核切分：每核 ≤ 32K 元素
constexpr int64_t MAX_ELEMENTS_PER_CORE = 32 * 1024;
int64_t maxCoresForData = (totalNum + MAX_ELEMENTS_PER_CORE - 1) / MAX_ELEMENTS_PER_CORE;
usedCoreNum = std::min(coreNum, maxCoresForData);

// UB 切分
constexpr uint64_t UB_RESERVED_BYTES = 8 * 1024;
int64_t maxUbFactor = (ubSize - UB_RESERVED_BYTES) / (4 * typeSize);
Kernel (square.h):


// DoubleBuffer Process
CopyIn(0, ubLength_);
for (int64_t i = 0; i &lt; loopCount; i++) {
    Compute(ubLength_);           // y = x * x
    CopyOut(progress, ubLength_);
    if (i + 1 &lt; loopCount) {
        CopyIn(progress + ubLength_, ubLength_);  // 预取下一tile
    }
}

// Compute: 使用向量 Mul 指令
AscendC::Mul(yBuf, xBuf, xBuf, currentNum);
优化点
优化项	效果
向量指令	AscendC::Mul 替代 for 循环批量计算
DoubleBuffer	CopyIn/Compute/CopyOut 三操作流水线并行
32字节对齐	符合 CANN 内存访问要求
多核负载均衡	避免单核处理大数据超时
UB 内存管理	预留 8KB + 按 alignNum 对齐
