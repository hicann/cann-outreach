# 东南大学

## 团队信息

- 提交者: RuiRin
- 身份: 学生
- 单位: 东南大学

## 成员

- RuiRin (gcw_rlZOWc6b): 提交者

## 算子: op_02_mul

整体结构

mul.cpp (kernel 入口)
├── KernelMul 类 — 核心计算逻辑
│   ├── Init — 初始化队列、GlobalTensor
│   └── Process — CopyIn → Compute → CopyOut
└── mul 函数 — 模板入口，根据数据类型实例化
KernelMul 类
Init 函数：


__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t blockLength) {
    // 1. 初始化队列缓冲区（单缓冲，depth=1）
    pipe.InitBuffer(xQue, 1, blockLength * sizeof(DT_X));
    pipe.InitBuffer(yQue, 1, blockLength * sizeof(DT_X));
    pipe.InitBuffer(zQue, 1, blockLength * sizeof(DT_X));

    // 2. 绑定 GlobalTensor，每个核处理一段连续数据
    xGm.SetGlobalBuffer((__gm__ DT_X*)x + AscendC::GetBlockIdx() * blockLength, blockLength);
    yGm.SetGlobalBuffer((__gm__ DT_X*)y + AscendC::GetBlockIdx() * blockLength, blockLength);
    zGm.SetGlobalBuffer((__gm__ DT_X*)z + AscendC::GetBlockIdx() * blockLength, blockLength);
}
Process 函数（CopyIn → Compute → CopyOut）：


// 1. CopyIn: GM → UB
LocalTensor xLocal = xQue.AllocTensor();  // 分配 UB 空间
LocalTensor yLocal = yQue.AllocTensor();
DataCopy(xLocal, xGm, blockLength);      // 从 GM 复制到 UB
DataCopy(yLocal, yGm, blockLength);
xQue.EnQue(xLocal);                      // 入队
yQue.EnQue(yLocal);

// 2. Compute: UB → UB（向量乘法）
LocalTensor xBuf = xQue.DeQue();        // 取输入
LocalTensor yBuf = yQue.DeQue();
LocalTensor zBuf = zQue.AllocTensor();   // 分配输出空间
Mul(zBuf, xBuf, yBuf, blockLength);      // 向量乘法指令
xQue.FreeTensor(xBuf);                   // 释放输入
yQue.FreeTensor(yBuf);
zQue.EnQue(zBuf);                        // 结果入队

// 3. CopyOut: UB → GM
LocalTensor zResult = zQue.DeQue();     // 取结果
DataCopy(zGm, zResult, blockLength);   // 写回 GM
zQue.FreeTensor(zResult);              // 释放
数据流

Host (TilingFunc)
    ↓ blockDim = core_num, tiling_data.blockLength
    ↓ &lt;&lt;&lt;core_num, stream&gt;&gt;&gt;(x, y, z, tiling)
Device (每个 core)
    ↓ SetGlobalBuffer(blockIdx * blockLength)
    ↓ CopyIn: GM → UB
    ↓ Compute: Mul(z = x * y)
    ↓ CopyOut: UB → GM
关键点
项目	值
缓冲策略	单缓冲 (depth=1)
计算指令	AscendC::Mul 向量指令
Tiling	Host 侧计算，每个核处理 length/core_num 元素
数据类型	DT_X = float32 或 float16
