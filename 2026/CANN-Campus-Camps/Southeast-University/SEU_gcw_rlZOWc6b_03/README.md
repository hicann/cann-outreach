# 东南大学

## 团队信息

- 提交者: RuiRin
- 身份: 学生
- 单位: 东南大学

## 成员

- RuiRin (gcw_rlZOWc6b): 提交者

## 算子: op_03_relu

整体架构

Relu = Tiling(Host) + Kernel(GPU/NPU)
       ↓                  ↓
  多核分割策略        向量计算
Host 侧 (relu_tiling.cpp)
负责 tiling 计算，将任务分配到多个 core：


// 1. 多核切分：92160 元素分给 64 个 core
blockFactor = 92160 / 64 = 1440  // 每个 core 处理 1440 个元素

// 2. UB 切分：每个 core 的 1440 元素分成多次处理
ubFactor = min(1440, UB可用空间) = 1440  // 单次处理完

// 3. 设置 block 数量
context-&gt;SetBlockDim(64);
Kernel 侧 (relu.h)
数据结构：


template &lt;typename T&gt;
class Relu {
    TPipe pipe;                              // 计算管道
    TQue&lt;QuePosition::VECIN, 2&gt; inputQueueX;  // 输入队列 (双缓冲)
    TQue&lt;QuePosition::VECOUT, 2&gt; outputQueueY; // 输出队列 (双缓冲)
    int64_t blockLength_;  // 每个核处理的数据量
    int64_t ubLength_;     // 每次UB处理的数据量
};
核心流程 (Process)：


Process() {
    loopCount = (totalNum + ubLength_ - 1) / ubLength_;
    
    for i in 0..loopCount:
        CopyIn(progress, currentNum);   // GM → UB
        Compute(currentNum);            // 计算 ReLU
        CopyOut(progress, currentNum);  // UB → GM
}
Compute 核心：


AscendC::Relu(yBuf, xBuf, currentNum);  // y = max(0, x)
内存布局：


每个核的 GM 地址偏移 = blockIdx * blockLength
每个核处理 blockLength 个元素
数据流

GM (输入)  ──CopyIn──&gt;  UB (inputQueueX)  ──Compute──&gt;  UB (outputQueueY)  ──CopyOut──&gt;  GM (输出)
                ↓                                                      ↑
           1440 元素                                              1440 元素
支持的数据类型
float32 (MODE_1)
float16 / bf16 (MODE_0)
