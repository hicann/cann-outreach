# 东南大学

## 团队信息

- 提交者: 程康博
- 身份: 学生
- 单位: 东南大学

## 成员

- 程康博 (ArgoNorthrop): 提交者

## 算子: op_03_relu

基于 AscendC 编程语言完成 Lesson 03 ReLU 激活算子（y = max(0, x)）的原生开发，支持 float32 和 float16 数据类型，输入输出张量形状为 (8, 2048)，数据格式 ND。

本次是开源仓规范化框架（op_host 拆成 relu_tiling.cpp / relu_infershape.cpp / relu_def.cpp 三文件）的首次实现，和前两题（Sub/Mul 单文件模式）差异较大，完整实现了三部分：

1. **op_host/relu_tiling.cpp（host 侧 tiling）**：
   - 从 TilingContext 读输入 shape（注意 TilingContext::GetInputShape 返回 StorageShape*，需通过 GetStorageShape() 拿到 gert::Shape 后再遍历乘 totalNum）
   - dtype → 32B 对齐系数（fp32=8, fp16=16）
   - CeilDiv(totalNum, coreNum) 算 blockFactorRaw → CeilAlign 对齐
   - CeilDiv(totalNum, blockFactor) 算实际 SetBlockDim（可以 &lt; coreNum）
   - ubFactor = CeilAlign(2048, alignElements)
   - 根据 dtype 选 schMode（fp16/bf16→RELU_TPL_SCH_MODE_0→half, else→RELU_TPL_SCH_MODE_1→float），GET_TPL_TILING_KEY + SetTilingKey 注册

2. **op_kernel/relu.h（kernel 侧类实现）**：
   - namespace NsRelu::Relu&lt;T&gt; 模板类，private 成员加了 tilingDataPtr_（Init 里存 tiling 指针，Process 里读 totalNum）
   - Init：pipe.InitBuffer 两个 queue（VECIN 2×ubFactor + VECOUT 2×ubFactor）+ SetGlobalBuffer + 从 tiling 读 blockFactor/ubFactor
   - Process：从 tilingDataPtr_-&gt;totalNum 算每核真实长度 = min(blockFactor, totalNum - blockFactor*coreIdx)，空核跳过，最后一块 currentNum 正确截断
   - CopyIn / Compute / CopyOut：AllocTensor → DataCopy → DeQue → AscendC::Relu → EnQue → FreeTensor，流水线三段式

3. **op_host/relu_infershape.cpp / relu_def.cpp / op_kernel/relu.cpp（kernel 入口）**：模板已完整，无需改动
