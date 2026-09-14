# 北京海致科技集团股份有限公司

## 团队信息

- 提交者: JingHuang
- 身份: 企业员工
- 单位: 北京海致科技集团股份有限公司

## 成员

- JingHuang (qq_50236812): 提交者

## 算子: op_01_mul

1. 三级流水线与双缓冲（Double Buffering）：
      • 使用 AscendC::TPipe 与 AscendC::TQue 构建 CopyIn -&gt; Compute -&gt;CopyOut三级流水线。
      • 设置 BUFFER_NUM = 2 实现乒乓双缓冲，有效掩盖数据搬运时延。
  2. 内存对齐与切分（Tiling）：
      • 形状为 (8, 2048)，单 Tile 处理长度为 128。
      • 对于 float32（4 字节），128 × 4 B = 512 B（16 × 32 B 对齐）；对于float16（2 字节），128 × 2 B = 256 B（8 × 32 B 对齐），完美满足片上 Unified Buffer 32 字节对齐要求。
  3. 多精度与通用多核适配：
      • 利用 C++ 模板支持 float32（float）与 float16（half）。
      • 在 run_kernel 中根据 info_x.tensors[0].dtype 动态分发调用，同时自适应availableCoreNum。
