# 上海珑京信息科技有限公司

## 团队信息

- 提交者: HexSama
- 身份: 企业员工
- 单位: 上海珑京信息科技有限公司

## 成员

- HexSama (Genshin-moyu): 提交者

## 算子: op_03_relu

CopyIn → CompareScalar → Select → CopyOut。host 侧 tiling 通过 PlatformAscendC 拿真实核数和 UB 容量，元素总数按核数 CeilDiv 均分，尾核按剩余量处理；单块封顶 4096 元素并按 256B 对齐（fp32 即 64 的倍数、fp16 为 128），这是 Compare 接口的对齐硬要求；fp16/bf16 和 fp32 用 tilingKey 分成两套模板。核内 BUFFER_NUM=2 双缓冲，搬运和计算流水重叠；relu 本体用 CompareScalar(x&gt;0) 出掩码、Select 按掩码取 x 或 0，掩码放 TBuf 不占队列，不走 Max 是因为 Compare+Select 这套组合在 CANN 各版本上行为一致，跨版本稳。16384 个元素 8 核各分 2048，单块一次搬算完。
