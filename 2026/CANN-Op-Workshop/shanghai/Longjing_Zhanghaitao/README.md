# 上海珑京信息科技有限公司

## 团队信息

- 提交者: HexSama
- 身份: 企业员工
- 单位: 上海珑京信息科技有限公司

## 成员

- HexSama (Genshin-moyu): 提交者

## 算子: op_02_mul

实现矢量乘法算子 z = x * y，工程为注册制自定义算子结构。host 侧 TilingFunc 遍历输入 shape 得到总元素数 16384，SetBlockDim(8) 起 8 核均分，每核再切 8 个 tile 配双缓冲；InferShape 令输出 shape 等于输入，InferDataType 令输出 dtype 跟随输入。kernel 侧 KernelMul 按 CopyIn（DataCopy 搬入 UB）→ Compute（AscendC::Mul 逐块相乘）→ CopyOut（写回 GM）三段流水处理，靠 EnQue/DeQue 衔接，类模板化适配 fp16/fp32，dtype 分发由框架的 TilingKey 机制完成，编译验证通过。
