# 上海珑京信息科技有限公司

## 团队信息

- 提交者: HexSama
- 身份: 企业员工
- 单位: 上海珑京信息科技有限公司

## 成员

- HexSama (Genshin-moyu): 提交者

## 算子: op_01_sub

实现矢量减法算子 z = x - y，思路参考官方 add 的写法。host 侧 run_kernel 遍历输入 shape 算出总元素数（8×2048=16384），开 8 个核均分，每核 2048 个元素再切 8 个 tile，配 BUFFER_NUM=2 的双缓冲，单块 128 个元素。核内三段流水：CopyIn 用 DataCopy 把 x、y 搬进 UB，Compute 调 AscendC::Sub 逐块算减法，CopyOut 写回 GM，中间靠 EnQue/DeQue 衔接，搬运和计算能重叠。fp16、fp32 各实例化一个模板，run_kernel 按 dtype 分发，编译验证通过。
