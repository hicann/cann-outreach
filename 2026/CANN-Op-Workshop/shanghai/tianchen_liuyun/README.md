# 天臣集团

## 团队信息

- 提交者: chinayun_6401
- 身份: 企业员工
- 单位: 天臣集团

## 成员

- chinayun_6401 (chinayun_6401): 提交者

## 算子: op_02_mul

本算子基于 Ascend C 实现矢量乘法，通过 Tiling 完成数据分块，Kernel 采用多核并行计算，将输入数据搬运至 UB 后调用 `AscendC::Mul` 完成 `z=x*y`，最后将结果写回全局内存，支持 float32 和 float16 数据类型。
