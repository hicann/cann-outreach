# AtanhCustom AscendC 自定义算子工程

功能：
- 输入：FP16 ND Tensor
- 输出：FP16 ND Tensor
- 算子：逐元素 Atanh
- 平台：AscendC / CANN 7.x~8.x（需根据实际版本调整）

目录：
op_proto/     算子定义
op_host/      tiling与shape推导
op_kernel/    AI Core kernel实现
test/         精度验证脚本

生成标准CANN工程时，可使用msopgen生成基础CMake结构：
msopgen gen -i op_proto/atanh_custom.json -c ai_core-ascend910b -lan cpp -out ./build
