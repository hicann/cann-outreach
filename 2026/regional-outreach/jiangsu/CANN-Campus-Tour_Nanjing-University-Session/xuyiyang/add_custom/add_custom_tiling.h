/*
 * add_custom host 侧实现
 * 包含: shape 推导 / 数据类型与格式校验 / tiling 计算
 * 框架: Ascend C (CANN kernel 框架, 非高层 API 版本)
 */
#include "register/op_def_registry.h"

using namespace ge;

namespace ops {

constexpr int32_t FP16_ALIGN_NUM = 16; // fp16 一个 32B block 的元素数

class AddCustom : public OpDef {
public:
    explicit AddCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({DT_FLOAT16})
            .Format({FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({DT_FLOAT16})
            .Format({FORMAT_ND});
        this->Output("z")
            .DataType({DT_FLOAT16})
            .Format({FORMAT_ND});
        this->Attr("tileLength")
            .Int(2048); // 每块处理元素个数, 与 device 侧 TILE_LENGTH 对应
        OpDefFactoryRegister(name, this);
    }
};
OP_ADD(AddCustom);

} // namespace ops
