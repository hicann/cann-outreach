/*!
 * \file square_def.cpp
 * \brief Square 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {
class Square : public OpDef {
public:
    explicit Square(const char* name) : OpDef(name)
    {
    this->Input("input_x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("output")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Square);
} // namespace ops
