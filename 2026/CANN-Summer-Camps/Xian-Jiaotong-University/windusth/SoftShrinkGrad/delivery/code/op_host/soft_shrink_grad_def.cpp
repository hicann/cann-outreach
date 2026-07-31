/*!
 * \file soft_shrink_grad_def.cpp
 * \brief SoftShrinkGrad 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {
class SoftShrinkGrad : public OpDef {
public:
    explicit SoftShrinkGrad(const char* name) : OpDef(name)
    {
    this->Input("input_grad")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("input_x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("output_y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Attr("lambd")
        .AttrType(OPTIONAL)
        .Float(0.5);
    this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(SoftShrinkGrad);
} // namespace ops
