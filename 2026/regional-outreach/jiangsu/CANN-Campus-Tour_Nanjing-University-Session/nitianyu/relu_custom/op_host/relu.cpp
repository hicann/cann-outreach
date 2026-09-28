#include "relu_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    ReluTilingData tiling;
    
    // 获取 4 维 Shape [N4, N3, N2, N1] 并展平为 1 维计算总长度
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    uint32_t totalLength = 1;
    for (int i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++) {
        totalLength *= x_shape->GetStorageShape().GetDim(i);
    }

    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    // 简单切分策略：均分给所有 Vector Core，每个 Core 内部再分 8 块做流水线
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(8);

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ReluCustom : public OpDef {
public:
    explicit ReluCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(ReluCustom);
}
