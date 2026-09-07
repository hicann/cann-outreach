#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        MulTilingData *tiling = context->GetTilingData<MulTilingData>();

        const gert::StorageShape* x_shape = context->GetInputShape(0);
        uint32_t totalLength = 1;
        for (int i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++) {
            totalLength *= x_shape->GetStorageShape().GetDim(i);
        }

        const uint32_t coreNum = 8;
        uint32_t blockLength = (totalLength + coreNum - 1) / coreNum;

        // 使用接近上限的 Local Memory（250KB）
        const uint32_t localMemSize = 250 * 1024;
        uint32_t elemBytes = (DT_X == ge::DT_FLOAT) ? 4 : 2;
        const uint32_t BUFFER_NUM = 2;
        const uint32_t ALIGN_ELEMS = 64 / elemBytes;   // 64字节对齐
        uint32_t maxTileLen = localMemSize / (elemBytes * BUFFER_NUM);
        maxTileLen = (maxTileLen / ALIGN_ELEMS) * ALIGN_ELEMS;
        if (maxTileLen == 0) maxTileLen = ALIGN_ELEMS;

        // 目标：尽量 tileNum = 1，如果 blockLength 能装下
        uint32_t tileNum = 1;
        if (blockLength > maxTileLen * BUFFER_NUM) {
            tileNum = (blockLength + maxTileLen * BUFFER_NUM - 1) / (maxTileLen * BUFFER_NUM);
        }

        tiling->totalLength = totalLength;
        tiling->tileNum = tileNum;

        context->SetBlockDim(coreNum);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape* x1_shape = context->GetInputShape(0);
        gert::Shape* z_shape = context->GetOutputShape(0);
        *z_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Mul : public OpDef {
    public:
        explicit Mul(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("z")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Mul);
}