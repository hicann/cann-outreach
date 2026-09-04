// Host侧Tiling实现
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    // The fixed-size test shape fits in UB when each core handles one tile.
    constexpr uint32_t TILE_NUM = 1;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);

        const ge::DataType dtype_x = tensor_x->GetDataType();
        const uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        const gert::StorageShape *x_shape = context->GetInputShape(0);
        uint64_t length_x = 1;
        const auto &storage_shape = x_shape->GetStorageShape();
        for (int64_t i = 0; i < storage_shape.GetDimNum(); ++i) {
            length_x *= static_cast<uint64_t>(storage_shape.GetDim(i));
        }

        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->length = static_cast<uint32_t>(length_x);
        tiling->tileNum = TILE_NUM;

        uint32_t block_dim = static_cast<uint32_t>(platform.GetCoreNumAiv());
        if (block_dim == 0) {
            block_dim = 1;
        }
        if (block_dim > tiling->length) {
            block_dim = tiling->length;
        }
        while (block_dim > 1 && (tiling->length % block_dim) != 0) {
            --block_dim;
        }
        context->SetBlockDim(block_dim);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *z_shape = context->GetOutputShape(0);
        *z_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return GRAPH_SUCCESS;
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
}  // namespace ops
