#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t dtype_size = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_x));
        uint32_t length_x = tensor_x->GetShapeSize();

        uint32_t align_elem = 32 / dtype_size;

        uint32_t block_num;
        uint32_t block_len;
        if (length_x < align_elem) {
            block_num = 1;
            block_len = length_x;
        } else {
            uint32_t max_cores = length_x / align_elem;
            block_num = static_cast<uint32_t>(num_cores_aiv) < max_cores
                            ? static_cast<uint32_t>(num_cores_aiv)
                            : max_cores;
            if (block_num == 0) {
                block_num = 1;
            }
            block_len = length_x / block_num;
            block_len = (block_len / align_elem) * align_elem;
        }

        uint32_t ub_cap = static_cast<uint32_t>(ub_size / (3 * dtype_size));
        uint32_t tile_len = block_len < ub_cap ? block_len : ub_cap;
        tile_len = (tile_len / align_elem) * align_elem;
        if (tile_len == 0) {
            tile_len = block_len;
        }
        if (tile_len == 0) {
            tile_len = 1;
        }

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->length = length_x;
        tiling->blockNum = block_num;
        tiling->blockLen = block_len;
        tiling->tileLen = tile_len;

        context->SetBlockDim(static_cast<int32_t>(block_num));

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
        context->SetOutputDataType(0,inputDataType);
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
}  // namespace ops
