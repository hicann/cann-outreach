#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    constexpr uint32_t BUFFER_NUM = 2;
    constexpr uint32_t QUEUE_NUM = 3;
    constexpr uint32_t DATA_BLOCK_SIZE = 32;
    constexpr uint32_t MIN_ELEMENTS_PER_CORE = 1024;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t dtype_size_x = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_x));
        uint32_t total_length = static_cast<uint32_t>(tensor_x->GetShapeSize());

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        uint32_t core_num = std::max(num_cores_aiv, 1);
        if (total_length != 0) {
            uint32_t cores_needed = (total_length + MIN_ELEMENTS_PER_CORE - 1) / MIN_ELEMENTS_PER_CORE;
            core_num = std::min(core_num, std::max(cores_needed, 1U));
        } else {
            core_num = 1;
        }

        uint32_t block_length = total_length == 0 ? 0 : (total_length + core_num - 1) / core_num;
        uint32_t align_num = DATA_BLOCK_SIZE / dtype_size_x;
        uint64_t max_tile_length = ub_size / (BUFFER_NUM * QUEUE_NUM * dtype_size_x);
        max_tile_length = (max_tile_length / align_num) * align_num;
        max_tile_length = std::max<uint64_t>(max_tile_length, align_num);

        uint32_t tile_length = align_num;
        if (block_length != 0) {
            uint64_t wanted_length = std::min<uint64_t>(block_length, max_tile_length);
            tile_length = static_cast<uint32_t>(
                ((wanted_length + align_num - 1) / align_num) * align_num);
        }

        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->totalLength = total_length;
        tiling->blockLength = block_length;
        tiling->tileLength = tile_length;

        context->SetBlockDim(core_num);
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
        context->SetOutputDataType(0, context->GetInputDataType(0));
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