// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        if (context == nullptr || context->GetPlatformInfo() == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
        if (tensor_x == nullptr || tensor_y == nullptr || num_cores_aiv <= 0) {
            return ge::GRAPH_FAILED;
        }
        const ge::DataType dtype_x = tensor_x->GetDataType();
        if ((dtype_x != ge::DT_FLOAT && dtype_x != ge::DT_FLOAT16) ||
            tensor_y->GetDataType() != dtype_x) {
            return ge::GRAPH_FAILED;
        }
        const int64_t length_x = tensor_x->GetShapeSize();
        if (length_x < 0 || static_cast<uint64_t>(length_x) > UINT32_MAX ||
            tensor_y->GetShapeSize() != length_x) {
            return ge::GRAPH_FAILED;
        }

        // Three queues, each with two buffers. Limit each tile to 1024 elements.
        const uint64_t dtype_size_x = ge::GetSizeByDataType(dtype_x);
        const uint64_t align_num = 32 / dtype_size_x;
        const uint64_t max_tile_length =
            (ub_size / (3 * 2 * dtype_size_x) / align_num) * align_num;
        if (max_tile_length == 0) {
            return ge::GRAPH_FAILED;
        }
        const uint64_t tile_length = max_tile_length < 1024 ? max_tile_length : 1024;

        // Split in units of 32 bytes so distinct cores never share a DMA block.
        const uint64_t total_length = static_cast<uint64_t>(length_x);
        uint64_t block_num = 1;
        uint64_t block_length = 0;
        if (total_length > 0) {
            const uint64_t aligned_blocks = (total_length + align_num - 1) / align_num;
            block_num = static_cast<uint64_t>(num_cores_aiv);
            block_num = block_num < aligned_blocks ? block_num : aligned_blocks;
            const uint64_t blocks_per_core = (aligned_blocks + block_num - 1) / block_num;
            block_length = blocks_per_core * align_num;
            block_length = block_length < total_length ? block_length : total_length;
            block_num = (total_length + block_length - 1) / block_length;
        }

        // Keep the template's datatype-based kernel selection.
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling->length = static_cast<uint32_t>(total_length);
        tiling->blockLength = static_cast<uint32_t>(block_length);
        tiling->tileLength = static_cast<uint32_t>(tile_length);
        context->SetBlockDim(static_cast<uint32_t>(block_num));
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace == nullptr) {
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        if (context == nullptr) {
            return GRAPH_FAILED;
        }
        const auto *inputShape = context->GetInputShape(0);
        auto *outputShape = context->GetOutputShape(0);
        if (inputShape == nullptr || outputShape == nullptr) {
            return GRAPH_FAILED;
        }
        *outputShape = *inputShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const auto dtype = context->GetInputDataType(0);
        if ((dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) ||
            context->GetInputDataType(1) != dtype) {
            return ge::GRAPH_FAILED;
        }
        context->SetOutputDataType(0, dtype);
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