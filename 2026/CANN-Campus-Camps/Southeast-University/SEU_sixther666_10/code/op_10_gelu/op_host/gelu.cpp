// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const auto *input_shape = context->GetInputShape(0);
        const auto *input_desc = context->GetInputDesc(0);
        if (input_shape == nullptr || input_desc == nullptr || num_cores_aiv <= 0) {
            return ge::GRAPH_FAILED;
        }
        const ge::DataType dtype_input_x = input_desc->GetDataType();
        if (dtype_input_x != ge::DT_FLOAT16 && dtype_input_x != ge::DT_FLOAT) {
            return ge::GRAPH_FAILED;
        }
        const int64_t shape_size = input_shape->GetStorageShape().GetShapeSize();
        if (shape_size < 0) {
            return ge::GRAPH_FAILED;
        }
        const uint64_t length = static_cast<uint64_t>(shape_size);

        // Float32 computes directly in the queues; half needs two conversion buffers.
        constexpr uint64_t reserved_ub = 8192;
        constexpr uint32_t tile_alignment = 64;
        constexpr uint32_t max_tile_length = 2048;
        const uint32_t element_size = ge::GetSizeByDataType(dtype_input_x);
        const uint32_t float_buffers = dtype_input_x == ge::DT_FLOAT ? 4 : 6;
        const uint32_t bytes_per_element = 4 * element_size + float_buffers * sizeof(float);
        if (ub_size < reserved_ub + tile_alignment * bytes_per_element) {
            return ge::GRAPH_FAILED;
        }
        uint64_t tile_length = (ub_size - reserved_ub) / bytes_per_element;
        tile_length = tile_length < max_tile_length ? tile_length : max_tile_length;
        tile_length = tile_length / tile_alignment * tile_alignment;
        // Expose more cores for medium tensors, with at least 256 elements per tile.
        uint64_t target_tile = length / num_cores_aiv + (length % num_cores_aiv != 0);
        target_tile = target_tile < 256 ? 256 : target_tile;
        target_tile = (target_tile + tile_alignment - 1) / tile_alignment * tile_alignment;
        tile_length = target_tile < tile_length ? target_tile : tile_length;
        const uint64_t tile_count = length / tile_length + (length % tile_length != 0);
        const uint32_t core_count = tile_count == 0 ? 1 :
            static_cast<uint32_t>(tile_count < static_cast<uint64_t>(num_cores_aiv)
                ? tile_count : num_cores_aiv);

        const uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);
        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling->length = length;
        tiling->tileLength = static_cast<uint32_t>(tile_length);
        tiling->coreCount = core_count;
        context->SetBlockDim(core_count);
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
        const auto *input_shape = context->GetInputShape(0);
        auto *output_shape = context->GetOutputShape(0);
        if (input_shape == nullptr || output_shape == nullptr) {
            return GRAPH_FAILED;
        }
        *output_shape = *input_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const DataType dtype = context->GetInputDataType(0);
        if (dtype != DT_FLOAT16 && dtype != DT_FLOAT) {
            return GRAPH_FAILED;
        }
        return context->SetOutputDataType(0, dtype);
    }
}  // namespace ge

namespace ops {
    class Gelu : public OpDef {
    public:
        explicit Gelu(const char *name) : OpDef(name) {
            this->Input("input_x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Gelu);
}  // namespace ops
