// Host侧Tiling实现
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        constexpr int64_t kMaxLastDim = 10240;
        constexpr uint64_t kTileAlignment = 64;
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        if (ub_size == 0) {
            return ge::GRAPH_FAILED;
        }
        if (num_cores_aiv <= 0) {
            return ge::GRAPH_FAILED;
        }

        const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
        if (tensor_input_x == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const auto &shape = tensor_input_x->GetOriginShape();
        const size_t dim_num = shape.GetDimNum();
        if (dim_num == 0) {
            return ge::GRAPH_FAILED;
        }
        for (size_t i = 0; i < dim_num; ++i) {
            const int64_t dim = shape.GetDim(i);
            if (dim <= 0) {
                return ge::GRAPH_FAILED;
            }
        }
        const int64_t last_dim = shape.GetDim(dim_num - 1);
        if (last_dim < 1 || last_dim > kMaxLastDim) {
            return ge::GRAPH_FAILED;
        }

        const ge::DataType dtype_input_x = tensor_input_x->GetDataType();
        if (dtype_input_x != ge::DT_FLOAT16 && dtype_input_x != ge::DT_FLOAT) {
            return ge::GRAPH_FAILED;
        }
        const int dtype_size_input_x = ge::GetSizeByDataType(dtype_input_x);
        if (dtype_size_input_x <= 0) {
            return ge::GRAPH_FAILED;
        }

        uint64_t length_input_x = 1;
        for (size_t i = 0; i < dim_num; ++i) {
            const uint64_t dim = static_cast<uint64_t>(shape.GetDim(i));
            if (length_input_x > std::numeric_limits<uint64_t>::max() / dim) {
                return ge::GRAPH_FAILED;
            }
            length_input_x *= dim;
        }

        const uint64_t dtype_size = static_cast<uint64_t>(dtype_size_input_x);
        if (length_input_x > std::numeric_limits<uint64_t>::max() / dtype_size) {
            return ge::GRAPH_FAILED;
        }

        // Core selection is independent of UB tile size. Keep tiny inputs on one
        // core; expose parallelism for medium inputs instead of filling one UB.
        constexpr uint64_t kTargetBytesPerCore = 2048;
        const uint64_t grain = kTargetBytesPerCore / dtype_size;
        const uint64_t desired_cores = (length_input_x - 1) / grain + 1;
        const uint64_t block_count = desired_cores < static_cast<uint64_t>(num_cores_aiv)
            ? desired_cores : static_cast<uint64_t>(num_cores_aiv);
        const uint64_t align_elements = 32 / dtype_size;
        const uint64_t units = (length_input_x - 1) / align_elements + 1;
        const uint64_t max_core_elements = ((units - 1) / block_count + 1) * align_elements;

        constexpr uint64_t kReserveBytes = 1024;
        constexpr uint64_t kMaxTileElements = 8192;
        uint64_t tile_elements_u64 = max_core_elements < kMaxTileElements
            ? ((max_core_elements + kTileAlignment - 1) / kTileAlignment) * kTileAlignment
            : kMaxTileElements;
        uint32_t erf_tmp_bytes = 0;
        uint32_t buffer_count = 1;
        while (tile_elements_u64 >= kTileAlignment) {
            buffer_count = max_core_elements > tile_elements_u64 ? 2 : 1;
            // Both paths use fused GELU. No high-level Erf scratch is required.
            const uint64_t float_buffers = dtype_input_x == ge::DT_FLOAT ? 2 : 4;
            const uint64_t bytes_per_element =
                2 * buffer_count * dtype_size + float_buffers * sizeof(float);
            const uint64_t tmp_bytes = 0;
            const uint64_t required = tile_elements_u64 * bytes_per_element + tmp_bytes;
            if (ub_size > kReserveBytes && required <= ub_size - kReserveBytes) {
                erf_tmp_bytes = static_cast<uint32_t>(tmp_bytes);
                break;
            }
            if (ub_size <= kReserveBytes) {
                return ge::GRAPH_FAILED;
            }
            const uint64_t fused_bytes = 4 * dtype_size +
                (dtype_input_x == ge::DT_FLOAT ? 2 : 4) * sizeof(float);
            uint64_t next_tile = ((ub_size - kReserveBytes) / fused_bytes / kTileAlignment) * kTileAlignment;
            // Strictly decrease to select the multi-tile specialization on retry.
            if (next_tile >= tile_elements_u64) {
                next_tile = tile_elements_u64 - kTileAlignment;
            }
            tile_elements_u64 = next_tile;
        }
        if (tile_elements_u64 == 0) {
            return ge::GRAPH_FAILED;
        }

        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);
        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling->length = length_input_x;
        tiling->tile_elements = static_cast<uint32_t>(tile_elements_u64);
        tiling->block_count = static_cast<uint32_t>(block_count);
        tiling->erf_tmp_bytes = erf_tmp_bytes;
        tiling->buffer_count = buffer_count;
        tiling->core_elements = (units / block_count) * align_elements;
        tiling->extra_core_count = static_cast<uint32_t>(units % block_count);
        context->SetBlockDim(static_cast<uint32_t>(block_count));
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
        if (context == nullptr || context->GetInputShape(0) == nullptr ||
            context->GetOutputShape(0) == nullptr) {
            return GRAPH_FAILED;
        }
        *context->GetOutputShape(0) = *context->GetInputShape(0);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        if (context == nullptr) {
            return GRAPH_FAILED;
        }
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return GRAPH_SUCCESS;
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
