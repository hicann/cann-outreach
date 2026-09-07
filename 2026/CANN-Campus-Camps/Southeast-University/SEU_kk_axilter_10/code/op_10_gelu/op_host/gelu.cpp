// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include <algorithm>
#include <limits>
#include <vector>

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        if (context->GetPlatformInfo() == nullptr || context->GetInputShape(0) == nullptr ||
            context->GetInputDesc(0) == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        if (num_cores_aiv <= 0 || ub_size <= 4128) {
            return ge::GRAPH_FAILED;
        }
        const ge::DataType dtype_input_x = context->GetInputDesc(0)->GetDataType();
        if (dtype_input_x != ge::DT_FLOAT && dtype_input_x != ge::DT_FLOAT16) {
            return ge::GRAPH_FAILED;
        }
        const uint32_t typeBytes = dtype_input_x == ge::DT_FLOAT ? 4U : 2U;
        const auto &shape = context->GetInputShape(0)->GetStorageShape();
        uint64_t length = 1;
        for (size_t i = 0; i < shape.GetDimNum(); ++i) {
            const int64_t dim = shape.GetDim(i);
            if (dim <= 0 || length > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
                                        static_cast<uint64_t>(dim)) {
                return ge::GRAPH_FAILED;
            }
            length *= static_cast<uint64_t>(dim);
        }
        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        // Flatten ND; all core starts are aligned for both supported dtypes.
        const auto ceilDiv = [](uint64_t a, uint64_t b) { return a / b + (a % b != 0); };
        // Tune parallelism independently of the kernel: aim for 512 elements
        // per core until the hardware core limit is reached.
        constexpr uint32_t TARGET_ELEMENTS_PER_CORE = 512;
        uint32_t cores = static_cast<uint32_t>(std::min<uint64_t>(
            ceilDiv(length, TARGET_ELEMENTS_PER_CORE), num_cores_aiv));
        // FP32 CompareScalar consumes 64 elements per 256-byte repeat. A
        // 64-element block alignment is also sufficient for FP16 GM access.
        constexpr uint64_t BLOCK_ALIGN = 64;
        const uint64_t blockLength = ceilDiv(ceilDiv(length, cores), BLOCK_ALIGN) * BLOCK_ALIGN;
        cores = static_cast<uint32_t>(ceilDiv(length, blockLength));
        // FP16 uses the high-level GELU scratch buffer; FP32 uses the direct
        // fitted gate. Use the remaining UB for adaptive double buffering.
        uint32_t tileLength = static_cast<uint32_t>(std::min<uint64_t>(blockLength, 16384));
        uint32_t tmpBytes = 0;
        for (;;) {
            const uint32_t buffers = blockLength > tileLength ? 2U : 1U;
            if (dtype_input_x == ge::DT_FLOAT16) {
                uint32_t maxTmp = 0;
                uint32_t minTmp = 0;
                const ge::Shape tileShape(std::vector<int64_t>{static_cast<int64_t>(tileLength)});
                AscendC::GetGeluMaxMinTmpSize(tileShape, typeBytes, maxTmp, minTmp);
                const uint64_t minAligned =
                    (static_cast<uint64_t>(std::max<uint32_t>(minTmp, 32)) + 31) / 32 * 32;
                const uint64_t queueBytes =
                    static_cast<uint64_t>(tileLength) * 2 * buffers * typeBytes;
                if (queueBytes + minAligned <= ub_size - 4096) {
                    const uint64_t available = (ub_size - 4096 - queueBytes) / 32 * 32;
                    const uint64_t preferred =
                        (static_cast<uint64_t>(maxTmp) + 31) / 32 * 32;
                    tmpBytes = static_cast<uint32_t>(
                        std::max<uint64_t>(minAligned, std::min<uint64_t>(available, preferred)));
                    break;
                }
                if (tileLength <= BLOCK_ALIGN) {
                    return ge::GRAPH_FAILED;
                }
                tileLength = std::max<uint32_t>(
                    BLOCK_ALIGN, (tileLength / 2) / BLOCK_ALIGN * BLOCK_ALIGN);
                continue;
            }

            // FP32 fitted gate uses arg/value and one comparison bit per
            // element; the mask buffer is rounded to 32 bytes.
            constexpr uint32_t workBytesPerElement = 8;
            const uint64_t maskBytes = (static_cast<uint64_t>(tileLength) + 255) / 256 * 32;
            const uint64_t tensorBytes = static_cast<uint64_t>(tileLength) *
                                         (2 * buffers * typeBytes + workBytesPerElement) + maskBytes;
            if (tensorBytes <= ub_size - 4096) {
                break;
            }
            if (tileLength <= BLOCK_ALIGN) {
                return ge::GRAPH_FAILED;
            }
            // Mask costs at most tile/8 + 32 bytes. Jump close to the largest
            // feasible aligned tile, then let the loop recheck the exact size.
            const uint64_t available = ub_size - 4096 - 32;
            const uint64_t bytesTimesEight =
                static_cast<uint64_t>(2 * buffers * typeBytes + workBytesPerElement) * 8 + 1;
            uint64_t nextTile = available * 8 / bytesTimesEight;
            nextTile = std::min<uint64_t>(nextTile, tileLength - BLOCK_ALIGN) /
                       BLOCK_ALIGN * BLOCK_ALIGN;
            if (nextTile < BLOCK_ALIGN) {
                return ge::GRAPH_FAILED;
            }
            tileLength = static_cast<uint32_t>(nextTile);
        }
        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        tiling->length = length;
        tiling->blockLength = blockLength;
        tiling->tileLength = tileLength;
        tiling->erfTmpBytes = tmpBytes;
        // 配置启动核数
        context->SetBlockDim(cores);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        *context->GetOutputShape(0) = *context->GetInputShape(0);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
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
