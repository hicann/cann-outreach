// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <limits>

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
        if (tensor_x == nullptr || tensor_y == nullptr || num_cores_aiv <= 0) {
            return ge::GRAPH_FAILED;
        }
        ge::DataType dtype_x = tensor_x->GetDataType(); // 获取数据类型
        if ((dtype_x != ge::DT_FLOAT && dtype_x != ge::DT_FLOAT16) ||
            tensor_y->GetDataType() != dtype_x ||
            tensor_x->GetShapeSize() != tensor_y->GetShapeSize()) {
            return ge::GRAPH_FAILED;
        }
        int dtype_size_x = ge::GetSizeByDataType(dtype_x); // 获取数据类型的字长
        const auto shape_size = tensor_x->GetShapeSize();
        if (shape_size < 0 || static_cast<uint64_t>(shape_size) >
            std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        const uint32_t length_x = static_cast<uint32_t>(shape_size);
        const uint32_t align_elements = 32 / dtype_size_x;
        const uint32_t total_blocks = static_cast<uint32_t>(
            (static_cast<uint64_t>(length_x) + align_elements - 1) / align_elements);
        const uint32_t core_count = std::max(1U, std::min(
            static_cast<uint32_t>(num_cores_aiv), total_blocks));
        // Three queues, each with two buffers. Reserve space for runtime use.
        constexpr uint64_t reserve_bytes = 1024;
        if (ub_size < reserve_bytes + 6 * 32) {
            return ge::GRAPH_FAILED;
        }
        const uint32_t tile_bytes = static_cast<uint32_t>(
            std::min<uint64_t>((ub_size - reserve_bytes) / 6 / 32 * 32, 8192));
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        // 示例: 计算tiling方案并填充tiling结构体
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->length = length_x;
        tiling->tileLength = tile_bytes / dtype_size_x;
        // 配置启动核数
        context->SetBlockDim(core_count);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const auto *x_shape = context->GetInputShape(0);
        auto *z_shape = context->GetOutputShape(0);
        if (x_shape == nullptr || z_shape == nullptr) {
            return GRAPH_FAILED;
        }
        *z_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
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
