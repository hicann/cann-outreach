// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    const uint32_t BUFFER_NUM = 1;
    const uint32_t UB_RESERVE = 4 * 1024;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 1. 获取平台信息（AI Core 向量核数量、UB 大小）
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t num_cores_aiv = static_cast<uint32_t>(platform.GetCoreNumAiv());
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        if (ub_size == 0) {
            ub_size = 192 * 1024;
        }

        // 2. 获取算子输入信息（shape、dtype）
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t dtype_size = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_x));
        if (dtype_size == 0) {
            dtype_size = 1;
        }
        uint32_t total_length = tensor_x->GetShapeSize();
        (void)tensor_y;

        // 3. 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 4. 计算tiling方案并填充tiling结构体
        uint32_t block_dim = (num_cores_aiv == 0) ? 1 : num_cores_aiv;
        if (block_dim > total_length && total_length > 0) {
            block_dim = total_length;
        }
        const uint32_t BLOCK_DIM_CAP = 8;
        if (block_dim > BLOCK_DIM_CAP) {
            block_dim = BLOCK_DIM_CAP;
        }
        uint32_t elem_per_block = 32 / dtype_size;
        uint32_t block_length = (total_length + block_dim - 1) / block_dim;
        block_length = (block_length + elem_per_block - 1) / elem_per_block * elem_per_block;
        if (block_length == 0) {
            block_length = elem_per_block;
        }
        uint64_t ub_available = (ub_size > UB_RESERVE) ? (ub_size - UB_RESERVE) : (ub_size / 2);
        uint32_t ub_max_elem = static_cast<uint32_t>(ub_available / (dtype_size * 3 * BUFFER_NUM));
        ub_max_elem = ub_max_elem / elem_per_block * elem_per_block;
        uint32_t tile_length = block_length;
        if (tile_length > ub_max_elem && ub_max_elem >= elem_per_block) {
            tile_length = ub_max_elem;
        }
        if (tile_length < elem_per_block) {
            tile_length = elem_per_block;
        }
        uint32_t tile_num = (block_length + tile_length - 1) / tile_length;

        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->totalLength = total_length;
        tiling->blockLength = block_length;
        tiling->tileNum = tile_num;
        tiling->tileLength = tile_length;

        // 5. 配置启动核数
        context->SetBlockDim(block_dim);
        // 6. 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape* x_shape = context->GetInputShape(0);
        gert::Shape* z_shape = context->GetOutputShape(0);
        *z_shape = *x_shape;
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
}  // namespace ops
