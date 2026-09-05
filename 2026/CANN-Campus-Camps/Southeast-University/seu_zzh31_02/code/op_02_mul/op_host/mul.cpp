// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    // 32字节对齐 256 的辅助函数
    static inline uint32_t AlignUp(uint64_t value, uint32_t factor) {
        return static_cast<uint32_t>((value + factor - 1) / factor * factor);
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 获取算子输入数组信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType(); // 获取数据类型
        int dtype_size_x = ge::GetSizeByDataType(dtype_x); // 获取数据类型的字长
        uint32_t total_length = tensor_x->GetShapeSize(); // 获取元素个数

        // 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // ---- 常量 ----
        const uint32_t MIN_TILING_BITS_SIZE_PER_CORE = 32768; // 每核最小数据量：4KB = 32768 bit
        const uint32_t ELEM_ALIGN_FACTOR = 512;               // blockLength 512 元素对齐
        const uint32_t ALIGN_256 = 256;                       // tileLength 256B 对齐
        const uint32_t NUM_BUFFERS = 3;                       // x/y/z 三队列

        uint32_t min_dtype_bits = static_cast<uint32_t>(dtype_size_x) * 8; // fp32=32 fp16=16

        // ---- coreNum = min(ceil(totalLength*minDtypeBits/MIN_TILING_BITS_SIZE_PER_CORE), availableCoreNum)，且>=1 ----
        uint32_t core_num = 1;
        if (total_length > 0) {
            uint64_t required_bits = static_cast<uint64_t>(total_length) * min_dtype_bits;
            uint64_t expected = (required_bits + MIN_TILING_BITS_SIZE_PER_CORE - 1) / MIN_TILING_BITS_SIZE_PER_CORE;
            if (expected > static_cast<uint64_t>(num_cores_aiv)) {
                expected = static_cast<uint64_t>(num_cores_aiv);
            }
            if (expected > 1) {
                core_num = static_cast<uint32_t>(expected);
            }
        }

        // ---- blockLength = align512(ceil(totalLength / coreNum)) ----
        uint32_t block_length = 0;
        uint32_t per_core = (total_length + core_num - 1) / core_num;
        if (per_core > 0) {
            block_length = AlignUp(per_core, ELEM_ALIGN_FACTOR);
        }

        // ---- blockNum = blockLength==0 ? 1 : ceil(totalLength / blockLength)（避免除0） ----
        uint32_t block_num = 1;
        if (block_length > 0) {
            block_num = (total_length + block_length - 1) / block_length;
            if (block_num == 0) {
                block_num = 1;
            }
        }

        // ---- tileLength = align256B(ubSize / (numBuffers * elemBytes)) ----
        uint32_t max_elem_num = static_cast<uint32_t>(ub_size / (NUM_BUFFERS * static_cast<uint32_t>(dtype_size_x)));
        uint32_t align_elem = ALIGN_256 / static_cast<uint32_t>(dtype_size_x);
        uint32_t tile_length = 0;
        if (align_elem > 0) {
            tile_length = (max_elem_num / align_elem) * align_elem;
        }
        if (tile_length == 0) {
            tile_length = align_elem; // 保底，避免除0
        }

        // 填充tiling结构体
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->totalLength = total_length;
        tiling->blockNum = block_num;
        tiling->blockLength = block_length;
        tiling->tileLength = tile_length;

        // 配置启动核数
        context->SetBlockDim(block_num);
        // 配置workspace大小（mul纯逐元素，无需workspace）
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // z.shape = x.shape
        const gert::Shape *input_shape = context->GetInputShape(0);
        gert::Shape *output_shape = context->GetOutputShape(0);
        *output_shape = *input_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // z.dtype = x.dtype
        ge::DataType input_type = context->GetInputDataType(0);
        context->SetOutputDataType(0, input_type);
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