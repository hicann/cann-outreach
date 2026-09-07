// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 获取输入张量信息
        const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_input_x = tensor_input_x->GetDataType();
        int dtype_size = ge::GetSizeByDataType(dtype_input_x);
        uint32_t totalLength = tensor_input_x->GetShapeSize();

        // 配置tiling key, 实现kernel侧不同数据类型/算法的区分
        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        // 32字节对齐对应的元素个数
        uint32_t ALIGN_NUM = 32 / dtype_size;

        // 每个tile的元素个数: in/out 队列(双缓冲)按UB一半预算
        const uint32_t BUFFER_NUM = 2;
        uint32_t perElem = BUFFER_NUM * dtype_size * 2;   // in+out 队列
        uint32_t blockSize = static_cast<uint32_t>((ub_size / 2) / perElem);
        blockSize = (blockSize / ALIGN_NUM) * ALIGN_NUM;
        const uint32_t MAX_BLOCK = 16384;
        if (blockSize > MAX_BLOCK) {
            blockSize = MAX_BLOCK;
        }
        if (blockSize < ALIGN_NUM) {
            blockSize = ALIGN_NUM;
        }

        // 启动核数: 每个核至少处理MIN_CHUNK个元素, 中等数据量也能用满AIV核
        const uint32_t MIN_CHUNK = 512;
        uint32_t needed = (totalLength + MIN_CHUNK - 1) / MIN_CHUNK;
        uint32_t aivNum = static_cast<uint32_t>(num_cores_aiv);
        if (aivNum > needed) {
            aivNum = needed;
        }
        if (aivNum == 0) {
            aivNum = 1;
        }

        // 按核均匀分配(按ALIGN_NUM对齐), 余数拆成ALIGN_NUM块分给前几个核, 保证负载均衡
        uint32_t coreSize = (totalLength / aivNum / ALIGN_NUM) * ALIGN_NUM;
        uint32_t coreRemain = totalLength - aivNum * coreSize;
        uint32_t extraCores = coreRemain / ALIGN_NUM;
        uint32_t lastRemainder = coreRemain % ALIGN_NUM;

        // 填充tiling结构体
        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        tiling->totalLength = totalLength;
        tiling->ALIGN_NUM = ALIGN_NUM;
        tiling->blockSize = blockSize;
        tiling->coreSize = coreSize;
        tiling->extraCores = extraCores;
        tiling->lastRemainder = lastRemainder;

        // 配置启动核数
        context->SetBlockDim(aivNum);

        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *input_shape = context->GetInputShape(0);
        gert::Shape *output_shape = context->GetOutputShape(0);
        *output_shape = *input_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
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