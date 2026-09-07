// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    namespace {
        // UB缓冲预算份数: 输入/输出队列各双缓冲(2+2) + 4个fp32计算临时缓冲
        // + 1个half升精度缓冲(fp16输入) = 9份上限; fp32输入路径仅需2+2+4=8
        constexpr uint32_t UB_BUFFER_BUDGET = 9;
        // kernel内统一按fp32中间精度计算, 每元素按4字节做缓冲预算
        constexpr uint32_t CALC_ELEM_BYTES = 4;
    }  // namespace

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 获取算子输入信息
        const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_input_x = tensor_input_x->GetDataType(); // 获取数据类型
        int dtype_size_input_x = ge::GetSizeByDataType(dtype_input_x); // 获取数据类型的字长
        uint32_t length_input_x = static_cast<uint32_t>(tensor_input_x->GetShapeSize()); // 获取元素个数
        // 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        // 计算tiling方案并填充tiling结构体
        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        tiling->length = length_input_x;

        // 依据UB容量确定单次循环处理的元素数(32B对齐)
        // fp16输入也会在kernel内升到fp32计算, 因此按fp32元素预算统一计算分片大小
        uint32_t alignUnit = static_cast<uint32_t>(32 / dtype_size_input_x); // fp32为8, fp16为16
        uint32_t tileLength = alignUnit;
        if (length_input_x > 0 && ub_size > 0) {
            uint64_t elemPerBuf =
                ub_size / (UB_BUFFER_BUDGET * static_cast<uint64_t>(CALC_ELEM_BYTES));
            elemPerBuf = (elemPerBuf / alignUnit) * alignUnit;
            if (elemPerBuf > alignUnit) {
                tileLength = static_cast<uint32_t>(elemPerBuf);
            }
        }
        tiling->tileLength = tileLength;

        // 配置启动核数: 不超过AI Vector核数, 也不超过总元素数
        uint32_t blockDim = 1;
        if (length_input_x > 0) {
            uint32_t coreNum = num_cores_aiv > 0 ? static_cast<uint32_t>(num_cores_aiv) : 1U;
            blockDim = coreNum < length_input_x ? coreNum : length_input_x;
        }
        tiling->blockDim = blockDim;
        context->SetBlockDim(blockDim);

        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
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
