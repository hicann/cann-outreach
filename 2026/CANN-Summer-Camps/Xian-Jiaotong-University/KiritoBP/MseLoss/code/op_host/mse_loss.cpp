// Host侧Tiling实现
#include "register/op_def_registry.h"

#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mse_loss_tiling.h"

#include "../op_kernel/tiling_key_mse_loss.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 获取算子输入数组信息
        const gert::Tensor *tensor_predict = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_label = context->GetRequiredInputTensor(1);
        ge::DataType dtype_predict = tensor_predict->GetDataType();
        uint32_t length_predict = tensor_predict->GetShapeSize();

        // 验证 predict 与 label shape 一致性
        uint32_t length_label = tensor_label->GetShapeSize();
        if (length_predict != length_label) {
            return ge::GRAPH_FAILED;
        }

        // 获取算子输入属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const char *attr_reduction = attrs->GetStr(0);

        // 配置tiling key, 从而实现kernel侧不同数据类型的区分
        uint32_t DT_PREDICT = static_cast<uint32_t>(dtype_predict);
        ASCENDC_TPL_SEL_PARAM(context, DT_PREDICT);

        // 计算tiling方案并填充tiling结构体
        MseLossTilingData *tiling = context->GetTilingData<MseLossTilingData>();
        tiling->length = length_predict;

        // 配置启动核数
        context->SetBlockDim(num_cores_aiv);

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
    class MseLoss : public OpDef {
    public:
        explicit MseLoss(const char *name) : OpDef(name) {
            this->Input("predict")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("label")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("reduction").AttrType(OPTIONAL).String("mean");
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(MseLoss);
}  // namespace ops
