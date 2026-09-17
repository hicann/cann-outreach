// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    // 单块最大元素个数（float 时约 32KB，兼容 UB 容量）
    const uint32_t MAX_TILE_LENGTH = 8192;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取算子输入数组信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();

        // 计算总元素个数
        auto shape_x = context->GetInputShape(0)->GetStorageShape();
        uint32_t totalLength = static_cast<uint32_t>(shape_x.GetShapeSize());

        // 根据数据类型计算 32 字节对齐的元素个数
        uint32_t dataTypeSize = (DT_X == ge::DT_FLOAT16) ? 2u : 4u;
        uint32_t alignNum = 32u / dataTypeSize;  // float: 8, half: 16

        uint32_t tileNum = 1;
        uint32_t tileLength = totalLength;
        uint32_t lastTileLength = totalLength;

        if (totalLength == 0) {
            // 空 Tensor，直接返回
            tileNum = 1;
            tileLength = 0;
            lastTileLength = 0;
        } else if (totalLength <= MAX_TILE_LENGTH) {
            // 小数据：单块处理，避免多次循环开销
            tileNum = 1;
            tileLength = totalLength;
            lastTileLength = totalLength;
        } else {
            // 大数据：按 MAX_TILE_LENGTH 分块，并对齐到 32 字节
            tileNum = (totalLength + MAX_TILE_LENGTH - 1) / MAX_TILE_LENGTH;
            tileLength = (totalLength + tileNum - 1) / tileNum;
            // 向上对齐
            tileLength = ((tileLength + alignNum - 1) / alignNum) * alignNum;
            // 对齐后重新计算分块数
            tileNum = (totalLength + tileLength - 1) / tileLength;
            lastTileLength = totalLength - (tileNum - 1) * tileLength;
        }

        tiling->totalLength = totalLength;
        tiling->tileNum = tileNum;
        tiling->tileLength = tileLength;
        tiling->lastTileLength = lastTileLength;

        // 单核处理
        context->SetBlockDim(1);

        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *zShape = context->GetOutputShape(0);
        // 输出 shape 与输入 x 一致（x 与 y 要求同 shape）
        *zShape = *xShape;
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        auto dtX = context->GetInputDataType(0);
        context->SetOutputDataType(0, dtX);
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