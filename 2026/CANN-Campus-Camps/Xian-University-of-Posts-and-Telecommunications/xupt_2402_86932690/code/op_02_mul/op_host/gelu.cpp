// Host 侧 Tiling 实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    constexpr int64_t MAX_UB_FACTOR = 4096;  // 单 Tile 最多元素数（fp16 混合精度下 UB 仍安全）
    constexpr int64_t BLOCK_BYTES = 32;      // Ascend 数据搬运基本块
    constexpr uint32_t MAX_CORE_NUM = 16;    // 最多使用的核数

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());

        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();

        // 计算总元素数量
        const gert::StorageShape *x_shape = context->GetInputShape(0);
        int64_t totalNum = 1;
        for (int i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++) {
            totalNum *= x_shape->GetStorageShape().GetDim(i);
        }

        // 数据类型对应字节数：float32 = 4, float16 = 2
        int64_t typeSize = (DT_X == static_cast<uint32_t>(ge::DT_FLOAT16)) ? 2 : 4;
        int64_t alignNum = BLOCK_BYTES / typeSize;  // 32B 对齐所需元素数

        // 固定使用所有可用核（不塌缩），每个核处理连续的 base 块，最后一个核处理余数
        uint32_t usedCoreNum = totalNum < static_cast<int64_t>(MAX_CORE_NUM) ? static_cast<uint32_t>(totalNum) : MAX_CORE_NUM;
        if (usedCoreNum < 1) {
            usedCoreNum = 1;
        }

        // 每核基础块（向下对齐到 32B）
        int64_t blockLength = (totalNum / usedCoreNum / alignNum) * alignNum;
        if (blockLength < alignNum) {
            blockLength = alignNum;
        }
        // 保证最后一个核仍有数据可处理
        while (usedCoreNum > 1 && blockLength * (usedCoreNum - 1) >= totalNum) {
            --usedCoreNum;
            blockLength = (totalNum / usedCoreNum / alignNum) * alignNum;
            if (blockLength < alignNum) {
                blockLength = alignNum;
            }
        }

        // 负载均衡：余数 >= 1 个对齐块时，均摊给前 extraBlocks 个核，消除尾核长尾
        int64_t rem = totalNum - blockLength * usedCoreNum;
        int64_t extraBlocks = 0;
        int64_t lastBlockLength = 0;
        if (rem >= alignNum) {
            extraBlocks = rem / alignNum;                     // 前 extraBlocks 个核各多 1 块
            int64_t remTail = rem - extraBlocks * alignNum;   // 余下零头（< alignNum）
            lastBlockLength = blockLength + remTail;          // 第 extraBlocks 核的负载
        } else {
            extraBlocks = 0;
            lastBlockLength = totalNum - blockLength * (usedCoreNum - 1); // 尾核全收（零头 < alignNum）
        }

        // 单 Tile 上限，并保证 32B 对齐
        int64_t ubLength = blockLength < MAX_UB_FACTOR ? blockLength : MAX_UB_FACTOR;
        ubLength = (ubLength / alignNum) * alignNum;
        if (ubLength <= 0) {
            ubLength = alignNum;
        }
        if (ubLength > blockLength) {
            ubLength = blockLength;
        }

        tiling->totalNum = totalNum;
        tiling->usedCoreNum = usedCoreNum;
        tiling->blockLength = blockLength;
        tiling->lastBlockLength = lastBlockLength;
        tiling->ubLength = ubLength;
        tiling->alignNum = alignNum;
        tiling->extraBlocks = extraBlocks;

        context->SetBlockDim(usedCoreNum);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
} // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *z_shape = context->GetOutputShape(0);
        *z_shape = *x_shape;
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        auto x_dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, x_dtype);
        return ge::GRAPH_SUCCESS;
    }
} // namespace ge

namespace ops {
    class Gelu : public OpDef {
    public:
        explicit Gelu(const char *name) : OpDef(name) {
            this->Input("input_x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Gelu);
} // namespace ops
