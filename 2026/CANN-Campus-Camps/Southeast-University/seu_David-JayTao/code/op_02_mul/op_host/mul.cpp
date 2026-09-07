// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        // ============================================================
        // 1. 获取硬件信息
        // ============================================================
        auto platform =
            platform_ascendc::PlatformAscendC(
                context->GetPlatformInfo());

        int32_t numCoresAiv = platform.GetCoreNumAiv();

        // ============================================================
        // 2. 获取输入 Tensor 信息
        // ============================================================
        const gert::Tensor *tensorX =
            context->GetRequiredInputTensor(0);

        ge::DataType dtypeX = tensorX->GetDataType();

        uint32_t totalLength =
            static_cast<uint32_t>(tensorX->GetShapeSize());

        // ============================================================
        // 3. 配置 Tiling Key
        //    让 Kernel 根据 float16 / float32 生成不同模板实例
        // ============================================================
        uint32_t DT_X =
            static_cast<uint32_t>(dtypeX);

        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // ============================================================
        // 4. 本题固定使用最多 8 个 Vector Core
        // ============================================================
        uint32_t blockDim =
            static_cast<uint32_t>(numCoresAiv >= 8 ? 8 : numCoresAiv);

        // 本题测试环境正常情况下 >= 8 Core
        // shape = (8, 2048)
        // totalLength = 16384
        constexpr uint32_t TILE_NUM = 8;
        constexpr uint32_t BUFFER_NUM = 2;

        uint32_t blockLength =
            totalLength / blockDim;

        uint32_t tileLength =
            blockLength / TILE_NUM / BUFFER_NUM;

        // ============================================================
        // 5. 填写 TilingData
        // ============================================================
        MulTilingData *tiling =
            context->GetTilingData<MulTilingData>();

        tiling->totalLength = totalLength;
        tiling->blockLength = blockLength;
        tiling->tileNum = TILE_NUM;
        tiling->tileLength = tileLength;

        // ============================================================
        // 6. 告诉 runtime 启动多少个 Core
        // ============================================================
        context->SetBlockDim(blockDim);

        // ============================================================
        // 7. 本题不需要额外 workspace
        // ============================================================
        size_t *currentWorkspace =
            context->GetWorkspaceSizes(1);

        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
static graphStatus InferShape(
    gert::InferShapeContext *context)
    {
        const gert::Shape *xShape =
            context->GetInputShape(0);

        gert::Shape *zShape =
            context->GetOutputShape(0);

        auto dimNum = xShape->GetDimNum();

        zShape->SetDimNum(dimNum);

        for (size_t i = 0; i < dimNum; ++i) {
            zShape->SetDim(i, xShape->GetDim(i));
        }

        return ge::GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(
        gert::InferDataTypeContext *context)
    {
        ge::DataType dtype =
            context->GetInputDataType(0);

        context->SetOutputDataType(0, dtype);

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
