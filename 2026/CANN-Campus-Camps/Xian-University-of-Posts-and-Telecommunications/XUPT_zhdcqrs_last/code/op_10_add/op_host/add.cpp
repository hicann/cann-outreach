// // Host侧Tiling实现
// #include "register/op_def_registry.h"
// #include "tiling/platform/platform_ascendc.h"

// #include "../op_kernel/add_tiling.h"
// #include "../op_kernel/tiling_key_add.h"

// namespace optiling {
//     static ge::graphStatus TilingFunc(gert::TilingContext *context) {
//         // 示例: 获取平台信息
//         auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
//         int32_t num_cores_aiv = platform.GetCoreNumAiv();
//         uint64_t ub_size;
//         platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
//         // 示例: 获取算子输入数组信息
//         const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
//         const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
//         ge::DataType dtype_x = tensor_x->GetDataType(); // 获取数据类型
//         int dtype_size_x = ge::GetSizeByDataType(dtype_x); // 获取数据类型的字长
//         uint32_t length_x = tensor_x->GetShapeSize(); // 获取元素个数
//         uint32_t size_x = tensor_x->GetSize(); // 获取内存大小
//         // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
//         uint32_t DT_X = static_cast<uint32_t>(dtype_x);
//         ASCENDC_TPL_SEL_PARAM(context, DT_X);
//         // 示例: 计算tiling方案并填充tiling结构体
//         AddTilingData *tiling = context->GetTilingData<AddTilingData>();
//         tiling->length = length_x;
//         // 配置启动核数
//         context->SetBlockDim(num_cores_aiv);
//         // 配置workspace大小
//         size_t *currentWorkspace = context->GetWorkspaceSizes(1);
//         currentWorkspace[0] = 0;
//         return ge::GRAPH_SUCCESS;
//     }
// }  // namespace optiling

// namespace ge {
//     static graphStatus InferShape(gert::InferShapeContext *context) {
//         return GRAPH_SUCCESS;
//     }
//     static graphStatus InferDataType(gert::InferDataTypeContext *context) {
//         return ge::GRAPH_SUCCESS;
//     }
// }  // namespace ge

// namespace ops {
//     class Add : public OpDef {
//     public:
//         explicit Add(const char *name) : OpDef(name) {
//             this->Input("x")
//                 .ParamType(REQUIRED)
//                 .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
//                 .Format({ge::FORMAT_ND, ge::FORMAT_ND});
//             this->Input("y")
//                 .ParamType(REQUIRED)
//                 .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
//                 .Format({ge::FORMAT_ND, ge::FORMAT_ND});
//             this->Output("z")
//                 .ParamType(REQUIRED)
//                 .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
//                 .Format({ge::FORMAT_ND, ge::FORMAT_ND});
//             this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
//             this->AICore()
//                 .SetTiling(optiling::TilingFunc)
//                 .AddConfig("ascend910b");
//         }
//     };
//     OP_ADD(Add);
// }  // namespace ops
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/add_tiling.h"
#include "../op_kernel/tiling_key_add.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 1. 获取昇腾平台信息
    auto platform =
        platform_ascendc::PlatformAscendC(
            context->GetPlatformInfo());

    int32_t numCoresAiv = platform.GetCoreNumAiv();

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    if (numCoresAiv <= 0) {
        numCoresAiv = 1;
    }

    // 2. 获取输入 x
    const gert::Tensor* inputX =
        context->GetRequiredInputTensor(0);

    if (inputX == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType inputDtype =
        inputX->GetDataType();

    int32_t dtypeSize =
        ge::GetSizeByDataType(inputDtype);

    if (dtypeSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t totalLength =
        static_cast<uint32_t>(
            inputX->GetShapeSize());

    // 3. 根据输入类型设置 Tiling Key
    uint32_t DT_INPUT_X =
        static_cast<uint32_t>(inputDtype);

    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    // 4. 计算32字节对齐对应的元素数量
    int64_t alignElements = 32 / dtypeSize;

    if (alignElements <= 0) {
        alignElements = 1;
    }

    // 5. 计算每个核处理的元素数
    int64_t blockFactor =
        (static_cast<int64_t>(totalLength)
         + numCoresAiv - 1)
        / numCoresAiv;

    // 向上进行32字节对齐
    blockFactor =
        (blockFactor + alignElements - 1)
        / alignElements
        * alignElements;

    if (blockFactor < alignElements) {
        blockFactor = alignElements;
    }

    // 6. 计算实际使用核数
    int64_t usedCoreNum =
        (static_cast<int64_t>(totalLength)
         + blockFactor - 1)
        / blockFactor;

    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    if (usedCoreNum > numCoresAiv) {
        usedCoreNum = numCoresAiv;
    }

    /*
     * 7. 计算 UB 单次处理量
     *
     * 预留8KB系统空间。
     * Add需要三块UB空间：
     * 1份输入x + 1份输入y + 1份输出z。
     */
    uint64_t usableUbSize =
        (ubSize > 8192)
            ? (ubSize - 8192)
            : ubSize;

    int64_t oneBufferElements =
        static_cast<int64_t>(usableUbSize)
        / dtypeSize
        / 3;

    int64_t ubFactor =
        oneBufferElements
        / alignElements
        * alignElements;

    if (ubFactor <= 0) {
        ubFactor = alignElements;
    }

    /*
     * 单次处理量不需要超过每核总数据量，
     * 这样可以减少不必要的UB申请。
     */
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    // 8. 写入 TilingData
    AddTilingData* tiling =
        context->GetTilingData<AddTilingData>();

    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tiling->length = totalLength;
    tiling->blockFactor =
        static_cast<uint32_t>(blockFactor);
    tiling->ubFactor =
        static_cast<uint32_t>(ubFactor);

    // 9. 设置启动核数
    context->SetBlockDim(
        static_cast<uint32_t>(usedCoreNum));

    // 10. 本算子不需要workspace
    size_t* workspace =
        context->GetWorkspaceSizes(1);

    if (workspace != nullptr) {
        workspace[0] = 0;
    }

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(
    gert::InferShapeContext* context)
{
    const gert::Shape* inputShape =
        context->GetInputShape(0);

    gert::Shape* outputShape =
        context->GetOutputShape(0);

    if (inputShape == nullptr ||
        outputShape == nullptr) {
        return GRAPH_FAILED;
    }

    // 输出z的形状与输入x一致
    *outputShape = *inputShape;

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(
    gert::InferDataTypeContext* context)
{
    // 输出z的数据类型与输入x一致
    ge::DataType inputDtype =
        context->GetInputDataType(0);

    context->SetOutputDataType(
        0,
        inputDtype);

    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Add : public OpDef {
public:
    explicit Add(const char* name)
        : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(
                ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Add);

}  // namespace ops