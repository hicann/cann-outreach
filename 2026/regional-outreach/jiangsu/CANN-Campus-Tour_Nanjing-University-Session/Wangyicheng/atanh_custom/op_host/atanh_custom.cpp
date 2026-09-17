/*
 * @file atanh_custom.cpp
 * @brief atanh 算子 Host 侧实现：算子原型注册 + Tiling 分片策略
 *
 * 算子规格：
 *  - 输入：x，1 个输入，4 维 ND 张量 [N4, N3, N2, N1]
 *  - 输出：y，1 个输出，与输入同 shape
 *  - 数据类型：float16
 *  - 数据格式：FORMAT_ND
 *  - 计算：y = atanh(x) = 0.5 * ln((1 + x) / (1 - x))
 */

#include "atanh_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

// 双缓冲数量：一边搬运数据一边计算，隐藏拷贝延迟
constexpr uint32_t BUFFER_NUM = 2;

// tile 对齐长度：half 类型 DataCopy 要求 32 字节对齐（16 个 half）。
// 这里对齐到 256（512 字节），兼顾 MTE 搬运效率。
constexpr uint32_t ALIGN_LENGTH = 256;

/*
 * TilingFunc：Host 侧分片函数，决定如何把总数据切分到多个核、多个 tile 上。
 * 关键产出：
 *   1. blockDim：参与计算的核数
 *   2. totalLength / blockLength / tileLength：分片参数，传给 Kernel
 */
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AtanhCustomTilingData tiling;

    // 1. 获取平台信息：AIV（Vector）核数、UB（Unified Buffer）大小
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aivNum == 0) {
        aivNum = 1;
    }

    // 2. 获取输入 shape（4 维 ND：[N4, N3, N2, N1]），逐元素操作只需总长度
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    uint32_t totalLength = 1;
    for (int32_t i = 0; i < inputShape->GetStorageShape().GetDimNum(); ++i) {
        totalLength *= static_cast<uint32_t>(inputShape->GetStorageShape().GetDim(i));
    }

    // 3. 每个核的基准处理长度（向上取整，最后一个核由 Kernel 侧修正）
    uint32_t blockLength = (totalLength + aivNum - 1) / aivNum;

    // 4. 根据 UB 大小反推单个 tile 的最大长度。
    //    每个 tile 的 UB 开销（含双缓冲 BUFFER_NUM=2）：
    //      输入 half   : tileLength * 2 字节
    //      输出 half   : tileLength * 2 字节
    //      中间 float×3: tileLength * 4 字节 * 3
    //    单元素开销 = BUFFER_NUM * (2 + 2 + 3 * 4) = 32 字节
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t perElemBytes = static_cast<uint64_t>(BUFFER_NUM) *
                            (2 * sizeof(uint16_t) + 2 * sizeof(uint16_t) + 3 * sizeof(float));
    // 留 20% 余量给框架与对齐开销
    uint32_t maxTileLength = static_cast<uint32_t>((ubSize * 8 / 10) / perElemBytes);
    maxTileLength = (maxTileLength / ALIGN_LENGTH) * ALIGN_LENGTH;
    if (maxTileLength == 0) {
        maxTileLength = ALIGN_LENGTH;
    }

    // 5. 确定 tile 长度：取 blockLength 与 maxTileLength 的较小值，再向上对齐到 256
    uint32_t tileLength = (blockLength < maxTileLength) ? blockLength : maxTileLength;
    tileLength = ((tileLength + ALIGN_LENGTH - 1) / ALIGN_LENGTH) * ALIGN_LENGTH;

    tiling.set_totalLength(totalLength);
    tiling.set_blockLength(blockLength);
    tiling.set_tileLength(tileLength);

    // 6. 设置核数与 tiling 数据
    context->SetBlockDim(aivNum);
    context->SetTilingData(&tiling, sizeof(tiling));

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {

class AtanhCustom : public OpDef {
public:
    explicit AtanhCustom(const char* name) : OpDef(name)
    {
        // 输入 x：必选，float16，ND 格式
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // 输出 y：必选，float16，ND 格式
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // shape / dtype 推导：逐元素算子，输出与输入保持一致
        // （单算子 API 调用模式下可省略，工程化开发建议保留）
        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        // 绑定 Tiling 实现
        this->AICore().SetTiling(optiling::TilingFunc);

        // 注册支持的芯片型号，请按实际硬件替换：
        //   ascend310 / ascend310p / ascend910 / ascend910b / ascend910c ...
        this->AICore().AddConfig("ascend910b");
    }
};

// 注册算子类型（算子类型名大驼峰：AtanhCustom）
OP_ADD(AtanhCustom);

}  // namespace ops
