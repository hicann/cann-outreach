/*!
 * \file add_custom_tiling.cpp
 * \brief add_custom 算子的 tiling（分块）策略实现
 *
 * 本文件提供 tiling 逻辑，将计算任务划分为较小的块，以便在 AI Core 上高效并行执行：
 *   1. 核切分：将元素总数均分到多个 AI Core（尽量多用核并行）
 *   2. UB 切分：按每核可用的 UB 大小计算单次处理块大小（考虑双缓冲共 6 块）
 *   3. 设置 tiling key（仅 float16 一种调度模式）
 */

#include <set>
#include <cstring>
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/add_custom_tiling_data.h"
#include "../op_kernel/add_custom_tiling_key.h"

namespace optiling {

// 矢量系统缓冲区大小为 0
constexpr uint32_t WS_SYS_SIZE = 0U;
// add_custom 要求输入输出为 2 维
constexpr int32_t DIMS_LIMIT = 2;

// 缓冲区数量常量：2 个输入 + 1 个输出，双缓冲共 6 块
constexpr int64_t BUFFER_NUM = 6;
// float16 类型大小（字节）
constexpr int64_t TYPE_SIZE = 2;

// add_custom 编译信息结构
struct AddCustomCompileInfo {};

/*!
 * \brief 从输入输出张量获取 shape 和属性信息
 *
 * 校验内容：
 *   - 输入 x、y 与输出 z 的 shape 维度必须为 2 维（如 [N2, N1]）
 *   - 数据类型仅支持 FLOAT16
 *   - 计算元素总数
 *
 * \param context 指向 tiling 上下文的指针
 * \param totalIdx 输出参数：要处理的元素总数
 * \param dataType 输出参数：输入张量的数据类型
 * \return 成功返回 ge::GRAPH_SUCCESS
 */
ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    // 获取输入 x 的 shape 信息
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);

    // 获取输入 y 的 shape 信息
    auto inputY = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputY);

    // 获取输出 z 的 shape 信息
    auto outZ = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outZ);

    // shape 校验：确保所有张量都是 2 维
    OP_CHECK_IF(inputX->GetDimNum() != DIMS_LIMIT || inputY->GetDimNum() != DIMS_LIMIT ||
                    outZ->GetDimNum() != DIMS_LIMIT,
                OP_LOGE(context,
                        "AddCustom: inputx, inputy, outputz shape dim = %zu, %zu, %zu, should be equal %d",
                        inputX->GetDimNum(), inputY->GetDimNum(), outZ->GetDimNum(), DIMS_LIMIT),
                return ge::GRAPH_FAILED);

    // 元素总数（2 维 [N2, N1] 一维化）
    totalIdx = inputX->GetShapeSize();

    // 数据类型校验：仅支持 float16
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();

    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "AddCustom: invalid dtype, only FLOAT16 is supported");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

/*!
 * \brief 获取算子执行所需的工作空间大小
 *
 * add_custom 不需要额外工作空间，大小设置为 0。
 */
ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

/*!
 * \brief add_custom 算子的主 tiling 函数
 *
 * 计算最优 tiling 策略：
 *   1. 检索平台信息（UB 大小、AI Core 数量）
 *   2. 检索 shape 和属性信息（校验 2 维、float16）
 *   3. Core tiling：将工作划分到多个 AI Core 并行计算
 *   4. UB tiling：根据每核可用 UB 内存计算单次处理块大小
 *   5. 设置 tiling key（float16 调度模式）
 *
 * \param context 指向 tiling 上下文的指针
 * \return 成功计算 tiling 策略返回 ge::GRAPH_SUCCESS
 */
static ge::graphStatus AddCustomTilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行时信息
    uint64_t ubSize;
    int64_t coreNum;
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    // 2、获取 shape、属性信息
    int64_t totalIdx;
    ge::DataType dataType;
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3、获取 WorkspaceSize 信息
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    // 4、设置 tiling 信息
    AddCustomTilingData* tiling = context->GetTilingData<AddCustomTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(AddCustomTilingData), 0, sizeof(AddCustomTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 优先做核切分，尽量用更多的核并行计算
    tiling->totalNum = totalIdx;
    tiling->blockFactor = Ops::Base::CeilDiv(totalIdx, coreNum);
    int64_t usedCoreNum = Ops::Base::CeilDiv(totalIdx, tiling->blockFactor);

    // 计算 UB 切分：2 输入 + 1 输出、双缓冲共 6 块 UB tensor，按 MTE 对齐要求取整
    int64_t ubCanUse = static_cast<int64_t>(ubSize);
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    tiling->ubFactor = Ops::Base::FloorAlign(Ops::Base::FloorDiv((ubCanUse / TYPE_SIZE), BUFFER_NUM), ubBlockSize);

    // 设置使用的 AI Core 数量
    context->SetBlockDim(usedCoreNum);

    // 5、根据数据类型设置 tiling key（float16 -> mode 0）
    if (dataType == ge::DT_FLOAT16) {
        uint64_t tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
        context->SetTilingKey(tilingKey);
    } else {
        OP_LOGE(context, "AddCustom: get dtype error");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

/*!
 * \brief 解析 add_custom 算子的 tiling 信息
 *
 * 对于 add_custom 算子，不需要静态 tiling，直接返回成功。
 */
static ge::graphStatus TilingParseForAddCustom([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling 注册入口：将 tiling 函数、tiling 解析函数和编译信息注册到系统中
IMPL_OP_OPTILING(AddCustom).Tiling(AddCustomTilingFunc).TilingParse<AddCustomCompileInfo>(TilingParseForAddCustom);
} // namespace optiling
