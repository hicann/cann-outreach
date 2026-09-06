/*!
 * \file relu_tiling.cpp
 * \brief Relu 绠楀瓙 Tiling 瀹炵幇
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
// 涓?op_kernel/relu.h 涓?BUFFER_NUM 淇濇寔涓€鑷达紙鍙岀紦鍐诧級
constexpr int64_t HOST_BUFFER_NUM = 2;
// UB 棰勭畻瀹夊叏浣欓噺锛氬疄闄呭彲鐢ㄩ噺鎸?90% 璁★紝闃叉 InitBuffer 瑙﹂《
constexpr int64_t UB_SAFE_RATIO_NUM = 9;
constexpr int64_t UB_SAFE_RATIO_DEN = 10;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 骞冲彴淇℃伅锛欰IV 鏍告暟涓庡崟鏍?UB 瀹归噺
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 杈撳叆淇℃伅锛歞type 鍐冲畾鍏冪礌瀛楅暱涓庡榻愬厓绱犳暟锛宻hape 鍐冲畾鍒囧垎
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    // 鏈畻瀛愪粎娉ㄥ唽 DT_FLOAT / DT_FLOAT16锛堣 relu_def.cpp锛夛紝瀛楅暱鐩存帴鎸?dtype 鍒ゆ柇
    int64_t typeSize = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) ? 2 : 4;
    OP_CHECK_IF(typeSize <= 0, OP_LOGE(context, "invalid dtype size"), return ge::GRAPH_FAILED);
    const gert::Tensor* xTensor = context->GetRequiredInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xTensor);
    int64_t totalNum = static_cast<int64_t>(xTensor->GetShapeSize());

    uint32_t ubBlockSize = GetUbBlockSize(context);        // 32B锛欴ataCopy 鏈€灏忓榻愬潡锛堟ā鏉垮嚱鏁伴渶浼?context锛?
    int64_t alignElems = static_cast<int64_t>(ubBlockSize) / typeSize;  // fp16=16, fp32=8

    if (totalNum <= 0) {
        // 绌鸿緭鍏ラ槻寰★細鐩存帴浠ユ渶灏忛厤缃繑鍥?
        tiling->totalNum = 0;
        tiling->blockFactor = alignElems;
        tiling->ubFactor = alignElems;
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
        return ge::GRAPH_SUCCESS;
    }

    // ---- 鏍搁棿鍒囧垎 ----
    // blockFactor = ceil(totalNum / coreNum) 鍚戜笂瀵归綈鍒?32B 鍏冪礌鏁帮細
    //   淇濊瘉姣忔牳 GM 绐楀彛璧峰鍦板潃 (blockIdx * blockFactor) 濮嬬粓 32B 瀵归綈锛?
    //   涓斿熬鏍哥殑鍓╀綑閲?(totalNum - start) 涔熸槸 alignElems 鐨勫€嶆暟锛孋opyIn/CopyOut 鏃犻渶 mask銆?
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignElems);

    // ---- 鏍稿唴鍒囧垎 ----
    // ubFactor 鍙?UB 棰勭畻绾︽潫锛? 涓槦鍒?x/y) 脳 BUFFER_NUM 浠藉弻缂撳啿 脳 鍏冪礌瀛楅暱 鈮?ubSize锛?
    // 鍙︾暀 10% 瀹夊叏浣欓噺锛岄槻姝?InitBuffer 瑙﹂《銆?
    int64_t ubFactor = static_cast<int64_t>(ubSize) / (2 * HOST_BUFFER_NUM * typeSize);
    ubFactor = ubFactor / UB_SAFE_RATIO_NUM * UB_SAFE_RATIO_DEN;
    ubFactor = FloorAlign(ubFactor, alignElems);
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;   // 鍗曟牳鏁版嵁涓€杞寰椾笅灏变笉蹇呭杞?
    }
    if (ubFactor < alignElems) {
        ubFactor = alignElems;    // 鏋佸皬杈撳叆鐨勫厹搴曚笅闄?
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 鍚姩鏍告暟锛氭伆濂借鐩栨暟鎹墍闇€鐨勬牳鏁帮紝閬垮厤绌鸿浆鏍?
    int64_t usedCores = CeilDiv(totalNum, blockFactor);
    context->SetBlockDim(static_cast<uint32_t>(usedCores));

    // 鏍规嵁杈撳叆 dtype 閫夋嫨 tilingKey
    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling