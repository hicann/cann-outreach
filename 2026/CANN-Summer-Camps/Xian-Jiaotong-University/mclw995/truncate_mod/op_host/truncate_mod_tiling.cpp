/*!
 * \file truncate_mod_tiling.cpp
 * \brief TruncateMod 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/truncate_mod_tiling_data.h"
#include "../op_kernel/truncate_mod_tiling_key.h"
#include <algorithm>
#include <vector>
#include <set>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

static std::vector<int64_t> ComputeBroadcastShape(
    const std::vector<int64_t>& s1,
    const std::vector<int64_t>& s2)
{
    int64_t d1 = static_cast<int64_t>(s1.size());
    int64_t d2 = static_cast<int64_t>(s2.size());
    int64_t rd = std::max(d1, d2);
    std::vector<int64_t> out(rd, 1);
    for (int64_t i = 0; i < rd; ++i) {
        int64_t v1 = (i < rd - d1) ? 1 : s1[static_cast<size_t>(i - (rd - d1))];
        int64_t v2 = (i < rd - d2) ? 1 : s2[static_cast<size_t>(i - (rd - d2))];
        out[i] = std::max(v1, v2);
    }
    return out;
}

static int64_t ShapeSize(const std::vector<int64_t>& shape)
{
    int64_t s = 1;
    for (auto v : shape) s *= v;
    return s;
}

static std::vector<int64_t> ShapeToVec(const gert::Shape& s)
{
    std::vector<int64_t> v(s.GetDimNum());
    for (size_t i = 0; i < s.GetDimNum(); ++i) {
        v[i] = s.GetDim(i);
    }
    return v;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* pi = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, pi);
    auto ap = platform_ascendc::PlatformAscendC(pi);
    *coreNum = ap.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static uint32_t GeDtToAscendCDt(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT:   return static_cast<uint32_t>(C_DT_FLOAT);
        case ge::DT_FLOAT16: return static_cast<uint32_t>(C_DT_FLOAT16);
        case ge::DT_BF16:    return static_cast<uint32_t>(C_DT_BF16);
        case ge::DT_INT32:   return static_cast<uint32_t>(C_DT_INT32);
        case ge::DT_INT8:    return static_cast<uint32_t>(C_DT_INT8);
        case ge::DT_UINT8:   return static_cast<uint32_t>(C_DT_UINT8);
        default:             break;
    }
    return 0;
}

static ge::graphStatus TruncateModTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    auto sx = context->GetInputShape(0);
    auto sy = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, sx);
    OP_CHECK_NULL_WITH_CONTEXT(context, sy);

    auto vsx = ShapeToVec(sx->GetStorageShape());
    auto vsy = ShapeToVec(sy->GetStorageShape());
    auto outShape = ComputeBroadcastShape(vsx, vsy);
    int64_t totalNum = ShapeSize(outShape);

    auto inDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inDesc);
    ge::DataType dt = inDesc->GetDataType();

    const std::set<ge::DataType> supportDt = {
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT8, ge::DT_UINT8
    };
    OP_CHECK_IF(supportDt.count(dt) == 0, OP_LOGE(context, "unsupported dtype"), return ge::GRAPH_FAILED);

    int64_t typeSize = 4;
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) typeSize = 2;
    else if (dt == ge::DT_INT8 || dt == ge::DT_UINT8) typeSize = 1;
    else if (dt == ge::DT_FLOAT || dt == ge::DT_INT32) typeSize = 4;

    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = 0;

    TruncateModTilingData* td = context->GetTilingData<TruncateModTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    OP_CHECK_IF(
        memset_s(td, sizeof(TruncateModTilingData), 0, sizeof(TruncateModTilingData)) != EOK,
        OP_LOGE(context, "memset tiling error"), return ge::GRAPH_FAILED);

    td->totalNum = totalNum;
    td->outDims = static_cast<int64_t>(outShape.size());
    td->x1Dims = static_cast<int64_t>(vsx.size());
    td->x2Dims = static_cast<int64_t>(vsy.size());
    for (size_t i = 0; i < outShape.size() && i < 8; ++i) td->outShape[i] = outShape[i];
    for (size_t i = 0; i < vsx.size() && i < 8; ++i) td->x1Shape[i] = vsx[i];
    for (size_t i = 0; i < vsy.size() && i < 8; ++i) td->x2Shape[i] = vsy[i];

    if (totalNum == 0) {
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, GeDtToAscendCDt(dt), 0);
        return ge::GRAPH_SUCCESS;
    }

    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    td->blockFactor = CeilDiv(totalNum, coreNum);
    int64_t usedCoreNum = CeilDiv(totalNum, td->blockFactor);

    int64_t tempBytesPerElem = 16;
    if (dt == ge::DT_INT8 || dt == ge::DT_UINT8) tempBytesPerElem = 22;
    else if (dt == ge::DT_INT32) tempBytesPerElem = 20;
    int64_t totalBytesNoDB = 3 * typeSize + tempBytesPerElem;
    int64_t ubFactorNoDB = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize), totalBytesNoDB), ubBlockSize);
    int64_t useDoubleBuf = (td->blockFactor > ubFactorNoDB) ? 1 : 0;
    int64_t pipeBytesPerElem = (useDoubleBuf ? 6 : 3) * typeSize;
    int64_t totalBytes = pipeBytesPerElem + tempBytesPerElem;
    td->ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize), totalBytes), ubBlockSize);

    context->SetBlockDim(usedCoreNum);
    ASCENDC_TPL_SEL_PARAM(context, GeDtToAscendCDt(dt), static_cast<uint32_t>(useDoubleBuf));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTruncateMod([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TruncateModCompileInfo {};

IMPL_OP_OPTILING(TruncateMod).Tiling(TruncateModTilingFunc).TilingParse<TruncateModCompileInfo>(TilingParseForTruncateMod);

} // namespace optiling
