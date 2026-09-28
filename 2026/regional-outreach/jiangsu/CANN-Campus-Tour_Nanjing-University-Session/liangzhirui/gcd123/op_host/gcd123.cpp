/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * gcd123 算子 host 侧实现：
 *   - 算子原型注册（OpDef / OP_ADD）
 *   - 输出 shape 推导（4 维 ND broadcast，与 PyTorch broadcasting 规则一致）
 *   - 输出 dtype 推导
 *   - Tiling 计算：broadcast 有效 stride、多核切分、每核分块参数
 *
 * 算子功能：out = gcd(self, other)
 *   gcd(a, b) = |a|                       b == 0
 *             = gcd(b, fmod(a, b))        b != 0
 *   gcd(0, 0) = 0
 */
#include "gcd123_tiling.h"
#include "register/op_def_registry.h"
#include <iostream>

namespace optiling {

struct TilingCompileInfo {
    int64_t ub_size;
};

// 与 op_kernel/gcd123.cpp 中保持一致
const uint32_t CHUNK = 2048;          // UB 中单个 tile 的最大元素数（fp16）
const uint32_t MAX_BLOCK_DIM = 8;     // 最大使用核数
const uint32_t GCD_MAX_ITER = 64;     // GCD 固定迭代次数
const uint32_t USR_WORKSPACE = 256;   // 用户 workspace（保证 workspace 指针非空）
const uint32_t SYS_WORKSPACE = 0;     // 纯矢量算子无需 sys workspace

// 计算 broadcast 后的输出 shape；不可广播返回 false。
// 对动态 shape（维度为 -1）做透传处理。
static bool ComputeOutShape(const int64_t self[4], const int64_t other[4], int64_t out[4])
{
    for (int d = 0; d < 4; d++) {
        int64_t a = self[d];
        int64_t b = other[d];
        if (a == b) {
            out[d] = a;
        } else if (a < 0 || b < 0) {
            // 动态维度（-1）透传：优先取已知维度
            out[d] = ((a < 0) && (b < 0)) ? -1 : ((a < 0) ? b : a);
        } else if (a == 1) {
            out[d] = b;
        } else if (b == 1) {
            out[d] = a;
        } else {
            return false;
        }
    }
    return true;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::Shape* selfShape = context->GetInputShape(0);
    const gert::Shape* otherShape = context->GetInputShape(1);
    if (selfShape == nullptr || otherShape == nullptr ||
        selfShape->GetDimNum() != 4 || otherShape->GetDimNum() != 4) {
        std::cerr << "gcd123: input rank must be 4." << std::endl;
        return ge::GRAPH_FAILED;
    }

    int64_t selfDim[4];
    int64_t otherDim[4];
    int64_t outDim[4];
    for (int d = 0; d < 4; d++) {
        selfDim[d] = selfShape->GetDim(d);
        otherDim[d] = otherShape->GetDim(d);
    }
    if (!ComputeOutShape(selfDim, otherDim, outDim)) {
        std::cerr << "gcd123: self and other shapes are not broadcastable." << std::endl;
        return ge::GRAPH_FAILED;
    }

    uint64_t total = 1;
    for (int d = 0; d < 4; d++) {
        if (outDim[d] <= 0) {
            std::cerr << "gcd123: dynamic or invalid dim is not supported in tiling." << std::endl;
            return ge::GRAPH_FAILED;
        }
        total *= static_cast<uint64_t>(outDim[d]);
    }
    if (total == 0 || total > UINT32_MAX) {
        std::cerr << "gcd123: total element count out of range." << std::endl;
        return ge::GRAPH_FAILED;
    }
    uint32_t totalLength = static_cast<uint32_t>(total);
    uint32_t n4 = static_cast<uint32_t>(outDim[0]);
    uint32_t n3 = static_cast<uint32_t>(outDim[1]);
    uint32_t n2 = static_cast<uint32_t>(outDim[2]);
    uint32_t n1 = static_cast<uint32_t>(outDim[3]);

    // 计算输入的有效 stride：
    // 某维被广播（该维输入 size == 1）时 stride 置 0；否则使用该输入自身连续排布的 stride。
    // 例：self 为 [1,3,4,1]，out 为 [2,3,4,8]：
    //     self 有效 stride = [0, 4, 1, 0]，坐标 (i4,i3,i2,i1) 映射到 self 的线性偏移 i3*4+i2*1。
    uint32_t selfS[4];
    uint32_t otherS[4];
    uint64_t acc = 1;
    for (int d = 3; d >= 0; d--) {
        selfS[d] = (selfDim[d] == 1) ? 0 : static_cast<uint32_t>(acc);
        acc *= static_cast<uint64_t>(selfDim[d]);
    }
    acc = 1;
    for (int d = 3; d >= 0; d--) {
        otherS[d] = (otherDim[d] == 1) ? 0 : static_cast<uint32_t>(acc);
        acc *= static_cast<uint64_t>(otherDim[d]);
    }

    // 行内数据在 UB 中按 32B 对齐存放，行步长 = AlignUp(N1, 16)
    uint32_t pitch = ((n1 + 15) / 16) * 16;
    uint32_t rowsPerTile = 0;
    uint32_t tileSize = 0;
    uint32_t realPerTile = 0;
    if (n1 < CHUNK) {
        // 行对齐模式：每个 tile 处理整数行
        rowsPerTile = CHUNK / pitch;
        if (rowsPerTile < 1) {
            rowsPerTile = 1;
        }
        if (rowsPerTile > n2) {
            rowsPerTile = n2;
        }
        tileSize = rowsPerTile * pitch;
        realPerTile = rowsPerTile * n1;
    } else {
        // 行内切块模式：N1 超过 CHUNK 时按 CHUNK 大小在行内切块
        tileSize = CHUNK;
        realPerTile = CHUNK;
    }

    uint64_t tilesCount = (total + realPerTile - 1) / realPerTile;
    uint32_t blockDim = static_cast<uint32_t>((total + CHUNK - 1) / CHUNK);
    if (blockDim < 1) {
        blockDim = 1;
    }
    if (blockDim > MAX_BLOCK_DIM) {
        blockDim = MAX_BLOCK_DIM;
    }
    uint64_t coreLen = ((tilesCount + blockDim - 1) / blockDim) * realPerTile;
    if (coreLen > UINT32_MAX) {
        std::cerr << "gcd123: coreLen out of range." << std::endl;
        return ge::GRAPH_FAILED;
    }

    Gcd123TilingData tiling;
    tiling.set_blockDim(blockDim);
    tiling.set_totalLength(totalLength);
    tiling.set_coreLen(static_cast<uint32_t>(coreLen));
    tiling.set_tileSize(tileSize);
    tiling.set_rowsPerTile(rowsPerTile);
    tiling.set_pitch(pitch);
    tiling.set_maxIter(GCD_MAX_ITER);
    tiling.set_n4(n4);
    tiling.set_n3(n3);
    tiling.set_n2(n2);
    tiling.set_n1(n1);
    tiling.set_selfS0(selfS[0]);
    tiling.set_selfS1(selfS[1]);
    tiling.set_selfS2(selfS[2]);
    tiling.set_selfS3(selfS[3]);
    tiling.set_otherS0(otherS[0]);
    tiling.set_otherS1(otherS[1]);
    tiling.set_otherS2(otherS[2]);
    tiling.set_otherS3(otherS[3]);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(blockDim);
    context->SetTilingKey(1);

    if (context->GetWorkspaceNum() <= 0) {
        std::cerr << "gcd123: GetWorkspaceNum failed." << std::endl;
        return ge::GRAPH_FAILED;
    }
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = USR_WORKSPACE + SYS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

int32_t CheckOpSupport(const ge::Operator& op, ge::AscendString& result)
{
    std::string resJsonStr = "{\"ret_code\": \"0\",\"reason\": \"check_supported_stub\"}";
    result = ge::AscendString(resJsonStr.c_str());
    return 1;
}
} // namespace optiling

ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* selfShape = context->GetInputShape(0);
    const gert::Shape* otherShape = context->GetInputShape(1);
    gert::Shape* outShape = context->GetOutputShape(0);
    if (selfShape == nullptr || otherShape == nullptr || outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (selfShape->GetDimNum() != 4 || otherShape->GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    int64_t selfDim[4];
    int64_t otherDim[4];
    int64_t outDim[4];
    for (int d = 0; d < 4; d++) {
        selfDim[d] = selfShape->GetDim(d);
        otherDim[d] = otherShape->GetDim(d);
    }
    if (!optiling::ComputeOutShape(selfDim, otherDim, outDim)) {
        return ge::GRAPH_FAILED;
    }
    for (int d = 0; d < 4; d++) {
        if (outShape->SetDim(d, outDim[d]) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

namespace ops {
class Gcd123 : public OpDef {
public:
    explicit Gcd123(const char* name) : OpDef(name)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("other")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(InferShape)
            // .SetInferShapeRange(InferShapeRange)
            .SetInferDataType(InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .SetTilingParse(optiling::TilingPrepare)
            .SetCheckSupport(optiling::CheckOpSupport);

        OpAICoreConfig aicConfig;
        aicConfig.AsyncFlag(true)
            .DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(true)
            .PrecisionReduceFlag(true)
            .RangeLimitValue("limited");

        this->AICore().AddConfig("ascend910", aicConfig);
        this->AICore().AddConfig("ascend310p", aicConfig);
        this->AICore().AddConfig("ascend910b", aicConfig);
    }
};

OP_ADD(Gcd123, optiling::TilingCompileInfo);
} // namespace ops
