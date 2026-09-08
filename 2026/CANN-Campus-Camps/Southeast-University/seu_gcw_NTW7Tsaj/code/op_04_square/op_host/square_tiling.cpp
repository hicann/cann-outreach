/*!
 * \file square_tiling.cpp
 * \brief Square 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/square_tiling_data.h"
#include "../op_kernel/square_tiling_key.h"


namespace optiling {


using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;


constexpr uint32_t WS_SYS_SIZE = 0U;


/*
 * Double Buffer
 */
constexpr int32_t BUFFER_NUM = 2;


/*
 * 32Byte 对齐
 */
constexpr int64_t BLOCK_BYTE_SIZE = 32;


/*
 * 保守使用8核
 * 避免blockFactor扩张造成空核访问
 */
constexpr int64_t BLOCK_DIM_CAP = 8;


/*
 * UB预留
 */
constexpr uint64_t UB_RESERVE = 8 * 1024;


/*
 * 默认float32
 */
constexpr int64_t DEFAULT_TYPE_SIZE = 4;



static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{

    fe::PlatFormInfos* platformInfoPtr =
        context->GetPlatformInfo();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        platformInfoPtr);


    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(
            platformInfoPtr);


    coreNum =
        ascendcPlatform.GetCoreNumAiv();


    OP_CHECK_IF(
        coreNum == 0,
        OP_LOGE(context,"core num error"),
        return ge::GRAPH_FAILED);


    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);


    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(context,"UB size error"),
        return ge::GRAPH_FAILED);


    return ge::GRAPH_SUCCESS;
}



static ge::graphStatus GetWorkspaceSize(
    gert::TilingContext* context)
{

    size_t* workspace =
        context->GetWorkspaceSizes(1);


    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        workspace);


    workspace[0]=WS_SYS_SIZE;


    return ge::GRAPH_SUCCESS;
}



static ge::graphStatus SquareTilingFunc(
    gert::TilingContext* context)
{


    uint64_t ubSize;
    int64_t coreNum;


    OP_CHECK_IF(
        GetPlatformInfo(
            context,
            ubSize,
            coreNum)
        != ge::GRAPH_SUCCESS,

        OP_LOGE(context,"platform error"),

        return ge::GRAPH_FAILED);



    OP_CHECK_IF(
        GetWorkspaceSize(context)
        != ge::GRAPH_SUCCESS,

        OP_LOGE(context,"workspace error"),

        return ge::GRAPH_FAILED);



    SquareTilingData* tiling =
        context->GetTilingData<SquareTilingData>();


    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);


    /*
     * 获取输入大小
     */

    auto inputShape =
        context->GetInputShape(0);


    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputShape);



    int64_t totalNum=1;


    auto shape =
        inputShape->GetStorageShape();



    for(size_t i=0;i<shape.GetDimNum();i++)
    {
        totalNum *= shape.GetDim(i);
    }



    if(totalNum<=0)
    {
        totalNum=0;
    }


    /*
     * dtype大小
     */

    int64_t typeSize =
        DEFAULT_TYPE_SIZE;


    auto inputDesc =
        context->GetInputDesc(0);


    if(inputDesc!=nullptr)
    {
        int64_t size =
            static_cast<int64_t>(
                ge::GetSizeByDataType(
                    inputDesc->GetDataType()));


        if(size>0)
        {
            typeSize=size;
        }
    }



    /*
     * 32Byte包含元素数
     */

    int64_t elemPerBlock =
        BLOCK_BYTE_SIZE/typeSize;


    if(elemPerBlock<1)
    {
        elemPerBlock=1;
    }



    /*
     * block数量
     */

    int64_t blockDim =
        coreNum;


    if(blockDim>BLOCK_DIM_CAP)
    {
        blockDim=BLOCK_DIM_CAP;
    }



    if(totalNum>0 &&
       blockDim>totalNum)
    {
        blockDim=totalNum;
    }


    if(blockDim<=0)
    {
        blockDim=1;
    }



    /*
     * 每核处理数量
     */

    int64_t blockFactor =
        CeilDiv(
            totalNum,
            blockDim);



    /*
     * 32B 对齐
     */

    blockFactor =
        (blockFactor+elemPerBlock-1)
        /
        elemPerBlock
        *
        elemPerBlock;



    if(blockFactor<elemPerBlock)
    {
        blockFactor=elemPerBlock;
    }


    /*
     * UB计算
     * input buffer + output buffer
     */
    uint64_t availableUB = ubSize > UB_RESERVE ? ubSize - UB_RESERVE : ubSize / 2;

    int64_t ubFactor = availableUB / (typeSize * BUFFER_NUM * 2);
    ubFactor = ubFactor / elemPerBlock * elemPerBlock;
    if (ubFactor < elemPerBlock) {
        ubFactor = elemPerBlock;
    }
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    // ========== 新增 UB 容量检查 ==========
    uint64_t requiredUB = ubFactor * typeSize * BUFFER_NUM * 2;   // 实际需要分配的 UB 字节数
    if (requiredUB > availableUB) {
        OP_LOGE(context,
                "UB size insufficient: required %llu bytes, available %llu bytes, ubFactor=%lld, typeSize=%lld",
                requiredUB, availableUB, ubFactor, typeSize);
        return ge::GRAPH_FAILED;
    }
    // =====================================

    // 写 tiling
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    /*
     * tiling key
     */

    uint64_t key;


    if(inputDesc!=nullptr &&
    (inputDesc->GetDataType()==ge::DT_FLOAT16 ||
        inputDesc->GetDataType()==ge::DT_BF16))
    {

        key =
        GET_TPL_TILING_KEY(
            SQUARE_TPL_SCH_MODE_0);

    }
    else
    {

        key =
        GET_TPL_TILING_KEY(
            SQUARE_TPL_SCH_MODE_1);

    }


    context->SetTilingKey(key);



    return ge::GRAPH_SUCCESS;

}



static ge::graphStatus TilingParseForSquare(
    [[maybe_unused]]
    gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}



struct SquareCompileInfo{};



IMPL_OP_OPTILING(Square)
.Tiling(SquareTilingFunc)
.TilingParse<SquareCompileInfo>(
    TilingParseForSquare);



}