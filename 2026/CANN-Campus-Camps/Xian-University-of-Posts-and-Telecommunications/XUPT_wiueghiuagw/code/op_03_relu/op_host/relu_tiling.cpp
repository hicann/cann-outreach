 /*!
 	  * \file relu_tiling.cpp
 	  * \brief Relu 算子 Tiling 实现
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
 	 
 	     OP_CHECK_IF(
 	         coreNum == 0,
 	         OP_LOGE(context, "coreNum is 0"),
 	         return ge::GRAPH_FAILED);
 	 
 	     ascendcPlatform.GetCoreMemSize(
 	         platform_ascendc::CoreMemType::UB,
 	         ubSize);
 	 
 	     OP_CHECK_IF(
 	         ubSize == 0,
 	         OP_LOGE(context, "ubSize is 0"),
 	         return ge::GRAPH_FAILED);
 	 
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
 	     uint64_t ubSize = 0;
 	     int64_t coreNum = 0;
 	 
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
 	 
 	     /*
 	      * 鑾峰彇鐪熷疄杈撳叆鍏冪礌鏁伴噺銆?
 	      *
 	      * 涓嶇洿鎺ュ啓姝婚闈腑鐨?(8, 2048)锛屽洜涓哄伐绋嬭嚜甯︽祴璇曡繕瀛樺湪
 	      * (45, 2048) 绛?shape銆?
 	      */
 	     const auto* inputShape = context->GetInputShape(0);
 	     OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
 	 
 	     const gert::Shape storageShape = EnsureNotScalar(inputShape->GetStorageShape());
 	 
 	     const int64_t totalNum = storageShape.GetShapeSize();
 	 
 	     OP_CHECK_IF(
 	         totalNum <= 0,
 	         OP_LOGE(context, "invalid totalNum"),
 	         return ge::GRAPH_FAILED);
 	 
 	     /*
 	      * 鑾峰彇 dtype銆?
 	      */
 	     const auto* inputDesc = context->GetInputDesc(0);
 	     OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
 	 
 	     const ge::DataType dataType = inputDesc->GetDataType();
 	 
 	     /*
 	      * float32 : 4 Byte
 	      * float16 : 2 Byte
 	      *
 	      * 鍘熸ā鏉垮悓鏃跺垽鏂簡 BF16锛岃繖閲屼繚鎸佸師鏉ョ殑鍏煎閫昏緫銆?
 	      */
 	     int64_t typeSize = TYPE_SIZE;
 	 
 	     if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
 	         typeSize = 2;
 	     }
 	 
 	     /*
 	      * Ascend 鏁版嵁鎼繍鐨勫熀鏈?datablock 涓?32 Byte銆?
 	      *
 	      * float32: 32 / 4 = 8 elements
 	      * float16: 32 / 2 = 16 elements
 	      */
 	     constexpr int64_t BLOCK_BYTES = 32;
 	 
 	     const int64_t alignNum = BLOCK_BYTES / typeSize;
 	 
 	     /*
 	      * 澶氭牳鍒囧垎銆?
 	      *
 	      * 姣忎釜 AIV Core 鐩爣澶勭悊鑷冲皯绾?1024 涓厓绱狅紝
 	      * 閬垮厤杈撳叆涓嶅ぇ鏃跺惎鐢ㄨ繃澶?Core 甯︽潵鐨勮皟搴﹀紑閿€銆?
 	      */
 	     int64_t usedCoreNum = (totalNum + MIN_SPLIT_THRESHOLD - 1) / MIN_SPLIT_THRESHOLD;
 	 
 	     if (usedCoreNum > coreNum) {
 	         usedCoreNum = coreNum;
 	     }
 	 
 	     if (usedCoreNum < 1) {
 	         usedCoreNum = 1;
 	     }
 	 
 	     /*
 	      * 鍚戜笅瀵绘壘锛?
 	      * 1. totalNum 鍙互琚?core 鏁版暣闄?
 	      * 2. 姣忎釜 Core 鐨?blockFactor 婊¤冻 32B 瀵归綈
 	      *
 	      * 杩欐牱 Kernel 涓嶉渶瑕佸鐞?core remainder銆?
 	      */
 	     while (usedCoreNum > 1) {
 	         if ((totalNum % usedCoreNum == 0) &&
 	             ((totalNum / usedCoreNum) % alignNum == 0)) {
 	             break;
 	         }
 	 
 	         --usedCoreNum;
 	     }
 	 
 	     const int64_t blockFactor = totalNum / usedCoreNum;
 	 
 	     /*
 	      * UB 鍒囧垎銆?
 	      *
 	      * Kernel 涓細
 	      * input queue  x double buffer
 	      * output queue x double buffer
 	      *
 	      * 鍏遍渶瑕?4 浠?tile 绌洪棿銆?
 	      *
 	      * 鍚屾椂棰勭暀 8 KB UB锛岄伩鍏嶆妸鏁翠釜 UB 鍏ㄩ儴鍒嗛厤鎺夈€?
 	      */
 	     constexpr uint64_t UB_RESERVED_BYTES = 8 * 1024;
 	 
 	     const uint64_t usableUb = (ubSize > UB_RESERVED_BYTES) ? (ubSize - UB_RESERVED_BYTES) : ubSize;
 	 
 	     int64_t maxUbFactor =
 	         static_cast<int64_t>(usableUb / (4ULL * static_cast<uint64_t>(typeSize)));
 	 
 	     /*
 	      * UB tile 涔熶繚鎸?32B 瀵归綈銆?
 	      */
 	     maxUbFactor = (maxUbFactor / alignNum) * alignNum;
 	 
 	     if (maxUbFactor < alignNum) {
 	         maxUbFactor = alignNum;
 	     }
 	 
 	     int64_t ubFactor = (blockFactor < maxUbFactor) ? blockFactor : maxUbFactor;
 	 
 	     ubFactor = (ubFactor / alignNum) * alignNum;
 	 
 	     if (ubFactor <= 0) {
 	         ubFactor = alignNum;
 	     }
 	 
 	     /*
 	      * 鍐欏叆 TilingData銆?
 	      */
 	     tiling->totalNum = totalNum;
 	     tiling->blockFactor = blockFactor;
 	     tiling->ubFactor = ubFactor;
 	 
 	     /*
 	      * 璁剧疆瀹為檯鍚姩鐨?AIV Core 鏁般€?
 	      */
 	     context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
 	 
 	     /*
 	      * 鏍规嵁杈撳叆 dtype 閫夋嫨 tilingKey銆?
 	      *
 	      * mode 0 -> half
 	      * mode 1 -> float
 	      *
 	      * 涓庡師濮?relu.cpp 淇濇寔瀹屽叏涓€鑷淬€?
 	      */
 	     uint64_t tilingKey;
 	 
 	     if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
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
 	 
 	 IMPL_OP_OPTILING(Relu)
 	     .Tiling(ReluTilingFunc)
 	     .TilingParse<ReluCompileInfo>(TilingParseForRelu);
 	 
 	 } // namespace optiling