// Kernel侧核函数实现: Gelu
 	  	 // 按公式 gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2))) 逐元素计算。
 	  	 // 数据按32B对齐块在核间分配: 前tailBlockNum个"大核"各多处理一个对齐块,
 	  	 // 其余"小核"处理较少的对齐块; 每个核内部再按tileDataNum分片双缓冲处理。
 	  	 #include "kernel_operator.h"
 	  	 
 	  	 #include "gelu_tiling.h"
 	  	 #include "tiling_key_gelu.h"
 	  	 
 	  	 constexpr int32_t BUFFER_NUM = 2;
 	  	 
 	  	 template <class DT_INPUT_X>
 	  	 class KernelGelu {
 	  	 public:
 	  	     using TYPE_X = DT_INPUT_X;
 	  	     using TYPE_Y = DT_INPUT_X;
 	  	 
 	  	     __aicore__ inline KernelGelu() {}
 	  	 
 	  	     __aicore__ inline void Init(GM_ADDR input_x,
 	  	                                 GM_ADDR output,
 	  	                                 uint32_t small_core_data_num,
 	  	                                 uint32_t big_core_data_num,
 	  	                                 uint32_t final_big_tile_num,
 	  	                                 uint32_t final_small_tile_num,
 	  	                                 uint32_t tile_data_num,
 	  	                                 uint32_t small_tail_data_num,
 	  	                                 uint32_t big_tail_data_num,
 	  	                                 uint32_t tail_block_num) {
 	  	         const uint32_t core_num = AscendC::GetBlockIdx();
 	  	         // 大核在前, 小核在后, 先按大核步长计算本核数据在global memory上的起始偏移
 	  	         uint32_t global_buffer_index = big_core_data_num * core_num;
 	  	 
 	  	         this->tileDataNum = tile_data_num;
 	  	         this->coreDataNum = (core_num < tail_block_num) ? big_core_data_num : small_core_data_num;
 	  	         this->tileNum = (core_num < tail_block_num) ? final_big_tile_num : final_small_tile_num;
 	  	         this->tailDataNum = (core_num < tail_block_num) ? big_tail_data_num : small_tail_data_num;
 	  	 
 	  	         // 小核需要把前面多算的大核步长差量修正回来
 	  	         if (core_num >= tail_block_num) {
 	  	             global_buffer_index -= (big_core_data_num - small_core_data_num) * (core_num - tail_block_num);
 	  	         }
 	  	 
 	  	         this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_X *>(input_x) + global_buffer_index, this->coreDataNum);
 	  	         this->yGm.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_Y *>(output) + global_buffer_index, this->coreDataNum);
 	  	 
 	  	         this->pipe.InitBuffer(this->inQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
 	  	         this->pipe.InitBuffer(this->outQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
 	  	     }
 	  	 
 	  	     __aicore__ inline void Process() {
 	  	         for (uint32_t i = 0; i < this->tileNum; ++i) {
 	  	             // 最后一片tile可能不足tileDataNum, 按尾片大小处理
 	  	             const uint32_t current_tile_size = (i + 1 == this->tileNum) ? this->tailDataNum : this->tileDataNum;
 	  	             this->CopyIn(i, current_tile_size);
 	  	             this->Compute(current_tile_size);
 	  	             this->CopyOut(i, current_tile_size);
 	  	         }
 	  	     }
 	  	 
 	  	 private:
 	  	     AscendC::TPipe pipe;
 	  	     AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
 	  	     AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
 	  	     AscendC::GlobalTensor<TYPE_X> xGm;
 	  	     AscendC::GlobalTensor<TYPE_Y> yGm;
 	  	 
 	  	     uint32_t coreDataNum;
 	  	     uint32_t tileNum;
 	  	     uint32_t tileDataNum;
 	  	     uint32_t tailDataNum;
 	  	 
 	  	     const TYPE_X inv_sqrt2 = static_cast<TYPE_X>(0.70710678);
 	  	 
 	  	     __aicore__ inline void CopyIn(uint32_t progress, uint32_t current_tile_size) {
 	  	         AscendC::LocalTensor<TYPE_X> xLocal = this->inQueue.template AllocTensor<TYPE_X>();
 	  	         AscendC::DataCopy(xLocal, this->xGm[progress * this->tileDataNum], current_tile_size);
 	  	         this->inQueue.EnQue(xLocal);
 	  	     }
 	  	 
 	  	     __aicore__ inline void Compute(uint32_t current_tile_size) {
 	  	         AscendC::LocalTensor<TYPE_X> xLocal = this->inQueue.template DeQue<TYPE_X>();
 	  	         AscendC::LocalTensor<TYPE_Y> yLocal = this->outQueue.template AllocTensor<TYPE_Y>();
 	  	         // gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
 	  	         AscendC::Muls(yLocal, xLocal, this->inv_sqrt2, current_tile_size);   // x / sqrt(2)
 	  	         AscendC::Erf(yLocal, yLocal, current_tile_size);                     // erf(x / sqrt(2))
 	  	         AscendC::Adds(yLocal, yLocal, static_cast<TYPE_Y>(1.0f), current_tile_size); // 1 + erf(...)
 	  	         AscendC::Muls(yLocal, yLocal, static_cast<TYPE_Y>(0.5f), current_tile_size); // 0.5 * (1 + erf(...))
 	  	         AscendC::Mul(yLocal, yLocal, xLocal, current_tile_size);             // 0.5 * x * (1 + erf(...))
 	  	 
 	  	         this->outQueue.EnQue(yLocal);
 	  	         this->inQueue.FreeTensor(xLocal);
 	  	     }
 	  	 
 	  	     __aicore__ inline void CopyOut(uint32_t progress, uint32_t current_tile_size) {
 	  	         AscendC::LocalTensor<TYPE_Y> yLocal = this->outQueue.template DeQue<TYPE_Y>();
 	  	         AscendC::DataCopy(this->yGm[progress * this->tileDataNum], yLocal, current_tile_size);
 	  	         this->outQueue.FreeTensor(yLocal);
 	  	     }
 	  	 };
 	  	 
 	  	 template <typename DT_INPUT_X>
 	  	 __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
 	  	     REGISTER_TILING_DEFAULT(GeluTilingData);
 	  	     GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
 	  	     KernelGelu<DT_INPUT_X> op;
 	  	     op.Init(input_x,
 	  	             output,
 	  	             tiling_data.smallCoreDataNum,
 	  	             tiling_data.bigCoreDataNum,
 	  	             tiling_data.finalBigTileNum,
 	  	             tiling_data.finalSmallTileNum,
 	  	             tiling_data.tileDataNum,
 	  	             tiling_data.smallTailDataNum,
 	  	             tiling_data.bigTailDataNum,
 	  	             tiling_data.tailBlockNum);
 	  	     op.Process();
 	  	 }