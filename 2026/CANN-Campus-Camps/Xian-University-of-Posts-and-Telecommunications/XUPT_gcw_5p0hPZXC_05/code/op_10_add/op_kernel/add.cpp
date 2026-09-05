// Kernel侧核函数实现
 	 #include "kernel_operator.h"
 	 
 	 #include "add_tiling.h"
 	 #include "tiling_key_add.h"
 	 
 	 #define TILE_LENGTH 256
 	 
 	 template <class DT_X>
 	 class KernelAdd {
 	 public:
 	     __aicore__ inline KernelAdd() {}
 	 
 	     __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
 	         // 绑定全局内存地址
 	         xGm.SetGlobalBuffer((__gm__ DT_X *)x, length);
 	         yGm.SetGlobalBuffer((__gm__ DT_X *)y, length);
 	         zGm.SetGlobalBuffer((__gm__ DT_X *)z, length);
 	 
 	         // 核间任务分配: 按tile划分
 	         int32_t blockIdx = (int32_t)get_block_idx();
 	         int32_t blockNum = (int32_t)get_block_num();
 	         int32_t totalTile = (int32_t)((length + TILE_LENGTH - 1) / TILE_LENGTH);
 	         int32_t tilePerCore = (totalTile + blockNum - 1) / blockNum;
 	         startTile = blockIdx * tilePerCore;
 	         endTile = (startTile + tilePerCore > totalTile) ? totalTile : startTile + tilePerCore;
 	         totalLength = (int32_t)length;
 	 
 	         // 初始化管道与队列
 	         uint32_t tileBytes = TILE_LENGTH * (uint32_t)sizeof(DT_X);
 	         pipe.InitBuffer(inQueueX, 1, tileBytes);
 	         pipe.InitBuffer(inQueueY, 1, tileBytes);
 	         pipe.InitBuffer(outQueueZ, 1, tileBytes);
 	     }
 	 
 	     __aicore__ inline void CopyIn(int32_t progress) {
 	         AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
 	         AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
 	         int32_t offset = progress * TILE_LENGTH;
 	         int32_t len = (offset + TILE_LENGTH > totalLength) ? (totalLength - offset) : TILE_LENGTH;
 	         AscendC::DataCopy(xLocal, xGm[offset], len);
 	         AscendC::DataCopy(yLocal, yGm[offset], len);
 	         inQueueX.EnQue(xLocal);
 	         inQueueY.EnQue(yLocal);
 	     }
 	 
 	     __aicore__ inline void Compute(int32_t progress) {
 	         AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
 	         AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
 	         AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
 	         int32_t offset = progress * TILE_LENGTH;
 	         int32_t len = (offset + TILE_LENGTH > totalLength) ? (totalLength - offset) : TILE_LENGTH;
 	         AscendC::Add(zLocal, xLocal, yLocal, len);
 	         inQueueX.FreeTensor(xLocal);
 	         inQueueY.FreeTensor(yLocal);
 	         outQueueZ.EnQue(zLocal);
 	     }
 	 
 	     __aicore__ inline void CopyOut(int32_t progress) {
 	         AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
 	         int32_t offset = progress * TILE_LENGTH;
 	         int32_t len = (offset + TILE_LENGTH > totalLength) ? (totalLength - offset) : TILE_LENGTH;
 	         AscendC::DataCopy(zGm[offset], zLocal, len);
 	         outQueueZ.FreeTensor(zLocal);
 	     }
 	 
 	     __aicore__ inline void Process() {
 	         for (int32_t i = startTile; i < endTile; i++) {
 	             CopyIn(i);
 	             Compute(i);
 	             CopyOut(i);
 	         }
 	     }
 	 
 	 private:
 	     AscendC::TPipe pipe;
 	     AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX;
 	     AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueY;
 	     AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
 	     AscendC::GlobalTensor<DT_X> xGm;
 	     AscendC::GlobalTensor<DT_X> yGm;
 	     AscendC::GlobalTensor<DT_X> zGm;
 	     int32_t startTile;
 	     int32_t endTile;
 	     int32_t totalLength;
 	 };
 	 
 	 template <typename DT_X>
 	  __global__ __aicore__ void add(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
 	     REGISTER_TILING_DEFAULT(AddTilingData);
 	     GET_TILING_DATA_WITH_STRUCT(AddTilingData, tiling_data, tiling);
 	     KernelAdd<DT_X> op;
 	     op.Init(x, y, z, tiling_data.length);
 	     op.Process();
 	 }
