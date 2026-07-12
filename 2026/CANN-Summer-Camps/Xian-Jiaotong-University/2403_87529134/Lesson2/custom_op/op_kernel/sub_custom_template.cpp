#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"
constexpr uint32_t BUFFER_NUM = 2; // tensor num for each queue


class SubCustomTemplate {
public:
    __aicore__ inline SubCustomTemplate(){}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t size)
    {
        // 初始化函数，完成内存初始化相关操作
     this->blockLength = size / AscendC::GetBlockNum();     
     this->tileNum =8;                                      
     this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;  
        
     // get start index for current core, core parallel
     xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
     yGm.SetGlobalBuffer((__gm__ DTYPE_X *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
     zGm.SetGlobalBuffer((__gm__ DTYPE_X *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
     // pipe alloc memory to queue, the unit is Bytes
     pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
     pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
     pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
    }
    __aicore__ inline void Process()
    {
        // 核心处理函数，实现算子逻辑，调用私有成员函数CopyIn、Compute、CopyOut完成矢量算子的三级流水操作
    // loop count need to be doubled, due to double buffer
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        // tiling strategy, pipeline parallel
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 搬入函数，从Global Memory搬运数据至Local Memory，被核心Process函数调用
        // alloc tensor from queue memory
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> yLocal = inQueueY.AllocTensor<DTYPE_X>();
        // copy progress_th tile from global tensor to local tensor
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        // enque input tensors to VECIN queue
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // 计算函数，完成两个输入参数相减，得到最终结果，被核心Process函数调用
        // deque input tensors from VECIN queue
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> yLocal = inQueueY.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> zLocal = outQueueZ.AllocTensor<DTYPE_X>();
        // call Mul instr for computation
        AscendC::Sub(zLocal, xLocal, yLocal, this->tileLength);
        // enque the output tensor to VECOUT queue
        outQueueZ.EnQue<DTYPE_X>(zLocal);
        // free input tensors for reuse
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 搬出函数，将最终结果从Local Memory搬运到Global Memory上，被核心Process函数调用
        // deque output tensor from VECOUT queue
        AscendC::LocalTensor<DTYPE_X> zLocal = outQueueZ.DeQue<DTYPE_X>();
        // copy progress_th tile from local tensor to global tensor
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        // free output tensor for reuse
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;  // TPipe内存管理对象
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;  // 输入数据Queue队列管理对象，TPosition为VECIN
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;  // 输出数据Queue队列管理对象，TPosition为VECOUT
    AscendC::GlobalTensor<DTYPE_X> xGm;  // 管理输入输出Global Memory内存地址的对象，其中xGm, yGm为输入，zGm为输出
    AscendC::GlobalTensor<DTYPE_X> yGm;
    AscendC::GlobalTensor<DTYPE_X> zGm;
    uint32_t blockLength; // 每个核的计算数据长度
    uint32_t tileNum; // 每个核需要计算的数据块个数
    uint32_t tileLength; // 每个核内每个数据块的长度
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // TODO: user kernel impl
    SubCustomTemplate op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}