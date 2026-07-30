#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2; 

// 1. 将内核实现类定义为模板类，支持传入不同的数据类型 T
template<typename T>
class KernelSub {
public:
    __aicore__ inline KernelSub(){}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        // 获取当前核心的计算切片
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum;

        // 绑定 Global 内存，并根据模板类型 T 自动适配指针步长
        xGm.SetGlobalBuffer((__gm__ T*)(x) + GetBlockIdx() * this->blockLength, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ T*)(y) + GetBlockIdx() * this->blockLength, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ T*)(z) + GetBlockIdx() * this->blockLength, this->blockLength);

        // 为本地队列申请 Buffer，大小根据 sizeof(T) 动态计算
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<T> xLocal = inQueueX.template AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.template AllocTensor<T>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal); 
        inQueueY.EnQue(yLocal); 
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<T> xLocal = inQueueX.template DeQue<T>();
        LocalTensor<T> yLocal = inQueueY.template DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.template AllocTensor<T>();
        
        // 调用通用 Sub 接口，会自动适配 float16(half) 或 float 的基础矢量减法
        Sub(zLocal, xLocal, yLocal, this->tileLength);
        
        outQueueZ.EnQue(zLocal); 
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<T> zLocal = outQueueZ.template DeQue<T>();
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

// 2. 核函数入口：通过标准宏 DTYPE_X 获取当前编译分支的真实类型
extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    
    // 使用当前编译分支的数据类型（DTYPE_X 由编译环境自动注入，代表当前实例化的类型，如 half 或 float）
    KernelSub<DTYPE_X> op;
    op.Init(x, y, z, tilingData.size, 8);
    op.Process();
}