#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelSub {
public:
    __aicore__ inline KernelSub() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t size)
    {
        // 总长度先按 AI Core 数量平分；当前题目中 16384 个元素分给 8 个 core。
        blockLength = size / AscendC::GetBlockNum();

        // 每个 core 再切成两个 tile，方便用队列和片上缓存搬运数据。
        tileLength = blockLength / BUFFER_NUM;
        uint32_t blockOffset = blockLength * AscendC::GetBlockIdx();

        // x/y/z 都是 GM 全局内存地址。这里把指针偏移到当前 core 负责的数据段开头。
        xGm.SetGlobalBuffer((__gm__ T *)x + blockOffset, blockLength);
        yGm.SetGlobalBuffer((__gm__ T *)y + blockOffset, blockLength);
        zGm.SetGlobalBuffer((__gm__ T *)z + blockOffset, blockLength);

        // 为两个输入队列和一个输出队列申请 Local Memory 缓冲。
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        // 一个循环处理一个 tile：搬入 x/y -> 做减法 -> 搬出 z。
        for (int32_t i = 0; i < BUFFER_NUM; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 从 GM 读入当前 tile。后续计算只操作 LocalTensor，不直接在 GM 上算。
        AscendC::LocalTensor<T> xLocal = inQueueX.template AllocTensor<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.template AllocTensor<T>();
        AscendC::DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * tileLength], tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        // 取出当前 tile 的 x 和 y，执行逐元素 z = x - y。
        AscendC::LocalTensor<T> xLocal = inQueueX.template DeQue<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.template DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.template AllocTensor<T>();
        AscendC::Sub(zLocal, xLocal, yLocal, tileLength);
        outQueueZ.template EnQue<T>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 把当前 tile 的结果写回 GM，对应输出 tensor z。
        AscendC::LocalTensor<T> zLocal = outQueueZ.template DeQue<T>();
        AscendC::DataCopy(zGm[progress * tileLength], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    AscendC::GlobalTensor<T> zGm;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    // tiling key 由 host 侧根据输入 dtype 设置，保证 float16/float32 都能复用同一套逻辑。
    if (TILING_KEY_IS(1)) {
        KernelSub<half> op;
        op.Init(x, y, z, tilingData.size);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelSub<float> op;
        op.Init(x, y, z, tilingData.size);
        op.Process();
    }
}
