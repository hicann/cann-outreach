#include "kernel_operator.h"
#include "div_custom_template_tiling.h"
using namespace AscendC;

namespace {
constexpr int32_t BUFFER_NUM = 1;   // 队列深度：单缓冲，一个 buffer 在计算时，下一个 buffer 可并行搬运
constexpr int32_t TILE_SIZE = 2048; // 单次搬运/计算的数据量（元素个数），保证单次 UB 占用不超过上限
}

// 模板化核函数：T 为 half（float16）或 float（float32），与算子原型声明的 dtype 保持一致
template <typename T>
class KernelDivCustomTemplate {
public:
    __aicore__ inline KernelDivCustomTemplate(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                              const DivCustomTemplateTilingData &tilingData)
        : totalLength(tilingData.size)
    {
        // GM 侧输入输出 GlobalTensor，分别指向 x/y/z 的首地址
        xGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x), totalLength);
        yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y), totalLength);
        zGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(z), totalLength);
    }

    __aicore__ inline void Init()
    {
        // 通过 TPipe 为 VECIN / VECOUT 队列分配 UB 缓冲区（单位：字节，深度为 BUFFER_NUM）
        pipe.InitBuffer(xQueue, BUFFER_NUM, TILE_SIZE * sizeof(T));
        pipe.InitBuffer(yQueue, BUFFER_NUM, TILE_SIZE * sizeof(T));
        pipe.InitBuffer(zQueue, BUFFER_NUM, TILE_SIZE * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        // 多核切分：将 totalLength 个元素按核数均匀划分，每个核处理一段连续区间
        uint32_t blockDim = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        uint32_t perCore = (totalLength + blockDim - 1) / blockDim;
        uint32_t start = blockIdx * perCore;
        uint32_t end = (start + perCore > totalLength) ? totalLength : (start + perCore);

        // 每个核内再按 TILE_SIZE 分块，循环完成 搬运 -> 计算 -> 搬出
        for (uint32_t offset = start; offset < end; offset += TILE_SIZE) {
            uint32_t len = (end - offset > TILE_SIZE) ? TILE_SIZE : (end - offset);
            CopyIn(offset, len);
            Compute(offset, len);
            CopyOut(offset, len);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len)
    {
        // 从 GM 搬运一块 x/y 数据到 UB 并送入输入队列
        LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
        LocalTensor<T> yLocal = yQueue.AllocTensor<T>();
        DataCopy(xLocal, xGlobal[offset], len);
        DataCopy(yLocal, yGlobal[offset], len);
        xQueue.EnQue(xLocal);
        yQueue.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t offset, uint32_t len)
    {
        // 从队列取出输入，使用矢量除法指令完成 z = x / y，结果送入输出队列
        LocalTensor<T> xLocal = xQueue.DeQue<T>();
        LocalTensor<T> yLocal = yQueue.DeQue<T>();
        LocalTensor<T> zLocal = zQueue.AllocTensor<T>();
        Div(zLocal, xLocal, yLocal, len);
        zQueue.EnQue(zLocal);
        xQueue.FreeTensor(xLocal);
        yQueue.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len)
    {
        // 将计算结果从 UB 搬回 GM 的 z 输出
        LocalTensor<T> zLocal = zQueue.DeQue<T>();
        DataCopy(zGlobal[offset], zLocal, len);
        zQueue.FreeTensor(zLocal);
    }

private:
    GlobalTensor<T> xGlobal;   // GM 侧输入 x
    GlobalTensor<T> yGlobal;   // GM 侧输入 y
    GlobalTensor<T> zGlobal;   // GM 侧输出 z
    TPipe pipe;                                    // 用于分配 UB 缓冲区的流水线对象
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;   // x 输入队列
    TQue<QuePosition::VECIN, BUFFER_NUM> yQueue;   // y 输入队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> zQueue;  // z 输出队列
    uint32_t totalLength;                          // 元素总数（来自 host 侧 tiling）
};

extern "C" __global__ __aicore__ void div_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace,
                                                          GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(DivCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    // 根据 host 侧记录的 dtype 选择 half / float 实例，保证与算子原型声明的 fp16/fp32 支持一致
    if (tilingData.dtype == 1) {
        KernelDivCustomTemplate<float> op(x, y, z, tilingData);
        op.Init();
        op.Process();
    } else {
        KernelDivCustomTemplate<half> op(x, y, z, tilingData);
        op.Init();
        op.Process();
    }
}
