/**
 * DivCustomTemplate 算子 kernel 侧实现（Ascend C）
 * 功能：逐元素除法 z = x / y，支持 float16 / float32
 * 实现思路：Init 按核均分并初始化 UB 队列（Double Buffer）；
 *          Process 按 tile 循环 CopyIn -> Compute -> CopyOut。
 */

#include "div_custom_template_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

// Double Buffer：每个队列缓存 2 块 UB，实现搬运与计算流水并行
constexpr int32_t BUFFER_NUM = 2;
// dataType 标记（与 host 侧 tiling 约定一致）
constexpr uint32_t DTYPE_FP16 = 0;
constexpr uint32_t DTYPE_FP32 = 1;

template <typename T>
class KernelDiv {
public:
    __aicore__ inline KernelDiv() {}

    /**
     * 初始化：按核切分数据，设置 GM 全局指针，申请 UB 队列
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileLength)
    {
        // 数据按核均分：每个核处理 blockLength 个元素
        uint32_t blockLength = totalLength / GetBlockNum();
        this->tileLength = tileLength;
        // 每个核内按 tileLength 再切分，得到本核需要循环的次数
        this->tileNum = blockLength / tileLength;

        // GlobalTensor 管理 GM 数据，按 block_idx 偏移到本核负责的数据段
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x) + GetBlockIdx() * blockLength, blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y) + GetBlockIdx() * blockLength, blockLength);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(z) + GetBlockIdx() * blockLength, blockLength);

        // 申请 UB 队列：tileLength * sizeof(T) 字节，Double Buffer
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(T));
    }

    /**
     * 主流程：按 tile 循环执行 CopyIn -> Compute -> CopyOut
     */
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileNum; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    /**
     * 搬入：将第 tileIndex 个 tile 的 x/y 数据从 GM 拷贝到 UB，并入队等待计算
     */
    __aicore__ inline void CopyIn(uint32_t tileIndex)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        DataCopy(xLocal, xGm[tileIndex * tileLength], tileLength);
        DataCopy(yLocal, yGm[tileIndex * tileLength], tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    /**
     * 计算：出队 x/y，执行逐元素除法 z = x / y，结果入队等待搬出
     */
    __aicore__ inline void Compute(uint32_t tileIndex)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = inQueueY.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        // 矢量除法：z[i] = x[i] / y[i]
        Div(zLocal, xLocal, yLocal, tileLength);

        outQueueZ.EnQue(zLocal);
        // 计算完成后释放输入 UB 块，供 Double Buffer 复用
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    /**
     * 搬出：出队 z，将结果从 UB 拷贝回 GM 对应位置
     */
    __aicore__ inline void CopyOut(uint32_t tileIndex)
    {
        LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        DataCopy(zGm[tileIndex * tileLength], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;                                      // 流水管理器
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;   // 输入 x 队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;   // 输入 y 队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ; // 输出 z 队列
    GlobalTensor<T> xGm, yGm, zGm;                   // GM 全局数据指针
    uint32_t tileLength = 0;                         // 单 tile 长度
    uint32_t tileNum = 0;                            // 本核 tile 循环次数
};

/**
 * 算子入口函数：解析 tiling 数据，按数据类型实例化 KernelDiv 并执行
 */
extern "C" __global__ __aicore__ void div_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    // 从 GM 中解析 host 侧写入的 tiling 参数
    GET_TILING_DATA(tilingData, tiling);

    if (tilingData.dataType == DTYPE_FP32) {
        KernelDiv<float> op;
        op.Init(x, y, z, tilingData.totalLength, tilingData.tileLength);
        op.Process();
    } else {
        KernelDiv<half> op;
        op.Init(x, y, z, tilingData.totalLength, tilingData.tileLength);
        op.Process();
    }
}
