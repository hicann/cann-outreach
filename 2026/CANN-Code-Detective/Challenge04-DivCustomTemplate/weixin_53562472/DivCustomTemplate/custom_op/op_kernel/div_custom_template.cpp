/* div_custom_template.cpp (op_kernel)
 *
 * DivCustomTemplate 算子核函数实现：逐元素除法 z = x / y。
 *
 * 实现思路（Ascend C 矢量编程范式）：
 *   1. 将输入数据按 AI Core 数均分，每个核处理 blockLength 个元素；
 *   2. 核内再切分为 tileNum * BUFFER_NUM 个分片，循环执行
 *      CopyIn(GM->UB) -> Compute(Div) -> CopyOut(UB->GM)；
 *   3. 队列深度 BUFFER_NUM = 2（双缓冲），相邻分片的搬运与计算流水重叠，
 *      隐藏数据搬运时延；
 *   4. 核函数实现为模板类，同时支持 float16 / float32，由 host 侧 tiling
 *      记录的 dtype 字段在入口处分发实例。
 */

#include "kernel_operator.h"
#include "div_custom_template_tiling.h"

constexpr uint32_t BUFFER_NUM = 2; // tensor num for each queue（队列缓冲级数：双缓冲）

template <typename T>
class KernelDiv {
public:
    __aicore__ inline KernelDiv() {}

    // 初始化：划分本核负责的数据段，绑定 GM，并在 UB 上申请队列缓冲
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        // 1. 总数据均分到每个 AI Core：本核负责的元素数 = 总长度 / 核数
        //    （本期任务 shape 为 (8, 2048)，总长度 16384，可被 8 核整除）
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        // 2. 核内再按 tile 切分，每个 tile 均分给 BUFFER_NUM 级流水（双缓冲）
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 3. 绑定 GlobalTensor：按当前核号 GetBlockIdx() 偏移到本核负责的数据段
        this->xGm.SetGlobalBuffer((__gm__ T *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        this->yGm.SetGlobalBuffer((__gm__ T *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        this->zGm.SetGlobalBuffer((__gm__ T *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        // 4. 通过 TPipe 在 UB 上为队列分配缓冲，队列深度 = BUFFER_NUM（双缓冲）
        this->pipe.InitBuffer(this->inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        this->pipe.InitBuffer(this->inQueueY, BUFFER_NUM, this->tileLength * sizeof(T));
        this->pipe.InitBuffer(this->outQueueZ, BUFFER_NUM, this->tileLength * sizeof(T));
    }

    // 主流程：共 tileNum * BUFFER_NUM 个分片，逐片执行 CopyIn -> Compute -> CopyOut
    __aicore__ inline void Process()
    {
        const int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // 将本分片的 x/y 从 GM 搬入 UB，入队通知 Compute 输入已就绪
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<T> xLocal = this->inQueueX.template AllocTensor<T>();
        AscendC::LocalTensor<T> yLocal = this->inQueueY.template AllocTensor<T>();
        AscendC::DataCopy(xLocal, this->xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, this->yGm[progress * this->tileLength], this->tileLength);
        this->inQueueX.EnQue(xLocal);
        this->inQueueY.EnQue(yLocal);
    }

    // 出队输入张量，调用 AscendC::Div 完成逐元素除法，结果入队等待搬出
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<T> xLocal = this->inQueueX.template DeQue<T>();
        AscendC::LocalTensor<T> yLocal = this->inQueueY.template DeQue<T>();
        AscendC::LocalTensor<T> zLocal = this->outQueueZ.template AllocTensor<T>();
        AscendC::Div(zLocal, xLocal, yLocal, this->tileLength);
        this->outQueueZ.template EnQue<T>(zLocal);
        this->inQueueX.FreeTensor(xLocal);
        this->inQueueY.FreeTensor(yLocal);
    }

    // 出队计算结果，从 UB 搬回 GM 对应分片，并释放缓冲
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<T> zLocal = this->outQueueZ.template DeQue<T>();
        AscendC::DataCopy(this->zGm[progress * this->tileLength], zLocal, this->tileLength);
        this->outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    AscendC::GlobalTensor<T> zGm;
    uint32_t blockLength; // 本核负责的元素数
    uint32_t tileNum;     // 核内 tile 切分数
    uint32_t tileLength;  // 每个分片的元素数
};

extern "C" __global__ __aicore__ void div_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(DivCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // 根据 host 侧 tiling 记录的数据类型，分发对应的核函数模板实例
    if (tilingData.dtype == DIV_DTYPE_FLOAT16) {
        KernelDiv<half> op;
        op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
        op.Process();
    } else {
        KernelDiv<float> op;
        op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
        op.Process();
    }
}
