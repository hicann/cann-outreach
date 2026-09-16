#include "kernel_operator.h"
#include "div_custom_template_tiling.h"

// BUFFER_NUM = 2 启用 Double Buffer，使 GM->UB 搬运（MTE）与 UB 内向量计算（Vector）流水线重叠
constexpr int32_t BUFFER_NUM = 2;
// 队列深度，与 BUFFER_NUM 保持一致
constexpr int32_t QUEUE_DEPTH = 2;

// 矢量除法核函数类：z = x / y
// 数据通路：GM(x,y) --CopyIn--> UB(xLocal,yLocal) --Compute(Div)--> UB(zLocal) --CopyOut--> GM(z)
template <class dtypeX, class dtypeY, class dtypeZ>
class KernelDiv {
public:
    __aicore__ inline KernelDiv() {}

    // 初始化：根据 tiling 参数绑定 GM 张量、分配 UB 队列缓冲区
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t size, uint32_t tileNum)
    {
        // 每核处理的元素数：总元素按核数均分
        this->blockLength = size / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        // 每个 tile 的元素数：blockLength / (tileNum * BUFFER_NUM)
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 绑定 GM 张量，按当前核索引偏移到本核负责的数据段
        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        // 在 Unified Buffer 上为输入/输出队列分配 BUFFER_NUM 份 tileLength 大小的缓冲区
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dtypeX));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(dtypeY));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(dtypeZ));
    }

    // 主处理流程：循环执行 CopyIn -> Compute -> CopyOut，利用 Double Buffer 实现流水线重叠
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // 搬运输入：从 GM 拷贝当前 tile 的 x、y 到 UB，并入队等待计算
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // 计算：出队 x、y，执行逐元素除法 z = x / y，结果入队等待搬出
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();
        // 逐元素除法：zLocal = xLocal / yLocal
        AscendC::Div(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // 搬出结果：出队结果，从 UB 拷回 GM 对应区间
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;                                                     // UB 内存管理管道
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueX, inQueueY;  // 输入队列（Vector 入）
    AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> outQueueZ;          // 输出队列（Vector 出）
    AscendC::GlobalTensor<dtypeX> xGm;                                       // GM 上的输入 x 张量
    AscendC::GlobalTensor<dtypeY> yGm;                                       // GM 上的输入 y 张量
    AscendC::GlobalTensor<dtypeZ> zGm;                                       // GM 上的输出 z 张量
    uint32_t blockLength;                                                    // 每核处理的元素数
    uint32_t tileNum;                                                        // 每核内的分块次数
    uint32_t tileLength;                                                     // 每个 tile 的元素数
};

// 核函数入口：注册 TilingData 类型，获取 tiling 数据，实例化 KernelDiv 并执行
extern "C" __global__ __aicore__ void div_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(DivCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // DTYPE_X / DTYPE_Y / DTYPE_Z 由编译框架根据算子原型 dtype 定义自动生成
    KernelDiv<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tilingData.size, tilingData.tileNum);
    op.Process();
}
