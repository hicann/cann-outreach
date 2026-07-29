#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

// 全局常量定义
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t TILE_NUM   = 8;

template <typename dtypeX, typename dtypeY, typename dtypeZ>
class KernelSub
{
public:
    // 构造函数
    __aicore__ inline KernelSub() = default;

    /**
     * @brief 初始化流水线、队列、GM缓冲区、分块与分片长度
     * @param x 输入1全局地址
     * @param y 输入2全局地址
     * @param z 输出全局地址
     * @param totalLength 对齐后的总数据长度
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        uint32_t blockNum    = AscendC::GetBlockNum();
        uint32_t blockIdx    = AscendC::GetBlockIdx();
        uint32_t baseBlockLen= totalLength / blockNum;
        uint32_t tailBlockCnt= totalLength % blockNum;

        // 计算当前Block在GM中的偏移量
        uint32_t blockOffset = blockIdx * baseBlockLen + (blockIdx < tailBlockCnt ? blockIdx : tailBlockCnt);
        // 计算当前Block实际承载的数据长度
        this->blockLength    = baseBlockLen + (blockIdx < tailBlockCnt ? 1 : 0);
        // 计算单片Tile长度（向上取整均分所有分片）
        this->tileLength     = (this->blockLength + TILE_NUM * BUFFER_NUM - 1) / (TILE_NUM * BUFFER_NUM);

        // 绑定全局GM缓冲区
        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + blockOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + blockOffset, this->blockLength);

        // 非空数据时初始化流水线队列空间
        if (this->blockLength > 0)
        {
            pipe.InitBuffer(inQueueX,  BUFFER_NUM, this->tileLength * sizeof(dtypeX));
            pipe.InitBuffer(inQueueY,  BUFFER_NUM, this->tileLength * sizeof(dtypeY));
            pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(dtypeZ));
        }
    }

    /**
     * @brief 数据处理主流程：循环执行 CopyIn -> Compute -> CopyOut
     */
    __aicore__ inline void Process()
    {
        for (int32_t progress = 0; progress < TILE_NUM * BUFFER_NUM; progress++)
        {
            uint32_t tileOffset = progress * this->tileLength;
            if (tileOffset >= this->blockLength)
            {
                break;
            }

            this->currentTileLength = this->blockLength - tileOffset;
            if (this->currentTileLength > this->tileLength)
            {
                this->currentTileLength = this->tileLength;
            }

            CopyIn(progress);
            Compute(progress);
            CopyOut(progress);
        }
    }

private:
    /**
     * @brief GM数据搬运至Local输入队列
     * @param progress 当前分片序号
     */
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();

        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->currentTileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->currentTileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    /**
     * @brief 执行向量减法计算
     * @param progress 当前分片序号（未使用，占位保留入参）
     */
    __aicore__ inline void Compute(int32_t progress)
    {
        (void)progress;
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();

        AscendC::Sub(zLocal, xLocal, yLocal, this->currentTileLength);

        outQueueZ.EnQue<dtypeZ>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    /**
     * @brief 计算结果从Local队列写回GM内存
     * @param progress 当前分片序号
     */
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->currentTileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    // 流水线管理
    AscendC::TPipe pipe;

    // 输入输出队列
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;

    // 全局GM张量缓冲区
    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeY> yGm;
    AscendC::GlobalTensor<dtypeZ> zGm;

    // 分块、分片长度成员变量
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t currentTileLength;
};

// 内核入口函数
extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x,
                                                          GM_ADDR y,
                                                          GM_ADDR z,
                                                          GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(SubCustomTemplateTilingData, tilingData, tiling);

    KernelSub<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}