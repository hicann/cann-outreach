// Kernel侧核函数实现
#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;


template <class DT_X>
class KernelMul {
public:

    __aicore__ inline KernelMul() {}


    // ================================================================
    // Init
    // ================================================================

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t length,
        uint32_t blockLength,
        uint32_t tileLength,
        uint32_t tileNum,
        uint32_t lastTileLength)
    {
        // ------------------------------------------------------------
        // 保存Tiling参数
        // ------------------------------------------------------------

        length_ = length;
        blockLength_ = blockLength;
        tileLength_ = tileLength;

        // ------------------------------------------------------------
        // 当前Core编号
        // ------------------------------------------------------------

        blockIdx_ = GetBlockIdx();

        // ------------------------------------------------------------
        // 初始化GM Tensor
        // ------------------------------------------------------------

        xGm_.SetGlobalBuffer(
            (__gm__ DT_X *)x);

        yGm_.SetGlobalBuffer(
            (__gm__ DT_X *)y);

        zGm_.SetGlobalBuffer(
            (__gm__ DT_X *)z);

        // ------------------------------------------------------------
        // 当前Core负责的数据起始位置
        // ------------------------------------------------------------

        blockOffset_ =
            blockIdx_ * blockLength_;

        // ------------------------------------------------------------
        // 当前Core超出有效数据范围
        // ------------------------------------------------------------

        if (blockOffset_ >= length_) {

            actualBlockLength_ = 0;
            actualTileNum_ = 0;
            actualLastTileLength_ = 0;

            return;
        }

        // ------------------------------------------------------------
        // 计算当前Core实际处理的数据长度
        // ------------------------------------------------------------

        uint32_t remainingLength =
            length_ - blockOffset_;

        if (remainingLength < blockLength_) {

            actualBlockLength_ =
                remainingLength;

        } else {

            actualBlockLength_ =
                blockLength_;
        }

        // ------------------------------------------------------------
        // 根据当前Core实际数据量计算Tile数量
        // ------------------------------------------------------------

        actualTileNum_ =
            actualBlockLength_ / tileLength_;

        actualLastTileLength_ =
            actualBlockLength_ % tileLength_;

        if (actualLastTileLength_ != 0) {
            actualTileNum_++;
        }

        // ------------------------------------------------------------
        // 防止未使用参数产生编译警告
        // ------------------------------------------------------------

        (void)tileNum;
        (void)lastTileLength;

        // ------------------------------------------------------------
        // 初始化Pipe Buffer
        // ------------------------------------------------------------

        pipe_.InitBuffer(
            inQueueX_,
            1,
            tileLength_ * sizeof(DT_X));

        pipe_.InitBuffer(
            inQueueY_,
            1,
            tileLength_ * sizeof(DT_X));

        pipe_.InitBuffer(
            outQueueZ_,
            1,
            tileLength_ * sizeof(DT_X));
    }


    // ================================================================
    // Process
    // ================================================================

    __aicore__ inline void Process()
    {
        // ------------------------------------------------------------
        // 当前Core没有数据
        // ------------------------------------------------------------

        if (actualBlockLength_ == 0 ||
            actualTileNum_ == 0) {

            return;
        }

        // ------------------------------------------------------------
        // 逐Tile处理
        // ------------------------------------------------------------

        for (uint32_t i = 0;
             i < actualTileNum_;
             i++) {

            uint32_t currentTileLength;

            // --------------------------------------------------------
            // 判断当前Tile的有效数据长度
            // --------------------------------------------------------

            if ((i == actualTileNum_ - 1) &&
                (actualLastTileLength_ != 0)) {

                currentTileLength =
                    actualLastTileLength_;

            } else {

                currentTileLength =
                    tileLength_;
            }

            // --------------------------------------------------------
            // 当前Tile在GM中的绝对偏移
            // --------------------------------------------------------

            uint32_t offset =
                blockOffset_ +
                i * tileLength_;

            // --------------------------------------------------------
            // 边界检查
            // --------------------------------------------------------

            if (offset >= length_) {
                break;
            }

            uint32_t remain =
                length_ - offset;

            if (currentTileLength > remain) {
                currentTileLength = remain;
            }

            if (currentTileLength == 0) {
                break;
            }

            // --------------------------------------------------------
            // GM -> UB
            // --------------------------------------------------------

            CopyIn(
                offset,
                currentTileLength);

            // --------------------------------------------------------
            // Vector Mul
            // --------------------------------------------------------

            Compute(
                currentTileLength);

            // --------------------------------------------------------
            // UB -> GM
            // --------------------------------------------------------

            CopyOut(
                offset,
                currentTileLength);
        }
    }


private:


    // ================================================================
    // CopyIn
    // GM -> UB
    // ================================================================

    __aicore__ inline void CopyIn(
        uint32_t offset,
        uint32_t length)
    {
        // ------------------------------------------------------------
        // 申请x的LocalTensor
        // ------------------------------------------------------------

        LocalTensor<DT_X> xLocal =
            inQueueX_.AllocTensor<DT_X>();

        // ------------------------------------------------------------
        // 申请y的LocalTensor
        // ------------------------------------------------------------

        LocalTensor<DT_X> yLocal =
            inQueueY_.AllocTensor<DT_X>();

        // ------------------------------------------------------------
        // 判断是否32字节对齐
        // ------------------------------------------------------------

        constexpr uint32_t BLOCK_SIZE = 32;

        uint32_t dataBytes =
            length * sizeof(DT_X);

        bool isAligned =
            (dataBytes % BLOCK_SIZE == 0);

        // ------------------------------------------------------------
        // 普通对齐数据
        // ------------------------------------------------------------

        if (isAligned) {

            DataCopy(
                xLocal,
                xGm_[offset],
                length);

            DataCopy(
                yLocal,
                yGm_[offset],
                length);

        } else {

            // --------------------------------------------------------
            // 非32字节对齐数据
            //
            // 使用DataCopyPad进行搬运
            // 右侧补0
            // --------------------------------------------------------

            uint32_t alignedLength =
                ((dataBytes + BLOCK_SIZE - 1) /
                 BLOCK_SIZE) *
                BLOCK_SIZE;

            uint32_t alignedElements =
                alignedLength /
                sizeof(DT_X);

            uint32_t padElements =
                alignedElements - length;

            DataCopyExtParams copyParams;

            copyParams.blockCount = 1;
            copyParams.blockLen = dataBytes;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;

            DataCopyPadExtParams<DT_X> padParams;

            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding =
                static_cast<uint8_t>(padElements);
            padParams.paddingValue = static_cast<DT_X>(0);

            // --------------------------------------------------------
            // x
            // --------------------------------------------------------

            DataCopyPad(
                xLocal,
                xGm_[offset],
                copyParams,
                padParams);

            // --------------------------------------------------------
            // y
            // --------------------------------------------------------

            DataCopyPad(
                yLocal,
                yGm_[offset],
                copyParams,
                padParams);
        }

        // ------------------------------------------------------------
        // 放入Queue
        // ------------------------------------------------------------

        inQueueX_.EnQue(xLocal);
        inQueueY_.EnQue(yLocal);
    }


    // ================================================================
    // Compute
    // z = x * y
    // ================================================================

    __aicore__ inline void Compute(
        uint32_t length)
    {
        // ------------------------------------------------------------
        // 从Queue取出x
        // ------------------------------------------------------------

        LocalTensor<DT_X> xLocal =
            inQueueX_.DeQue<DT_X>();

        // ------------------------------------------------------------
        // 从Queue取出y
        // ------------------------------------------------------------

        LocalTensor<DT_X> yLocal =
            inQueueY_.DeQue<DT_X>();

        // ------------------------------------------------------------
        // 申请输出Tensor
        // ------------------------------------------------------------

        LocalTensor<DT_X> zLocal =
            outQueueZ_.AllocTensor<DT_X>();

        // ------------------------------------------------------------
        // 计算需要处理的32字节对齐长度
        // ------------------------------------------------------------

        constexpr uint32_t BLOCK_SIZE = 32;

        uint32_t dataBytes =
            length * sizeof(DT_X);

        uint32_t alignedBytes =
            ((dataBytes + BLOCK_SIZE - 1) /
             BLOCK_SIZE) *
            BLOCK_SIZE;

        uint32_t computeLength =
            alignedBytes /
            sizeof(DT_X);

        // ------------------------------------------------------------
        // Vector Mul
        // ------------------------------------------------------------

        Mul(
            zLocal,
            xLocal,
            yLocal,
            computeLength);

        // ------------------------------------------------------------
        // 放入输出Queue
        // ------------------------------------------------------------

        outQueueZ_.EnQue(zLocal);

        // ------------------------------------------------------------
        // 释放输入Tensor
        // ------------------------------------------------------------

        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
    }


    // ================================================================
    // CopyOut
    // UB -> GM
    // ================================================================

    __aicore__ inline void CopyOut(
        uint32_t offset,
        uint32_t length)
    {
        // ------------------------------------------------------------
        // 从输出Queue取出结果
        // ------------------------------------------------------------

        LocalTensor<DT_X> zLocal =
            outQueueZ_.DeQue<DT_X>();

        // ------------------------------------------------------------
        // 判断是否32字节对齐
        // ------------------------------------------------------------

        constexpr uint32_t BLOCK_SIZE = 32;

        uint32_t dataBytes =
            length * sizeof(DT_X);

        bool isAligned =
            (dataBytes % BLOCK_SIZE == 0);

        // ------------------------------------------------------------
        // 普通对齐数据
        // ------------------------------------------------------------

        if (isAligned) {

            DataCopy(
                zGm_[offset],
                zLocal,
                length);

        } else {

            // --------------------------------------------------------
            // 非32字节对齐数据
            //
            // 只向GM写回有效数据
            // --------------------------------------------------------

            DataCopyExtParams copyParams;

            copyParams.blockCount = 1;
            copyParams.blockLen = dataBytes;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;

            DataCopyPad(
                zGm_[offset],
                zLocal,
                copyParams);
        }

        // ------------------------------------------------------------
        // 释放输出Tensor
        // ------------------------------------------------------------

        outQueueZ_.FreeTensor(zLocal);
    }


private:

    // ================================================================
    // Pipe
    // ================================================================

    TPipe pipe_;


    // ================================================================
    // 输入Queue
    // ================================================================

    TQue<QuePosition::VECIN, 1> inQueueX_;

    TQue<QuePosition::VECIN, 1> inQueueY_;


    // ================================================================
    // 输出Queue
    // ================================================================

    TQue<QuePosition::VECOUT, 1> outQueueZ_;


    // ================================================================
    // GM Tensor
    // ================================================================

    GlobalTensor<DT_X> xGm_;

    GlobalTensor<DT_X> yGm_;

    GlobalTensor<DT_X> zGm_;


    // ================================================================
    // Tiling参数
    // ================================================================

    uint32_t length_;

    uint32_t blockLength_;

    uint32_t tileLength_;


    // ================================================================
    // 当前Core实际处理参数
    // ================================================================

    uint32_t actualBlockLength_;

    uint32_t actualTileNum_;

    uint32_t actualLastTileLength_;


    // ================================================================
    // Core信息
    // ================================================================

    uint32_t blockIdx_;

    uint32_t blockOffset_;
};


// ====================================================================
// Kernel入口函数
// ====================================================================

template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    // ---------------------------------------------------------------
    // 注册Tiling结构
    // ---------------------------------------------------------------

    REGISTER_TILING_DEFAULT(MulTilingData);

    // ---------------------------------------------------------------
    // 获取Tiling数据
    // ---------------------------------------------------------------

    GET_TILING_DATA_WITH_STRUCT(
        MulTilingData,
        tiling_data,
        tiling);

    // ---------------------------------------------------------------
    // 创建Kernel对象
    // ---------------------------------------------------------------

    KernelMul<DT_X> op;

    // ---------------------------------------------------------------
    // 初始化
    // ---------------------------------------------------------------

    op.Init(
        x,
        y,
        z,
        tiling_data.length,
        tiling_data.blockLength,
        tiling_data.tileLength,
        tiling_data.tileNum,
        tiling_data.lastTileLength);

    // ---------------------------------------------------------------
    // 执行计算
    // ---------------------------------------------------------------

    op.Process();

    (void)workspace;
}