/*
 * @file atanh_custom.cpp
 * @brief atanh 算子 Kernel 侧实现（运行在 Ascend AI Core / AIV 上）
 *
 * 计算式：y = atanh(x) = 0.5 * ln((1 + x) / (1 - x))
 *
 * 实现要点：
 *  1. 采用标准五段式结构：Init / Process / CopyIn / Compute / CopyOut
 *  2. 双缓冲（BUFFER_NUM=2）流水线，掩盖数据搬运延迟
 *  3. 中间计算使用 float32，保证 half 输入下的精度（尤其 |x| 接近 1 时）
 *  4. 非对齐尾块用 DataCopyPad 处理，避免 DataCopy 的 32 字节对齐约束
 */

#include "kernel_operator.h"
#include "atanh_custom_tiling.h"

using namespace AscendC;

// 双缓冲数量，与 Host 侧保持一致
constexpr uint32_t BUFFER_NUM = 2;

class AtanhCustomKernel {
public:
    __aicore__ inline AtanhCustomKernel() {}

    /*
     * Init：绑定全局内存、分配 UB 队列 buffer、解析 tiling 参数
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tilingGm)
    {
        // 反序列化 Host 侧写入的 tiling 参数
        GET_TILING_DATA(tilingData, tilingGm);

        totalLength = tilingData.totalLength;
        blockLength = tilingData.blockLength;
        tileLength = tilingData.tileLength;

        // 计算当前核处理的数据区间 [start, start + myLen)
        uint32_t start = GetBlockIdx() * blockLength;
        uint32_t myLen = 0;
        if (start < totalLength) {
            myLen = (start + blockLength <= totalLength) ? blockLength : (totalLength - start);
        }

        // 核内 tile 划分
        tileNum = myLen / tileLength;          // 完整 tile 数量
        lastTileLength = myLen % tileLength;   // 尾部剩余长度（可能非对齐）
        loopCount = tileNum + (lastTileLength > 0 ? 1 : 0);

        // 绑定当前核专属的全局内存区间
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x) + start, myLen);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y) + start, myLen);

        // 分配 UB 队列 buffer
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(half));
        // 中间计算 buffer（float32 精度）
        pipe.InitBuffer(tmpQueue1, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(tmpQueue2, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(tmpQueue3, BUFFER_NUM, tileLength * sizeof(float));
    }

    /*
     * Process：流水线总调度，循环执行 CopyIn -> Compute -> CopyOut
     */
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // 获取第 progress 个 tile 的实际长度（最后一个 tile 可能是非对齐尾块）
    __aicore__ inline uint32_t GetTileLength(uint32_t progress)
    {
        if (lastTileLength > 0 && progress == tileNum) {
            return lastTileLength;
        }
        return tileLength;
    }

    // CopyIn：GM -> UB
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        uint32_t len = GetTileLength(progress);
        uint32_t offset = progress * tileLength;

        if (len == tileLength) {
            // 常规路径：长度已对齐，直接 DataCopy
            DataCopy(xLocal, xGm[offset], tileLength);
        } else {
            // 尾块路径：长度非对齐，用 DataCopyPad 让硬件自动补足 32 字节对齐
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(len * sizeof(half));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            DataCopyPadExtParams padParams;
            padParams.isPad = false;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = 0;
            DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }

        inQueueX.EnQue(xLocal);
    }

    // Compute：UB 内矢量计算
    __aicore__ inline void Compute(uint32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();

        uint32_t len = GetTileLength(progress);

        // 中间 float32 buffer
        LocalTensor<float> tmp1 = tmpQueue1.AllocTensor<float>();
        LocalTensor<float> tmp2 = tmpQueue2.AllocTensor<float>();
        LocalTensor<float> tmp3 = tmpQueue3.AllocTensor<float>();

        const float one = 1.0f;
        const float negOne = -1.0f;
        const float scale = 0.5f;

        // ---- y = atanh(x) = 0.5 * ln((1 + x) / (1 - x)) ----
        // 1. half -> float32（低精度转高精度，无损失，用 CAST_NONE）
        Cast(tmp1, xLocal, RoundMode::CAST_NONE, len);

        // 2. tmp2 = 1 + x
        Adds(tmp2, tmp1, one, len);

        // 3. tmp3 = 1 - x
        Muls(tmp3, tmp1, negOne, len);
        Adds(tmp3, tmp3, one, len);

        // 4. tmp2 = (1 + x) / (1 - x)
        Div(tmp2, tmp2, tmp3, len);

        // 5. tmp2 = ln((1 + x) / (1 - x))
        Ln(tmp2, tmp2, len);

        // 6. tmp2 = 0.5 * ln(...)
        Muls(tmp2, tmp2, scale, len);

        // 7. float32 -> half（高精度转低精度，四舍五入，用 CAST_ROUND）
        Cast(yLocal, tmp2, RoundMode::CAST_ROUND, len);

        outQueueY.EnQue<half>(yLocal);
        inQueueX.FreeTensor(xLocal);
        tmpQueue1.FreeTensor(tmp1);
        tmpQueue2.FreeTensor(tmp2);
        tmpQueue3.FreeTensor(tmp3);
    }

    // CopyOut：UB -> GM
    __aicore__ inline void CopyOut(uint32_t progress)
    {
        LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        uint32_t len = GetTileLength(progress);
        uint32_t offset = progress * tileLength;

        if (len == tileLength) {
            DataCopy(yGm[offset], yLocal, tileLength);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(len * sizeof(half));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            DataCopyPadExtParams padParams;
            padParams.isPad = false;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = 0;
            DataCopyPad(yGm[offset], yLocal, copyParams, padParams);
        }

        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;                                            // UB 内存管理
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;         // 输入队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;       // 输出队列
    TQue<QuePosition::VECCALC, BUFFER_NUM> tmpQueue1;      // 中间 buffer 1
    TQue<QuePosition::VECCALC, BUFFER_NUM> tmpQueue2;      // 中间 buffer 2
    TQue<QuePosition::VECCALC, BUFFER_NUM> tmpQueue3;      // 中间 buffer 3

    GlobalTensor<half> xGm;   // 输入全局内存
    GlobalTensor<half> yGm;   // 输出全局内存

    uint32_t totalLength;     // 总元素数
    uint32_t blockLength;     // 每核基准长度
    uint32_t tileLength;      // 单个 tile 长度
    uint32_t tileNum;         // 完整 tile 数
    uint32_t lastTileLength;  // 尾块长度
    uint32_t loopCount;       // 循环次数
};

/*
 * 核函数入口：
 *  - 核函数名必须为"算子类型名转下划线命名"：AtanhCustom -> atanh_custom
 *  - 签名固定为 4 个 GM_ADDR：输入、输出、workspace、tiling
 */
extern "C" __global__ __aicore__ void atanh_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tilingGm)
{
    AtanhCustomKernel op;
    op.Init(x, y, workspace, tilingGm);
    op.Process();
}
