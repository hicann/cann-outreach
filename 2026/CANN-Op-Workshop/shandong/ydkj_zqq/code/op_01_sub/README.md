# op_01_sub

提交团队: 云鼎科技股份有限公司
提交者: 13287966125


## 算子实现介绍

#include &lt;cmath&gt;
#include "kernel_operator.h"

// Host 侧计算并传入的分块参数
struct SubCustomTilingData {
    uint32_t totalLength;  // 输入张量的元素总数
    uint32_t tileNum;      // 每核的数据分块数
};

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t QUEUE_DEPTH = 1;

template &lt;typename T&gt;
class KernelSub {
public:
    __aicore__ inline KernelSub() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR z,
        const SubCustomTilingData&amp; tiling)
    {
        // 将数据平均分配给各个核
        blockLength = tiling.totalLength / AscendC::GetBlockNum();
        tileNum = tiling.tileNum;
        tileLength = blockLength / tileNum / BUFFER_NUM;

        // 定位当前核负责的全局内存数据段
        const uint32_t offset =
            blockLength * AscendC::GetBlockIdx();

        xGm.SetGlobalBuffer((__gm__ T*)x + offset, blockLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + offset, blockLength);
        zGm.SetGlobalBuffer((__gm__ T*)z + offset, blockLength);

        // 为输入和输出分别分配片上缓冲区
        pipe.InitBuffer(
            inQueueX, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(
            inQueueY, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(
            outQueueZ, BUFFER_NUM, tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        // 当前配置下 tileNum=1、BUFFER_NUM=1，每核只循环一次
        const int32_t loopCount = tileNum * BUFFER_NUM;

        for (int32_t i = 0; i &lt; loopCount; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 从输入队列申请片上 LocalTensor
        AscendC::LocalTensor&lt;T&gt; xLocal =
            inQueueX.AllocTensor&lt;T&gt;();
        AscendC::LocalTensor&lt;T&gt; yLocal =
            inQueueY.AllocTensor&lt;T&gt;();

        const uint32_t offset = progress * tileLength;

        // 将 x、y 从全局内存搬到片上内存
        AscendC::DataCopy(xLocal, xGm[offset], tileLength);
        AscendC::DataCopy(yLocal, yGm[offset], tileLength);

        // 通知计算阶段：输入数据已准备好
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        // 获取两个输入
        AscendC::LocalTensor&lt;T&gt; xLocal =
            inQueueX.DeQue&lt;T&gt;();
        AscendC::LocalTensor&lt;T&gt; yLocal =
            inQueueY.DeQue&lt;T&gt;();

        // 申请输出空间
        AscendC::LocalTensor&lt;T&gt; zLocal =
            outQueueZ.AllocTensor&lt;T&gt;();

        // 逐元素计算 z = x - y
        AscendC::Sub(zLocal, xLocal, yLocal, tileLength);

        // 将结果交给写回阶段
        outQueueZ.EnQue&lt;T&gt;(zLocal);

        // 输入数据已经用完，释放对应缓冲区
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 获取计算结果
        AscendC::LocalTensor&lt;T&gt; zLocal =
            outQueueZ.DeQue&lt;T&gt;();

        // 将结果写回全局内存
        AscendC::DataCopy(
            zGm[progress * tileLength],
            zLocal,
            tileLength);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue&lt;AscendC::QuePosition::VECIN, QUEUE_DEPTH&gt;
        inQueueX, inQueueY;

    AscendC::TQue&lt;AscendC::QuePosition::VECOUT, QUEUE_DEPTH&gt;
        outQueueZ;

    AscendC::GlobalTensor&lt;T&gt; xGm, yGm, zGm;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

// NPU 侧核函数入口
template &lt;typename T&gt;
__global__ __vector__ void sub_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    SubCustomTilingData tiling)
{
    KernelSub&lt;T&gt; op;
    op.Init(x, y, z, tiling);
    op.Process();
}

// Host 侧入口：读取形状、选择类型并启动核函数
extern "C" void run_kernel(
    GM_ADDR x,
    const TensorGroupInfo&amp; info_x,
    GM_ADDR y,
    const TensorGroupInfo&amp; info_y,
    GM_ADDR z,
    const TensorGroupInfo&amp; info_z,
    int64_t availableCoreNum,
    aclrtStream stream)
{
    constexpr uint32_t blockDim = 8;

    const TensorInfo&amp; t0 = info_x.tensors[0];

    int64_t totalLength = 1;
    for (int64_t i = 0; i &lt; t0.numDims; ++i) {
        totalLength *= t0.shape[i];
    }

    // (8, 2048) 共 16384 个元素；
    // 启动 8 个核，每核一次处理 2048 个元素。
    SubCustomTilingData tiling = {
        static_cast&lt;uint32_t&gt;(totalLength),
        1
    };

    // dtype=1：float16；dtype=0：float32
    if (t0.dtype == 1) {
        sub_custom&lt;half&gt;&lt;&lt;&lt;blockDim, nullptr, stream&gt;&gt;&gt;(
            x, y, z, tiling);
    } else {
        sub_custom&lt;float&gt;&lt;&lt;&lt;blockDim, nullptr, stream&gt;&gt;&gt;(
            x, y, z, tiling);
    }
}
