#include "div_custom_template_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;       // 双缓冲：搬运与计算并行
constexpr uint32_t DTYPE_FP16 = 0;      // float16 标记
constexpr uint32_t DTYPE_FP32 = 1;      // float32 标记

// 矢量除法核函数类：z = x / y
template <typename T>
class KernelDiv {
public:
    __aicore__ inline KernelDiv() {}

    // 初始化：按核切分数据，设置 GM 指针，申请 UB 队列
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileLen) {
        uint32_t blockLength = totalLength / GetBlockNum();
        this->tileLength = tileLen;
        this->tileNum = blockLength / tileLen;

        // GlobalTensor 管理 GM 数据，按 block idx 偏移到本核负责的数据段
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + GetBlockIdx() * blockLength, blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + GetBlockIdx() * blockLength, blockLength);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(z) + GetBlockIdx() * blockLength, blockLength);

        // 申请 UB 队列：每个队列 2 块 UB，每块 tileLength * sizeof(T) 字节
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(T));
    }

    // 主流程：按 tile 循环执行 CopyIn -> Compute -> CopyOut
    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < tileNum; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // 搬入：将第 tileIndex 个 tile 的 x/y 数据从 GM 拷贝到 UB
    __aicore__ inline void CopyIn(uint32_t tileIndex) {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        DataCopy(xLocal, xGm[tileIndex * tileLength], tileLength);
        DataCopy(yLocal, yGm[tileIndex * tileLength], tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // 计算：出队 x/y，执行逐元素除法 z = x / y
    __aicore__ inline void Compute(uint32_t tileIndex) {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = inQueueY.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        Div(zLocal, xLocal, yLocal, tileLength);

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // 搬出：将结果 z 从 UB 拷贝回 GM
    __aicore__ inline void CopyOut(uint32_t tileIndex) {
        LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        DataCopy(zGm[tileIndex * tileLength], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;                                          // 流水管理器
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;       // 输入 x 队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;       // 输入 y 队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;     // 输出 z 队列
    GlobalTensor<T> xGm, yGm, zGm;                       // GM 全局数据指针
    uint32_t tileLength = 0;                             // 单 tile 长度
    uint32_t tileNum = 0;                                // 本核 tile 循环次数
};

extern "C" __global__ __aicore__ void div_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                          GM_ADDR workspace, GM_ADDR tiling) {
    // 从 GM 中解析 host 侧写入的 tiling 参数
    GET_TILING_DATA(tilingData, tiling);

    // 根据数据类型实例化对应模板并执行
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