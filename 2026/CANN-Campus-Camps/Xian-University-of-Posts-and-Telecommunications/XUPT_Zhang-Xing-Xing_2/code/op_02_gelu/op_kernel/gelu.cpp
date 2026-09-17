// ============================================================================
// op_02 Gelu 激活算子 —— Kernel 侧核函数实现（补全 class KernelGelu）
// 功能：y = GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))，x / y 均为 fp32/fp16，ND
//
// v16 修复（对照他人通过的参考代码）：
//   1. 【核心】核间分配改为 32 元素对齐块分配（大核/小核）：
//      连续均分时每核 start = blockIdx*perCore 可能不是 32 倍数，导致 GM 偏移
//      未 32 字节对齐，DataCopy 结果错乱（v15 点1 0.38%、点2 3.85%）。
//      块分配保证每核起始偏移始终是 32 的倍数，GM 地址严格对齐。
//   2. 计算长度直接用实际元素数（curLength），AscendC 内置接口内部处理不完整块。
//   3. 全类型统一精确 erf 公式（fp16 也直接算，与 golden 一致）。
// ============================================================================
#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr int32_t BUFFER_NUM = 2;   // 双缓冲
constexpr uint32_t kBlockSize = 32; // 核间分配的基本块大小（元素数）

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    /**
     * @brief 初始化：解析 tiling，按 32 元素块切分本核区间，分配 UB（按 tile 大小）
     */
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, GM_ADDR tiling)
    {
        GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
        this->totalLength = tiling_data.length;
        this->tileLength = tiling_data.tileLength;
        uint32_t blockDim = tiling_data.blockDim;

        // 本核区间：32 元素对齐块分配（前 extra 个核为大核，各多处理一个块）。
        // 这样每个核的 GM 起始偏移始终是 32 的倍数（32 字节对齐），保证 DataCopy 正确。
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t alignedLength = ((this->totalLength + kBlockSize - 1) / kBlockSize) * kBlockSize;
        uint32_t blockCount = alignedLength / kBlockSize;
        uint32_t base = blockCount / blockDim;
        uint32_t extra = blockCount % blockDim;
        uint32_t blocksThisCore = (blockIdx < extra) ? (base + 1) : base;
        uint32_t startBlock = blockIdx * base + ((blockIdx < extra) ? blockIdx : extra);
        uint32_t start = startBlock * kBlockSize;  // 元素偏移，恒为 32 倍数
        this->coreDataNum = blocksThisCore * kBlockSize;
        // 最后一个核可能超出真实总长（对齐后多出的部分），截断
        if (start + this->coreDataNum > this->totalLength) {
            this->coreDataNum = this->totalLength - start;
        }

        // 本核需要处理的 tile 数
        this->tileNum = (this->coreDataNum + this->tileLength - 1) / this->tileLength;

        // 绑定 GM（本核全区间，后续用 [t*tileLength] 定位每个 tile）
        xGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x + start, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)output + start, this->coreDataNum);

        // 分配 UB：输入/输出队列（双缓冲）。全部类型统一用原位 erf 公式
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_INPUT_X));
    }

    /**
     * @brief 流水线总调度：循环处理本核所有 tile（CopyIn -> Compute -> CopyOut）
     */
    __aicore__ inline void Process()
    {
        if (this->coreDataNum == 0) {
            return;  // 空核保护
        }
        for (uint32_t t = 0; t < this->tileNum; t++) {
            // 本 tile 实际元素数（最后一个 tile 可能不满）
            this->curLength = (t == this->tileNum - 1)
                                  ? (this->coreDataNum - t * this->tileLength)
                                  : this->tileLength;

            CopyIn(t);
            Compute(t);
            CopyOut(t);
        }
    }

private:
    /**
     * @brief CopyIn：把本 tile 的 x 从 GM 搬入 UB 输入队列
     */
    __aicore__ inline void CopyIn(int32_t tileIdx)
    {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueueX.AllocTensor<DT_INPUT_X>();
        AscendC::DataCopy(xLocal, xGm[tileIdx * this->tileLength], this->curLength);
        inQueueX.EnQue(xLocal);
    }

    /**
     * @brief Compute：按精确 erf 公式计算 GELU，全类型统一（原位计算，参考通过版）
     *   GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
     */
    __aicore__ inline void Compute(int32_t tileIdx)
    {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueueX.DeQue<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> yLocal = outQueueY.AllocTensor<DT_INPUT_X>();

        // 分五步在 yLocal 上原位计算（AscendC 内置接口支持 in-place）：
        // 注意：标量一律用 static_cast<DT_INPUT_X>（C 风格转换在 half 类类型下编译不过）
        AscendC::Muls(yLocal, xLocal, static_cast<DT_INPUT_X>(0.70710678f), this->curLength);  // x / sqrt(2)
        AscendC::Erf(yLocal, yLocal, this->curLength);                                         // erf(x / sqrt(2))
        AscendC::Adds(yLocal, yLocal, static_cast<DT_INPUT_X>(1.0f), this->curLength);         // 1 + erf(...)
        AscendC::Muls(yLocal, yLocal, static_cast<DT_INPUT_X>(0.5f), this->curLength);         // 0.5 * (1 + erf(...))
        AscendC::Mul(yLocal, yLocal, xLocal, this->curLength);                                 // 0.5 * x * (1 + erf(...))

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    /**
     * @brief CopyOut：把本 tile 计算结果从 UB 输出队列写回 GM
     */
    __aicore__ inline void CopyOut(int32_t tileIdx)
    {
        AscendC::LocalTensor<DT_INPUT_X> yLocal = outQueueY.DeQue<DT_INPUT_X>();
        AscendC::DataCopy(yGm[tileIdx * this->tileLength], yLocal, this->curLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;                                              // 流水线内存管理器
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;  // 输入 x 向量队列
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY; // 输出 y 向量队列
    AscendC::GlobalTensor<DT_INPUT_X> xGm, yGm;                      // GM 全局内存张量
    uint32_t totalLength;  // 总元素数
    uint32_t tileLength;   // 每 tile 元素数
    uint32_t tileNum;      // 本核 tile 数
    uint32_t coreDataNum;  // 本核实际处理元素数
    uint32_t curLength;    // 当前 tile 实际元素数
};

// ---------------------------------------------------------------------------
// 核函数入口（OpType=Gelu 的 kernel 入口，与原型声明一致）
// ---------------------------------------------------------------------------
template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output,
                                GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling);
    op.Process();
}