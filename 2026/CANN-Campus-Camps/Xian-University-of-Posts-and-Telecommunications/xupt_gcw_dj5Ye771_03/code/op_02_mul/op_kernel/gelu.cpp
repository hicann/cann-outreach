// Kernel侧核函数实现（单块化：整块一次处理，放不下 UB 则回退分块）
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"
constexpr int32_t BUFFER_NUM = 1;         // 单块化下用单缓冲（in+out+tmp1+tmp2 = 4 缓冲，省 UB 给整块）
constexpr uint32_t TILE_LEN = 128;        // 回退分块的块长（half/float 均 32 字节对齐）
constexpr uint32_t MAX_UB_BYTES = 196608; // 单核 UB 预算（4 缓冲合计 192KB，UB 为 256KB）

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, uint32_t length) {
        uint32_t blockNum = AscendC::GetBlockNum();       // 启动核数（host 固定 8）
        uint32_t coreId = AscendC::GetBlockIdx();
        uint32_t base = length / blockNum;                      // 每核基础长度
        uint32_t alignedBlock = (base / TILE_LEN) * TILE_LEN;   // 每核对齐长度（TILE_LEN 整数倍）
        this->startIdx = coreId * alignedBlock;                 // 每核起始下标（32 字节对齐）
        if (coreId == blockNum - 1) {
            // 最后一个核处理全部余量（其余核 blockLength 均为 TILE_LEN 的倍数，无尾块）
            this->blockLength = length - this->startIdx;
        } else {
            this->blockLength = alignedBlock;
        }
        if (this->blockLength < 0) {
            this->blockLength = 0;
        }
        // 单块化决策：本核对齐部分能放 UB 则整块一次搬入搬出，否则回退 TILE_LEN 分块
        uint32_t alignedLen = (this->blockLength / TILE_LEN) * TILE_LEN;
        uint32_t chunk = alignedLen;
        if (chunk == 0 || chunk * static_cast<uint32_t>(sizeof(DT_INPUT_X)) * 4 > MAX_UB_BYTES) {
            chunk = TILE_LEN;
        }
        this->alignedLen = alignedLen;
        this->chunkLen = chunk;
        xGm.SetGlobalBuffer((__gm__ DT_INPUT_X*)input_x + this->startIdx, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_INPUT_X*)output + this->startIdx, this->blockLength);
        pipe.InitBuffer(inQueue, BUFFER_NUM, chunkLen * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outQueue, BUFFER_NUM, chunkLen * sizeof(DT_INPUT_X));
        pipe.InitBuffer(tmpBuf1, chunkLen * sizeof(DT_INPUT_X));
        pipe.InitBuffer(tmpBuf2, chunkLen * sizeof(DT_INPUT_X));
    }

    __aicore__ inline void Process() {
        uint32_t mainCount = this->alignedLen / this->chunkLen; // 对齐部分的块数
        for (uint32_t i = 0; i < mainCount; i++) {
            CopyIn(i * this->chunkLen, this->chunkLen);
            Compute(this->chunkLen);
            CopyOut(i * this->chunkLen, this->chunkLen);
        }
        uint32_t tail = this->blockLength - this->alignedLen; // 非对齐残余（< TILE_LEN）
        if (tail > 0) {
            // 尾块：仅最后一个核可能出现。按 TILE_LEN 整块读取/写出，越界落在 GM 对齐填充区，安全。
            CopyIn(this->alignedLen, TILE_LEN);
            Compute(TILE_LEN);
            CopyOut(this->alignedLen, TILE_LEN);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t cnt) {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueue.AllocTensor<DT_INPUT_X>();
        AscendC::DataCopy(xLocal, xGm[offset], cnt);
        inQueue.EnQue(xLocal);
    }
    __aicore__ inline void Compute(uint32_t cnt) {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueue.DeQue<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> zLocal = outQueue.AllocTensor<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> t1 = tmpBuf1.Get<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> t2 = tmpBuf2.Get<DT_INPUT_X>();
        // gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        AscendC::Muls(t1, xLocal, (DT_INPUT_X)0.7071067811865476f, cnt); // x/sqrt(2)
        AscendC::Erf(t2, t1, cnt);                                       // erf(x/sqrt(2))
        AscendC::Adds(t2, t2, (DT_INPUT_X)1.0f, cnt);                    // 1+erf(x/sqrt(2))
        AscendC::Muls(t2, t2, (DT_INPUT_X)0.5f, cnt);                    // 0.5*(1+erf)
        AscendC::Mul(zLocal, xLocal, t2, cnt);                           // x*0.5*(1+erf)
        outQueue.EnQue<DT_INPUT_X>(zLocal);
        inQueue.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t cnt) {
        AscendC::LocalTensor<DT_INPUT_X> zLocal = outQueue.DeQue<DT_INPUT_X>();
        AscendC::DataCopy(zGm[offset], zLocal, cnt);
        outQueue.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2;
    AscendC::GlobalTensor<DT_INPUT_X> xGm;
    AscendC::GlobalTensor<DT_INPUT_X> zGm;
    uint32_t blockLength; // 本核实际长度
    uint32_t alignedLen;  // blockLength 对齐到 TILE_LEN 的部分
    uint32_t chunkLen;    // 每次处理的块长（能单块则整块，否则 TILE_LEN）
    uint32_t startIdx;    // 本核起始下标
};

template <typename DT_INPUT_X>
 __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data.length);
    op.Process();
}
