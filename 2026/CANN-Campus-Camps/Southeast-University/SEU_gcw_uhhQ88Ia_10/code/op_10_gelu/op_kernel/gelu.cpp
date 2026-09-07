// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

namespace {
// 输入/输出队列深度(双缓冲)
constexpr int32_t BUFFER_NUM = 2;
// GELU精确式(与torch等erf基准一致): GELU(x) = 0.5*x*(1+erf(x/sqrt(2)))
// erf用Abramowitz-Stegun 7.1.26近似(|erf误差|<=1.5e-7):
//   erf(u) = sign(u)*(1 - tau*exp(-u^2)),  tau = t*(a1+t*(a2+t*(a3+t*(a4+t*a5))))
//   t = 1/(1+p*|u|),  u = x/sqrt(2)
// 为避免sign分支/Select, 折叠为全程只依赖|u|的形式:
//   y = x/2 + (|u| - |u|*tau*exp(-u^2))/sqrt(2),  u = x/sqrt(2)
// 倒数统一用Div(1, denom)实现(官方Reciprocal对half/float不满足双千/双万分之一)。
// 系数必须全精度fp32(half量化会破坏fp32双万分之一), fp16输入先升fp32计算再转回。
constexpr float INV_SQRT2 = 0.7071067811865476f; // 1/sqrt(2)
constexpr float P = 0.3275911f;                  // A&S 7.1.26 系数
constexpr float A1 = 0.254829592f;
constexpr float A2 = -0.284496736f;
constexpr float A3 = 1.421413741f;
constexpr float A4 = -1.453152027f;
constexpr float A5 = 1.061405429f;

// 判断模板实参是否为half, 用于编译期选择是否升fp32计算
template <typename T>
struct IsHalf {
    static constexpr bool value = false;
};
template <>
struct IsHalf<half> {
    static constexpr bool value = true;
};
}  // namespace

template <typename T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output,
                                const GeluTilingData &tiling) {
        uint32_t length = tiling.length;
        uint32_t blockDim = tiling.blockDim;
        tileLength_ = tiling.tileLength;

        // 按32B对齐单元(整块)切分, 保证每个核的GM起始偏移都是32B整数倍(DataCopy源/目的地址对齐):
        //   fp32下每块8个元素, fp16下每块16个元素
        // 完整32B块在核间均分; 末尾不足一个32B块的全局尾部元素追加给
        // "切片结束于数据末端"的那个核(起点天然32B对齐, 处于张量最末端)
        uint32_t coreIdx = GetBlockIdx();
        uint32_t alignUnit = 32U / sizeof(T);
        uint32_t blocks = length / alignUnit;           // 完整32B块数
        uint32_t tailElem = length - blocks * alignUnit;
        uint32_t perCoreBlock = blocks / blockDim;      // 每核整块数
        uint32_t remBlock = blocks % blockDim;          // 前remBlock个核额外多得一块
        uint32_t perCore = perCoreBlock * alignUnit;    // 每核主体元素数(32B整倍数)
        if (coreIdx < remBlock) {
            start_ = coreIdx * (perCore + alignUnit);
            count_ = perCore + alignUnit;
        } else {
            start_ = remBlock * (perCore + alignUnit) + (coreIdx - remBlock) * perCore;
            count_ = perCore;
        }
        if (tailElem > 0) {
            uint32_t tailCore =
                (perCoreBlock > 0) ? (blockDim - 1)
                                   : ((remBlock > 0) ? (remBlock - 1) : 0);
            if (coreIdx == tailCore) {
                count_ += tailElem;
            }
        }

        xGm_.SetGlobalBuffer((__gm__ T *)input_x, length);
        yGm_.SetGlobalBuffer((__gm__ T *)output, length);

        // 输入/输出队列按实际元素字长分配; 计算缓冲统一按fp32分配以支持half升精度
        uint32_t queueBytes = tileLength_ * sizeof(T);
        uint32_t calcBytes = tileLength_ * sizeof(float);
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, queueBytes);
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, queueBytes);
        pipe_.InitBuffer(tmpBuf1_, calcBytes);
        pipe_.InitBuffer(tmpBuf2_, calcBytes);
        pipe_.InitBuffer(tmpBuf3_, calcBytes);
        pipe_.InitBuffer(tmpBuf4_, calcBytes);
        aLocal_ = tmpBuf1_.Get<float>();
        bLocal_ = tmpBuf2_.Get<float>();
        cLocal_ = tmpBuf3_.Get<float>();
        dLocal_ = tmpBuf4_.Get<float>();
        if constexpr (IsHalf<T>::value) {
            // 仅half输入需要独立的fp32转换缓冲
            pipe_.InitBuffer(castBuf_, calcBytes);
            fxLocal_ = castBuf_.Get<float>();
        }
    }

    __aicore__ inline void Process() {
        if (count_ == 0) {
            return;
        }
        uint32_t fullTileNum = count_ / tileLength_;
        for (uint32_t i = 0; i < fullTileNum; i++) {
            CopyIn(i, tileLength_);
            Compute(tileLength_);
            CopyOut(i, tileLength_);
        }
        // 尾部不足一个tile的数据
        uint32_t tailLen = count_ - fullTileNum * tileLength_;
        if (tailLen > 0) {
            CopyIn(fullTileNum, tailLen);
            Compute(tailLen);
            CopyOut(fullTileNum, tailLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t tileIdx, uint32_t len) {
        LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
        DataCopy(xLocal, xGm_[start_ + tileIdx * tileLength_], len);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY_.AllocTensor<T>();

        LocalTensor<float> fx;
        if constexpr (IsHalf<T>::value) {
            // half输入先无损提升到fp32, 全链计算在fp32中进行
            Cast(fxLocal_, xLocal, RoundMode::CAST_NONE, len);
            fx = fxLocal_;
        } else {
            // fp32输入可直接使用输入缓冲
            fx = xLocal;
        }

        // 以下按erf基准的A&S近似折叠式逐步计算(全程fp32):
        //   u = x/sqrt(2); y = x/2 + (|u| - |u|*tau*exp(-u^2))/sqrt(2)
        // 缓冲角色:a=u/denom/tau/F/...  b=u^2->|u|  c=e^-u^2  d=ones/t
        Muls(aLocal_, fx, INV_SQRT2, len);     // a = u
        Mul(bLocal_, aLocal_, aLocal_, len);   // b = u^2
        Muls(bLocal_, bLocal_, -1.0f, len);    // b = -u^2
        Exp(cLocal_, bLocal_, len);            // c = exp(-u^2)
        Abs(bLocal_, aLocal_, len);            // b = |u| (a已不再使用)
        Muls(aLocal_, bLocal_, P, len);        // a = p*|u|
        Adds(aLocal_, aLocal_, 1.0f, len);     // a = 1 + p*|u|
        Duplicate(dLocal_, 1.0f, len);         // d = 1
        Div(dLocal_, dLocal_, aLocal_, len);   // d = t = 1/(1+p*|u|)
        Duplicate(aLocal_, A5, len);           // a = A5
        // tau = t*(A1+t*(A2+t*(A3+t*(A4+t*A5)))): 从A5起逐级嵌套
        Mul(aLocal_, aLocal_, dLocal_, len);
        Adds(aLocal_, aLocal_, A4, len);
        Mul(aLocal_, aLocal_, dLocal_, len);
        Adds(aLocal_, aLocal_, A3, len);
        Mul(aLocal_, aLocal_, dLocal_, len);
        Adds(aLocal_, aLocal_, A2, len);
        Mul(aLocal_, aLocal_, dLocal_, len);
        Adds(aLocal_, aLocal_, A1, len);
        Mul(aLocal_, aLocal_, dLocal_, len);   // a = tau (d不再使用)
        Mul(aLocal_, aLocal_, cLocal_, len);   // a = tau*exp(-u^2) (c不再使用)
        Mul(aLocal_, aLocal_, bLocal_, len);   // a = |u|*tau*exp(-u^2)
        Muls(aLocal_, aLocal_, -1.0f, len);    // a = -|u|*tau*exp(-u^2)
        Add(aLocal_, aLocal_, bLocal_, len);   // a = |u| - |u|*tau*exp(-u^2) (b仍持|u|, 之后释放)
        Muls(aLocal_, aLocal_, INV_SQRT2, len);  // a = (|u| - ...)/sqrt(2)
        Muls(cLocal_, fx, 0.5f, len);          // c = x/2
        if constexpr (IsHalf<T>::value) {
            // y(fp32) = x/2 + (|u| - ...)/sqrt(2), 再就近舍入回half
            Add(aLocal_, aLocal_, cLocal_, len);
            Cast(yLocal, aLocal_, RoundMode::CAST_RINT, len);
        } else {
            // fp32可直接写输出缓冲
            Add(yLocal, aLocal_, cLocal_, len);
        }

        inQueueX_.FreeTensor(xLocal);
        outQueueY_.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx, uint32_t len) {
        LocalTensor<T> yLocal = outQueueY_.DeQue<T>();
        DataCopy(yGm_[start_ + tileIdx * tileLength_], yLocal, len);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    TBuf<QuePosition::VECCALC> tmpBuf1_;
    TBuf<QuePosition::VECCALC> tmpBuf2_;
    TBuf<QuePosition::VECCALC> tmpBuf3_;
    TBuf<QuePosition::VECCALC> tmpBuf4_;
    TBuf<QuePosition::VECCALC> castBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    LocalTensor<float> aLocal_;
    LocalTensor<float> bLocal_;
    LocalTensor<float> cLocal_;
    LocalTensor<float> dLocal_;
    LocalTensor<float> fxLocal_;
    uint32_t tileLength_;
    uint32_t start_;
    uint32_t count_;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data);
    op.Process();
}
