/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * gcd123 算子 kernel 侧实现（Ascend C 编程框架）：
 *   out = gcd(self, other)
 *
 * 算子规格：
 *   - 2 输入 self / other，1 输出 out，均为 fp16、ND 格式、4 维 shape [N4,N3,N2,N1]
 *   - self 与 other 需满足 broadcast 关系；out 的 shape 为 broadcast 后的 shape
 *
 * 实现要点：
 * 1. Broadcast 搬运（无需 Gather 指令，只用 DataCopy + Duplicate）：
 *    - 将输出按"行对齐 tile"或"行内切块"划分；每个 tile 的起点坐标由标量单元维护
 *      （i4/i3/i2/i1），输入偏移 = 坐标点乘该输入的有效 stride（广播维 stride 为 0）。
 *    - 行对齐模式下输入行模式只有 4 种：
 *        a. s3==1 且 s2==N1：整段连续拷贝（无广播时的快路径）；
 *        b. s3==1 且 s2==0 ：整行重复（dim2 广播）；
 *        c. s3==0 且 s2==0 ：整个 tile 为同一元素（标量广播）；
 *        d. s3==0 且 s2==1 ：逐行取单值后 Duplicate 填充（dim3 广播）。
 *    - UB 中每行按 32B 对齐存放（pitch = AlignUp(N1,16)），保证 Duplicate/矢量指令
 *      操作数地址满足 32 字节对齐约束；搬出时逐行写回 GM 的连续地址。
 * 2. GCD 计算（固定 64 次迭代的欧几里得算法，无数据相关分支，全矢量实现）：
 *    a = |self|, b = |other|
 *    每次迭代：q = trunc(a / b)，r = a - b*q（r 经符号/越界修正保证落在 [0, b)），
 *              (a, b) <- (b==0 ? a : b, b==0 ? 0 : r)
 *    收敛后 a 即 gcd 结果。对输入为整数且不超过 fp16 精确表示范围（|x| <= 2048）的
 *    数据，结果精确；b == 0 时通过 mask 保护（safeB = b==0 ? 1 : b）避免除零。
 */
#include "kernel_operator.h"
using namespace AscendC;

// 直接调用（kernel_invocation）模式：使用 POD 结构体自行解析 tiling。
// 算子工程（aclnn）模式：不定义 GCD123_DIRECT_INVOCATION，tiling 由工具链注册机制提供。
#ifdef GCD123_DIRECT_INVOCATION
#include "kernel_invocation/gcd123_tiling_direct.h"
#endif

// 与 op_host/gcd123.cpp 中保持一致：UB 中单个 tile 的最大元素数（fp16）
constexpr uint32_t CHUNK = 2048;

class KernelGcd123 {
public:
    __aicore__ inline KernelGcd123() {}

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR out,
        uint32_t totalLength, uint32_t coreLen, uint32_t tileSize, uint32_t rowsPerTile,
        uint32_t pitch, uint32_t maxIter,
        uint32_t n3, uint32_t n2, uint32_t n1,
        uint32_t selfS0, uint32_t selfS1, uint32_t selfS2, uint32_t selfS3,
        uint32_t otherS0, uint32_t otherS1, uint32_t otherS2, uint32_t otherS3)
    {
        this->totalLength = totalLength;
        this->coreLen = coreLen;
        this->tileSize = tileSize;
        this->rowsPerTile = rowsPerTile;
        this->pitch = pitch;
        this->maxIter = maxIter;
        this->n3 = n3;
        this->n2 = n2;
        this->n1 = n1;
        this->selfS0 = selfS0;
        this->selfS1 = selfS1;
        this->selfS2 = selfS2;
        this->selfS3 = selfS3;
        this->otherS0 = otherS0;
        this->otherS1 = otherS1;
        this->otherS2 = otherS2;
        this->otherS3 = otherS3;

        selfGm.SetGlobalBuffer((__gm__ half*)self, totalLength);
        otherGm.SetGlobalBuffer((__gm__ half*)other, totalLength);
        outGm.SetGlobalBuffer((__gm__ half*)out, totalLength);

        pipe.InitBuffer(selfBuf, tileSize * sizeof(half));
        pipe.InitBuffer(otherBuf, tileSize * sizeof(half));
        pipe.InitBuffer(aBuf0, tileSize * sizeof(half));
        pipe.InitBuffer(bBuf0, tileSize * sizeof(half));
        pipe.InitBuffer(aBuf1, tileSize * sizeof(half));
        pipe.InitBuffer(bBuf1, tileSize * sizeof(half));
        pipe.InitBuffer(qBuf, tileSize * sizeof(half));
        pipe.InitBuffer(tBuf, tileSize * sizeof(half));
        pipe.InitBuffer(rBuf, tileSize * sizeof(half));
        pipe.InitBuffer(rPlusBBuf, tileSize * sizeof(half));
        pipe.InitBuffer(rMinusBBuf, tileSize * sizeof(half));
        pipe.InitBuffer(safeBBuf, tileSize * sizeof(half));
        pipe.InitBuffer(rowValsBuf, tileSize * sizeof(half));
        pipe.InitBuffer(zeroBuf, tileSize * sizeof(half));
        pipe.InitBuffer(oneBuf, tileSize * sizeof(half));
        pipe.InitBuffer(qIntBuf, tileSize * sizeof(int32_t));
        pipe.InitBuffer(maskZeroBuf, tileSize * sizeof(uint8_t));
        pipe.InitBuffer(maskNegBuf, tileSize * sizeof(uint8_t));
        pipe.InitBuffer(maskGeBuf, tileSize * sizeof(uint8_t));
    }

    __aicore__ inline void Process()
    {
        uint32_t blockIdx = GetBlockIdx();
        uint64_t coreStart = static_cast<uint64_t>(blockIdx) * coreLen;
        uint32_t remaining = 0;
        if (coreStart < totalLength) {
            uint64_t end = coreStart + coreLen;
            if (end > totalLength) {
                end = totalLength;
            }
            remaining = static_cast<uint32_t>(end - coreStart);
        }
        if (remaining == 0) {
            return;
        }

        // 将核内起始元素偏移分解为 4 维坐标（标量单元执行，精确）
        uint32_t i1 = static_cast<uint32_t>(coreStart % n1);
        uint32_t t = static_cast<uint32_t>(coreStart / n1);
        uint32_t i2 = t % n2;
        t /= n2;
        uint32_t i3 = t % n3;
        uint32_t i4 = t / n3;
        uint32_t outOffset = static_cast<uint32_t>(coreStart);

        // 常用标量：0 / 1，填充常量 buffer 供矢量指令使用
        half zero = static_cast<half>(0.0f);
        half one = static_cast<half>(1.0f);
        Duplicate(zeroBuf.Get<half>(), zero, static_cast<int32_t>(tileSize));
        Duplicate(oneBuf.Get<half>(), one, static_cast<int32_t>(tileSize));

        while (remaining > 0) {
            if (n1 < CHUNK && remaining >= n1) {
                // ---------- 行对齐模式：整行切块 ----------
                uint32_t rows = rowsPerTile;
                uint32_t maxRows = n2 - i2;
                if (rows > maxRows) {
                    rows = maxRows;
                }
                uint32_t maxRowsByRemain = remaining / n1;
                if (rows > maxRowsByRemain) {
                    rows = maxRowsByRemain;
                }
                uint32_t T = rows * n1;     // 真实元素数
                uint32_t Tp = rows * pitch; // UB 中含行尾填充的元素数

                LoadInputAligned(selfGm, selfBuf.Get<half>(), selfS0, selfS1, selfS2, selfS3,
                    i2, i3, i4, rows, T, Tp);
                LoadInputAligned(otherGm, otherBuf.Get<half>(), otherS0, otherS1, otherS2, otherS3,
                    i2, i3, i4, rows, T, Tp);
                LocalTensor<half> res = GcdCompute(selfBuf.Get<half>(), otherBuf.Get<half>(), Tp);
                for (uint32_t r = 0; r < rows; r++) {
                    DataCopy(outGm[outOffset + r * n1], res[r * pitch], n1);
                }

                remaining -= T;
                outOffset += T;
                i2 += rows;
                if (i2 >= n2) {
                    i2 = 0;
                    i3++;
                    if (i3 >= n3) {
                        i3 = 0;
                        i4++;
                    }
                }
            } else {
                // ---------- 行内切块模式：N1 较大或尾部不足一行时 ----------
                uint32_t T = remaining;
                if (T > CHUNK) {
                    T = CHUNK;
                }
                uint32_t inRowMax = n1 - i1;
                if (T > inRowMax) {
                    T = inRowMax;
                }

                LoadInputInRow(selfGm, selfBuf.Get<half>(), selfS0, selfS1, selfS2, selfS3,
                    i1, i2, i3, i4, T);
                LoadInputInRow(otherGm, otherBuf.Get<half>(), otherS0, otherS1, otherS2, otherS3,
                    i1, i2, i3, i4, T);
                LocalTensor<half> res = GcdCompute(selfBuf.Get<half>(), otherBuf.Get<half>(), T);
                DataCopy(outGm[outOffset], res, T);

                remaining -= T;
                outOffset += T;
                i1 += T;
                if (i1 >= n1) {
                    i1 = 0;
                    i2++;
                    if (i2 >= n2) {
                        i2 = 0;
                        i3++;
                        if (i3 >= n3) {
                            i3 = 0;
                            i4++;
                        }
                    }
                }
            }
        }
    }

private:
    // 行对齐模式搬入：tile 为 rows 行完整行，行起点坐标为 (i4, i3, i2, 0)。
    // 有效 stride 规则：广播维 stride 为 0；由于 s3 必为 0 或 1、s2 必为 0 或 1 或 N1，
    // 行模式只存在以下 4 种组合。
    __aicore__ inline void LoadInputAligned(const GlobalTensor<half>& xGm, const LocalTensor<half>& xBuf,
        uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
        uint32_t i2, uint32_t i3, uint32_t i4, uint32_t rows, uint32_t T, uint32_t Tp)
    {
        uint32_t base = i4 * s0 + i3 * s1 + i2 * s2;
        if (s3 == 1) {
            // 最内维不广播：每行在源中是连续的 N1 个元素
            if (s2 == n1) {
                // dim2 也不广播：源数据整段连续
                if (pitch == n1) {
                    DataCopy(xBuf, xGm[base], T); // N1 为 16 对齐时的快路径
                } else {
                    for (uint32_t r = 0; r < rows; r++) {
                        DataCopy(xBuf[r * pitch], xGm[base + r * n1], n1);
                    }
                }
            } else {
                // dim2 广播（s2 == 0）：所有行读取同一段源行，拷贝一次后逐行复制
                DataCopy(xBuf, xGm[base], n1);
                for (uint32_t r = 1; r < rows; r++) {
                    DataCopy(xBuf[r * pitch], xBuf, n1);
                }
            }
        } else {
            // 最内维广播（s3 == 0）：每行只需一个元素
            if (s2 == 0) {
                // dim2 也广播：整个 tile 为同一元素
                half v = xGm.GetValue(base);
                Duplicate(xBuf, v, static_cast<int32_t>(Tp));
            } else {
                // dim2 不广播（s2 == 1）：各行取值在源中连续，先搬入行值向量再逐行填充
                DataCopy(rowValsBuf.Get<half>(), xGm[base], rows);
                for (uint32_t r = 0; r < rows; r++) {
                    Duplicate(xBuf[r * pitch], rowValsBuf.Get<half>().GetValue(r),
                        static_cast<int32_t>(n1));
                }
            }
        }
    }

    // 行内切块模式搬入：tile 位于同一行内，起点坐标含 i1 偏移。
    __aicore__ inline void LoadInputInRow(const GlobalTensor<half>& xGm, const LocalTensor<half>& xBuf,
        uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
        uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4, uint32_t T)
    {
        uint32_t base = i4 * s0 + i3 * s1 + i2 * s2 + i1 * s3;
        if (s3 == 1) {
            DataCopy(xBuf, xGm[base], T);
        } else {
            half v = xGm.GetValue(base);
            Duplicate(xBuf, v, static_cast<int32_t>(T));
        }
    }

    // 矢量 GCD：a = |self|, b = |other|，固定迭代欧几里得算法。
    // 返回值：存放结果的 LocalTensor（ping-pong 缓冲之一），元素个数为入参 count。
    __aicore__ inline LocalTensor<half> GcdCompute(const LocalTensor<half>& selfLocal,
        const LocalTensor<half>& otherLocal, uint32_t count)
    {
        LocalTensor<half> a0 = aBuf0.Get<half>();
        LocalTensor<half> b0 = bBuf0.Get<half>();
        LocalTensor<half> a1 = aBuf1.Get<half>();
        LocalTensor<half> b1 = bBuf1.Get<half>();
        LocalTensor<half> q = qBuf.Get<half>();
        LocalTensor<int32_t> qInt = qIntBuf.Get<int32_t>();
        LocalTensor<half> tmp = tBuf.Get<half>();
        LocalTensor<half> r = rBuf.Get<half>();
        LocalTensor<half> rPlusB = rPlusBBuf.Get<half>();
        LocalTensor<half> rMinusB = rMinusBBuf.Get<half>();
        LocalTensor<half> safeB = safeBBuf.Get<half>();
        LocalTensor<uint8_t> maskZero = maskZeroBuf.Get<uint8_t>();
        LocalTensor<uint8_t> maskNeg = maskNegBuf.Get<uint8_t>();
        LocalTensor<uint8_t> maskGe = maskGeBuf.Get<uint8_t>();

        Abs(a0, selfLocal, count);
        Abs(b0, otherLocal, count);
        LocalTensor<half> aCur = a0;
        LocalTensor<half> bCur = b0;
        LocalTensor<half> aNxt = a1;
        LocalTensor<half> bNxt = b1;

        for (uint32_t it = 0; it < maxIter; it++) {
            // safeB = (b == 0) ? 1 : b，防止除零
            Compare(maskZero, bCur, zeroBuf.Get<half>(), CMPMODE::EQ, count);
            Select(safeB, maskZero, oneBuf.Get<half>(), bCur, count);
            // q = trunc(a / b)：先矢量除，再 Cast 截断为 int32，最后转回 half
            Div(q, aCur, safeB, count);
            Cast(qInt, q, RoundMode::CAST_TRUNC, count);
            Cast(q, qInt, RoundMode::CAST_NONE, count);
            // r = a - b * q，随后修正到 [0, b)：r < 0 时 +b，r >= b 时 -b
            Mul(tmp, bCur, q, count);
            Sub(r, aCur, tmp, count);
            Compare(maskNeg, r, zeroBuf.Get<half>(), CMPMODE::LT, count);
            Add(rPlusB, r, bCur, count);
            Select(tmp, maskNeg, rPlusB, r, count);
            Compare(maskGe, tmp, bCur, CMPMODE::GE, count);
            Sub(rMinusB, tmp, bCur, count);
            Select(r, maskGe, rMinusB, tmp, count);
            // (a, b) <- (b == 0 ? a : b, b == 0 ? 0 : r)
            Select(aNxt, maskZero, aCur, bCur, count);
            Select(bNxt, maskZero, zeroBuf.Get<half>(), r, count);

            LocalTensor<half> swap = aCur;
            aCur = aNxt;
            aNxt = swap;
            swap = bCur;
            bCur = bNxt;
            bNxt = swap;
        }
        return aCur;
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> selfBuf, otherBuf;
    TBuf<TPosition::VECCALC> aBuf0, bBuf0, aBuf1, bBuf1;
    TBuf<TPosition::VECCALC> qBuf, tBuf, rBuf, rPlusBBuf, rMinusBBuf, safeBBuf;
    TBuf<TPosition::VECCALC> rowValsBuf, zeroBuf, oneBuf;
    TBuf<TPosition::VECCALC> qIntBuf;
    TBuf<TPosition::VECCALC> maskZeroBuf, maskNegBuf, maskGeBuf;
    GlobalTensor<half> selfGm, otherGm, outGm;

    uint32_t totalLength;
    uint32_t coreLen;
    uint32_t tileSize;
    uint32_t rowsPerTile;
    uint32_t pitch;
    uint32_t maxIter;
    uint32_t n3;
    uint32_t n2;
    uint32_t n1;
    uint32_t selfS0, selfS1, selfS2, selfS3;
    uint32_t otherS0, otherS1, otherS2, otherS3;
};

// 算子核函数入口：参数顺序固定为 输入、输出、workspace、tiling，勿调整。
// 函数名由算子工程约定为 <op小写名>_custom。
extern "C" __global__ __aicore__ void gcd123_custom(GM_ADDR self, GM_ADDR other, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    KernelGcd123 op;
    op.Init(self, other, out,
        tilingData.totalLength, tilingData.coreLen, tilingData.tileSize,
        tilingData.rowsPerTile, tilingData.pitch, tilingData.maxIter,
        tilingData.n3, tilingData.n2, tilingData.n1,
        tilingData.selfS0, tilingData.selfS1, tilingData.selfS2, tilingData.selfS3,
        tilingData.otherS0, tilingData.otherS1, tilingData.otherS2, tilingData.otherS3);
#if defined(GCD123_DIRECT_INVOCATION)
    op.Process();
#else
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
#endif
}

// 直接调用（kernel_invocation）模式的 host 侧启动函数（NPU 模式）
#if defined(GCD123_DIRECT_INVOCATION) && !defined(__CCE_KT_TEST__) && !defined(ASCENDC_CPU_DEBUG)
void gcd123_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* self, uint8_t* other,
    uint8_t* out, uint8_t* workspace, uint8_t* tiling)
{
    gcd123_custom<<<blockDim, l2ctrl, stream>>>(self, other, out, workspace, tiling);
}
#endif
