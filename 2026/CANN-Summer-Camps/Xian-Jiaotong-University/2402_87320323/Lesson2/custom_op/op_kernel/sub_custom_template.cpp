#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

// ========== Kernel 类定义 ==========
template<typename T>
class KernelSub {
public:
    __aicore__ inline KernelSub() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
        uint32_t totalLength, uint32_t tileNum) {
        this->totalLength = totalLength;
        this->tileNum = tileNum;

        xGm = reinterpret_cast<T*>(x);
        yGm = reinterpret_cast<T*>(y);
        zGm = reinterpret_cast<T*>(z);

        pipe.InitBuffer(queX, BUFFER_NUM * TILE_SIZE * sizeof(T));
        pipe.InitBuffer(queY, BUFFER_NUM * TILE_SIZE * sizeof(T));
        pipe.InitBuffer(queZ, BUFFER_NUM * TILE_SIZE * sizeof(T));
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        DataCopy(queX.Alloc(), xGm + progress * TILE_SIZE, TILE_SIZE * sizeof(T));
        DataCopy(queY.Alloc(), yGm + progress * TILE_SIZE, TILE_SIZE * sizeof(T));
    }

    // 【关键修改点】加法 → 减法
    __aicore__ inline void Compute(int32_t progress) {
        Sub(queZ.Alloc(), queX.Get(), queY.Get(), TILE_SIZE * sizeof(T));
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        DataCopy(zGm + progress * TILE_SIZE, queZ.Get(), TILE_SIZE * sizeof(T));
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> queX, queY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> queZ;

    T* xGm;
    T* yGm;
    T* zGm;

    uint32_t totalLength;
    uint32_t tileNum;

    static constexpr uint32_t TILE_SIZE = 32;
    static constexpr uint32_t BUFFER_NUM = 2;
};

// ========== 核函数入口 ==========
extern "C" __global__ __aicore__ void sub_custom_template(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {

    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    // 使用 float 类型（作业要求支持 float16 和 float32）
    // 这里以 float 为例，如需支持 float16 可改为 half
    KernelSub<half> op;
    uint32_t tileNum = (tilingData.size + 31) / 32;  // 向上取整
    op.Init(x, y, z, tilingData.size, tileNum);
    op.Process();
}