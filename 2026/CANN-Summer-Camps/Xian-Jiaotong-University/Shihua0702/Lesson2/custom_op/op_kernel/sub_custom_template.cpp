#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    // 获取总元素数（由 TilingFunc 计算的 data_sz）
    uint32_t totalSize = tilingData.size;

    // 将数据平均分配给 8 个 AICore
    uint32_t blockSize = totalSize / GetBlockNum();
    uint32_t start = GetBlockIdx() * blockSize;
    uint32_t len = (GetBlockIdx() == GetBlockNum() - 1) ?
                   (totalSize - start) : blockSize;
    if (len == 0) return;

    // 构造 GlobalTensor 用于数据拷贝
    GlobalTensor<float16_t> xGm, yGm, zGm;
    xGm.SetGlobalBuffer((__gm__ float16_t*)x + start, len);
    yGm.SetGlobalBuffer((__gm__ float16_t*)y + start, len);
    zGm.SetGlobalBuffer((__gm__ float16_t*)z + start, len);

    // 初始化 Pipe 并分配 Local Memory
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufX, bufY, bufZ;
    bufX = pipe.InitBuffer(len * sizeof(float16_t));
    bufY = pipe.InitBuffer(len * sizeof(float16_t));
    bufZ = pipe.InitBuffer(len * sizeof(float16_t));

    LocalTensor<float16_t> xLocal = bufX.Get<float16_t>(len);
    LocalTensor<float16_t> yLocal = bufY.Get<float16_t>(len);
    LocalTensor<float16_t> zLocal = bufZ.Get<float16_t>(len);

    // GM → LM：搬运输入数据到 Local Memory
    DataCopy(xLocal, xGm, len);
    DataCopy(yLocal, yGm, len);

    // 向量计算：z = x - y（逐元素减法）
    Sub(zLocal, xLocal, yLocal, len);

    // LM → GM：将结果写回 Global Memory
    DataCopy(zGm, zLocal, len);
}
