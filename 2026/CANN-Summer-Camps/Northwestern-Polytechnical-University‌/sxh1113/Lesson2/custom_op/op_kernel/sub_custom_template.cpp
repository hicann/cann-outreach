
#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"
#include "ascendc/ascendc_api.h"

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    
    constexpr uint32_t BLOCK_SIZE = 128;
    // 修复1：结构体字段名改为 size
    uint32_t totalElem = tilingData.size;
    uint32_t loopCnt = totalElem / BLOCK_SIZE;
    uint32_t remainElem = totalElem % BLOCK_SIZE;

    // 修复2：显式分配本地张量内存
    LocalTensor<half> xLocal, yLocal, zLocal;
    xLocal.InitBuffer(BLOCK_SIZE);
    yLocal.InitBuffer(BLOCK_SIZE);
    zLocal.InitBuffer(BLOCK_SIZE);

    uint8_t *inPtrX = reinterpret_cast<uint8_t *>(x);
    uint8_t *inPtrY = reinterpret_cast<uint8_t *>(y);
    uint8_t *outPtrZ = reinterpret_cast<uint8_t *>(z);

    for (uint32_t i = 0; i < loopCnt; ++i) {
        // 修复3：替换为昇腾标准 DataCopy 搬运接口
        AscendC::DataCopy(xLocal, inPtrX, BLOCK_SIZE);
        AscendC::DataCopy(yLocal, inPtrY, BLOCK_SIZE);
        // 修复4：替换为昇腾标准向量减法接口 Sub
        AscendC::Sub(zLocal, xLocal, yLocal, BLOCK_SIZE);
        AscendC::DataCopy(outPtrZ, zLocal, BLOCK_SIZE);

        inPtrX += BLOCK_SIZE * sizeof(half);
        inPtrY += BLOCK_SIZE * sizeof(half);
        outPtrZ += BLOCK_SIZE * sizeof(half);
    }

    if (remainElem > 0) {
        AscendC::DataCopy(xLocal, inPtrX, remainElem);
        AscendC::DataCopy(yLocal, inPtrY, remainElem);
        AscendC::Sub(zLocal, xLocal, yLocal, remainElem);
        AscendC::DataCopy(outPtrZ, zLocal, remainElem);
    }
}
// 修复5：删除文件末尾多余的 }