/*
 * add_custom 算子 - host 侧调用示例（CPU 调试版，ICPU_RUN_KF）
 * 输入：两个 float16 二维张量 [N2, N1]
 * 输出：z = x + y
 */
#include "data_utils.h"
#include "add_custom_tiling.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                 GM_ADDR workspace, GM_ADDR tiling);

// 生成 tiling 数据
static uint8_t *GenerateTiling(uint32_t totalLength, uint32_t tileLength, uint32_t &tilingSize)
{
    tilingSize = sizeof(AddCustomTilingData);
    auto *tiling = (AddCustomTilingData *)malloc(tilingSize);
    if (tiling == nullptr) {
        return nullptr;
    }
    tiling->totalLength = totalLength;
    tiling->tileLength = tileLength;
    return (uint8_t *)tiling;
}

int32_t main(int32_t argc, char *argv[])
{
    // 二维 shape [N2, N1]
    const uint32_t N2 = 8;
    const uint32_t N1 = 16;
    const uint32_t blockDim = 1;   // 单核调用（CPU 调试）
    const uint32_t tileLength = 64;

    uint32_t totalLength = N2 * N1;
    size_t inputByteSize = totalLength * sizeof(uint16_t); // float16 用 uint16_t 表示
    size_t outputByteSize = totalLength * sizeof(uint16_t);
    uint32_t tilingSize = 0;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *z = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16);
    uint8_t *tiling = GenerateTiling(totalLength, tileLength, tilingSize);

    if (x == nullptr || y == nullptr || z == nullptr || workspace == nullptr || tiling == nullptr) {
        printf("allocate memory failed\n");
        return -1;
    }

    // 读入 golden 数据文件（二进制 float16）
    ReadFile("./input/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile("./input/input_y.bin", inputByteSize, y, inputByteSize);

    // CPU 调试方式运行核函数
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_custom, blockDim, x, y, z, workspace, tiling);

    // 保存输出结果
    WriteFile("./output/output_z.bin", z, outputByteSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)z);
    AscendC::GmFree((void *)workspace);
    free(tiling);

    printf("add_custom finished, output written to ./output/output_z.bin\n");
    return 0;
}
