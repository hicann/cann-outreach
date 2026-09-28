/*
 * Relu 算子 host 侧验证程序（CPU 侧模拟/单测入口）
 * 构造 float16 输入 [N4,N3,N2,N1]，调用算子框架拉起 kernel，
 * 并与 CPU 参考实现（max(x,0)）比对结果。
 *
 * 说明：本文件需在安装了 CANN 与 AscendC CPU 模拟环境的机器上编译运行：
 *   bash scripts/build.sh run
 */
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include "relu_tiling.h"

// 由算子框架提供的 kernel 拉起封装（CPU 模拟环境下为桩实现）
extern "C" void relu_custom_do(const uint32_t shape[4],
                               const uint16_t *xHost, uint16_t *yHost);

// CPU 参考实现：half 按位转 float 后做 relu，再转回 half
static void relu_reference(const std::vector<uint16_t> &x, std::vector<uint16_t> &y)
{
    for (size_t i = 0; i < x.size(); i++) {
        __fp16 xi;
        std::memcpy(&xi, &x[i], sizeof(xi));
        __fp16 yo = xi > static_cast<__fp16>(0.0f) ? xi : static_cast<__fp16>(0.0f);
        std::memcpy(&y[i], &yo, sizeof(yo));
    }
}

int main()
{
    // 测试 shape [N4, N3, N2, N1]
    const uint32_t shape[4] = {2, 3, 4, 16};
    ReluTilingData tiling;
    ComputeTiling(shape, tiling);
    printf("totalNum = %u, tileSize = %u\n", tiling.totalNum, tiling.tileSize);

    std::vector<uint16_t> x(tiling.totalNum), y(tiling.totalNum, 0);
    // 构造输入：正负交替，覆盖 relu 的两个分支
    for (uint32_t i = 0; i < tiling.totalNum; i++) {
        __fp16 v = (i % 2 == 0) ? static_cast<__fp16>(i % 7 + 1)
                                : static_cast<__fp16>(-(i % 5 + 1));
        std::memcpy(&x[i], &v, sizeof(v));
    }

    relu_custom_do(shape, x.data(), y.data());

    std::vector<uint16_t> ref(tiling.totalNum, 0);
    relu_reference(x, ref);

    int bad = 0;
    for (uint32_t i = 0; i < tiling.totalNum; i++) {
        if (y[i] != ref[i]) {
            if (bad < 5) {
                printf("mismatch @%u: got=%04x expect=%04x\n", i, y[i], ref[i]);
            }
            bad++;
        }
    }
    if (bad == 0) {
        printf("PASS: relu result matches reference (%u elems)\n", tiling.totalNum);
        return 0;
    }
    printf("FAIL: %d mismatches\n", bad);
    return 1;
}
