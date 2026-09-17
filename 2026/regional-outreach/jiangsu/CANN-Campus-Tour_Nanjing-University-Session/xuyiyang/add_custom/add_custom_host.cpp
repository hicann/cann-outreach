/*
 * add_custom host 侧调用示例
 * 演示: 根据二维 shape [N2, N1] 计算 tiling 参数 totalLength,
 *       并通过 KernelManager 拉起 add_custom 核函数。
 *
 * 实际工程中也可使用 ACL 单算子调用方式 (acloprunner / acl.op),
 * 此处给出 CANN 样例通用的 Kernel Launch 写法。
 */
#include "add_custom_tiling.h"
#include "kernel_operator.h" // 仅用于 GM_ADDR 宏说明, host 编译时由适配层提供

#include <cstdint>

using namespace AscendC;

// tiling 结构: 传递给 device 侧的数据量信息
struct AddCustomTilingData {
    uint32_t totalLength; // N2 * N1
    uint32_t tileLength;  // 与 device 侧 TILE_LENGTH 一致
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength);

/*
 * 计算 tiling 并启动核函数
 * x, y, z : 已在 device 上分配好的内存指针 (经 GM_ADDR 转换)
 * n2, n1  : 二维 shape [N2, N1]
 */
void LaunchAddCustom(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t n2, uint32_t n1)
{
    AddCustomTilingData tiling;
    tiling.totalLength = n2 * n1;      // ND 平铺后的一维总长
    tiling.tileLength = 2048;          // 与 device 侧 TILE_LENGTH 对应

    // 单核场景: blockDim = 1, l2/ub 由框架自动管理
    add_custom<<<1, nullptr, nullptr>>>(x, y, z, tiling.totalLength);
}
