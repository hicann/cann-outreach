/**
 * @file inplace_rsqrt.cpp
 * @brief InplaceRsqrt 算子 Kernel 入口（自定义算子包 / 框架调用方式）
 *
 * 编译方式见仓库根目录 README.md 的「路线 A」。
 */

#include "kernel_operator.h"
#include "inplace_rsqrt.h"
#include "../op_host/inplace_rsqrt_tiling.h"

using namespace AscendC;
using namespace InplaceRsqrtNs;

/**
 * @param self      输入张量（4 维 [N4,N3,N2,N1]，ND，float16），同时是计算结果的落地点
 * @param selfOut   输出张量：算子原型中输出与输入同名 => 框架传入同一块地址（原地）；
 *                  若个别版本框架分配了独立输出地址，kernel 会自动多做一次搬出（见 Kernel 实现）
 * @param workspace 原地算子不需要 workspace，host tiling 中置 0
 * @param tiling    tiling 数据（totalLength / tileLength）
 *
 * 说明：kernel 入口名与算子类型 InplaceRsqrt 对应的 snake_case 形式保持一致，
 *       若 msopgen 生成的骨架入口名不同，请以骨架为准（只改函数名即可）。
 */
extern "C" __global__ __aicore__ void inplace_rsqrt(GM_ADDR self, GM_ADDR selfOut, GM_ADDR workspace, GM_ADDR tiling)
{
    // 纯 Vector 算子，显式声明只用 AIV，避免在 910B 上按 MIX 模式下发
    // （CANN 版本不支持该宏时，删除本行即可）
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // 框架写入的 tiling 缓冲按 TilingData 结构体解读
    // 若当前 CANN 版本的宏需要显式类型，可改为：
    //   GET_TILING_DATA_WITH_STRUCT(InplaceRsqrtTilingData, tilingData, tiling);
    GET_TILING_DATA(tilingData, tiling);

    InplaceRsqrt<half> op;
    op.Init(self, selfOut, tilingData.totalLength, tilingData.tileLength);
    op.Process();
}