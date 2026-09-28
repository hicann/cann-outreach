/*!
 * \file add_custom.cpp
 * \brief add_custom 算子的 kernel 入口函数
 *
 * 本文件实现了 add_custom 算子在 AI Core 上的 kernel 入口函数。
 * 该函数根据 tiling key（schMode）分发到相应的模板实现。
 * kernel 使用 tiling 策略将计算任务划分到多个 AI Core 上并行执行。
 *
 * 算子功能：z = x + y（逐元素加法）
 * 数据类型：float16
 * 数据格式：ND（逻辑 2 维 [N2, N1]，按元素总数一维化计算）
 */

#include "add_custom.h"

// 定义 add_custom 算子的 tiling key 枚举
// tiling key 用于区分不同数据类型的实现策略
enum class AddCustomTilingKey : uint32_t {
    TILING_KEY_FLOAT16 = 0, // float16 类型的 tiling key
};

// add_custom 算子的 kernel 入口函数
// 该函数是 AI Core 执行的入口点，根据模板参数 schMode 选择对应的数据类型实现
// @param x: 输入张量 x 的 GM 地址
// @param y: 输入张量 y 的 GM 地址
// @param z: 输出张量 z 的 GM 地址
// @param workspace: 工作空间的 GM 地址（add_custom 不使用）
// @param tiling: tiling 数据的 GM 地址，包含分块和内存管理信息
template <uint32_t schMode>
__global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    // 注册默认的 tiling 数据结构
    REGISTER_TILING_DEFAULT(AddCustomTilingData);
    // 从 GM 内存获取 tiling 数据
    GET_TILING_DATA_WITH_STRUCT(AddCustomTilingData, tilingData, tiling);

    // 根据 tiling key（schMode）分发到 float16 实现
    if constexpr (schMode == static_cast<uint32_t>(AddCustomTilingKey::TILING_KEY_FLOAT16)) {
        // float16 类型的实现
        NsAddCustom::AddCustom<half> op; // 算子 kernel 实例获取
        op.Init(x, y, z, &tilingData);   // 算子 kernel 实例初始化
        op.Process();                    // 算子 kernel 实例执行
    }
}
