/**
 * @file add_custom_tiling.h
 * @brief add_custom 算子 Tiling 参数定义（Host 侧计算，Kernel 侧读取）
 *
 * 说明：
 *  1. 该结构体由 Host 侧（main.cpp）填充，通过 kernel 入参以 GM_ADDR 的形式传入，
 *     Kernel 侧按内存布局直接读取，因此 Host / Kernel 两侧的字段类型与顺序必须完全一致，
 *     且不能包含虚函数、指针成员等与地址空间相关的成员。
 *  2. Tiling 只描述数据切分策略，不改变算子的数学语义（z = x + y）。
 */

#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H

// 定长类型 uint32_t 的来源：
//  Host 侧由 <stdint.h> 提供；Kernel 侧由 kernel_operator.h 提供。
//  Ascend C 的编译环境不保证提供 C++ 标准库头文件（且 kernel 侧本就禁止依赖 STL），
//  因此此处仅在头文件确实存在时才包含；Kernel 侧若不存在，则由 add_custom.cpp 中
//  先包含的 kernel_operator.h 提供的 typedef 满足需求。
#if defined(__has_include)
#if __has_include(<stdint.h>)
#include <stdint.h>
#endif
#else
#include <stdint.h>
#endif

// ---------------------------------------------------------------------------
// 切分参数（Host / Kernel 共用常量）
// ---------------------------------------------------------------------------

// 32Byte 对齐的元素个数，float16 下为 16 个元素（32B / 2B）
constexpr uint32_t ADD_CUSTOM_ALIGN_ELEM = 16;
// 32Byte 对齐字节数
constexpr uint32_t ADD_CUSTOM_ALIGN_BYTE = 32;
// uint16_t 类型的位宽
constexpr uint32_t ADD_CUSTOM_DTYPE_BYTE = 2;
// shape 支持的最大维度个数（本算子为 2 维，预留扩展）
constexpr uint32_t ADD_CUSTOM_MAX_SHAPE_DIM = 8;

// 单个 tile 的最小元素数
constexpr uint32_t ADD_CUSTOM_TILE_LENGTH_MIN = 256;
// 单个 tile 的最大元素数：8192 个 float16 = 16KB/块，
// 3 个队列 * 2 份 double buffer = 96KB UB 占用，控制在 UB 容量（192KB）以内
constexpr uint32_t ADD_CUSTOM_TILE_LENGTH_MAX = 8192;
// 每个核期望切分出的 tile 个数
constexpr uint32_t ADD_CUSTOM_TILE_NUM_TARGET = 8;
// 每个核处理的最小元素数，用于避免数据量较小时核数过多
constexpr uint32_t ADD_CUSTOM_MIN_ELEMS_PER_CORE = 1024;

// ---------------------------------------------------------------------------
// Tiling 结构体
// ---------------------------------------------------------------------------
struct AddCustomTiling {
    uint32_t totalLength;                      // x/y/z 的元素总个数 = N2 * N1
    uint32_t blockNum;                         // 实际使用的 AI Core 个数（blockDim）
    uint32_t blockLength;                      // 每个核处理的元素个数（32B 对齐）
    uint32_t tileNum;                          // 每个核内部的切分块数
    uint32_t tileLength;                       // 每个 tile 的元素个数（32B 对齐）
    uint32_t dimNum;                           // 参与计算的维度个数，本算子固定为 2
    uint32_t shape[ADD_CUSTOM_MAX_SHAPE_DIM];  // 输入/输出 shape：shape[0] = N2, shape[1] = N1
    uint32_t reserved;                         // 预留字段，保持结构体 4 字节对齐
};

#endif  // ADD_CUSTOM_TILING_H
