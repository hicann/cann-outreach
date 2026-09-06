// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// 队列双缓冲份数: host 侧 UB 预算与 kernel 侧 InitBuffer 共用此常量, 保证两侧一致
constexpr int32_t MUL_BUFFER_NUM = 2;

// Mul 算子 tiling 参数(EleWise 标准切分):
//   多核切分: 每核至少 4KB 数据, blockFormer 按 512 元素对齐, 最后一个 block 为较小的 blockTail;
//   UB 切分:  ubFormer 按 256B 对齐, 区分 普通block/末block 各自的循环次数与尾 tile 大小。
// 约束: 本方案所有 DataCopy 长度均为 32B 整数倍; dim0 * sizeof(dtype) 非 32B 整数倍时
//       末 tile 需改用 DataCopyPad(本算例 8x2048=16384 恒满足, 未启用)。
struct MulTilingData {
    uint32_t dim0;                // 输入元素总数(各维连乘)
    uint32_t blockNum;            // 虚拟 block 数 = 实际启动核数(SetBlockDim 的值)
    uint32_t blockFormer;         // 每核基础元素数, 512 元素对齐; 末 block 实际为
                                  // blockTail = dim0 - (blockNum - 1) * blockFormer
    uint32_t ubFormer;            // UB 单 tile 元素数, 256B 对齐
    uint32_t ubLoopOfFormerBlock; // 普通 block 的 UB 循环次数
    uint32_t ubTailOfFormerBlock; // 普通 block 尾 tile 元素数
    uint32_t ubLoopOfTailBlock;   // 末 block 的 UB 循环次数
    uint32_t ubTailOfTailBlock;   // 末 block 尾 tile 元素数
};
