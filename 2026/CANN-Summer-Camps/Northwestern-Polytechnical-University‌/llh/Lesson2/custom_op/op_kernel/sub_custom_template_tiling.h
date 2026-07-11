/**
 * @file    sub_custom_template_tiling.h
 * @brief   SubCustomTemplate 算子 — Tiling 数据结构 (Host 与 Kernel 共享)
 *
 * 本文件只含纯 C/C++ 语法，不含 __aicore__、__gm__ 等 ASC 关键字，
 * 可被 Host 侧 C++ 编译器和 Device 侧 ASC 编译器同时引用。
 *
 * 【Tiling 数据流】
 *   Host (CPU)                              Device (NPU)
 *   ──────────                              ────────────
 *   TilingFunc()                            每个 AI Core:
 *     ↓ 计算 totalLength + tileNum            REGISTER_TILING_DEFAULT
 *     ↓ 写入 TilingData                       GET_TILING_DATA_WITH_STRUCT
 *     ↓ 框架拷贝到 GM ──────────────→        读取 tiling_data → Init()
 *
 * 【参数说明】
 *   totalLength : 数据总元素数 (例: 8×2048 = 16384)
 *   tileNum     : 每核 tile 轮数，与 BUFFER_NUM 配合决定 tileLength
 *
 *   tileLength = blockLength / tileNum / BUFFER_NUM
 *   loopCount  = tileNum × BUFFER_NUM
 */

#ifndef SUB_CUSTOM_TEMPLATE_TILING_H
#define SUB_CUSTOM_TEMPLATE_TILING_H
#include <cstdint>

struct SubCustomTemplateTilingData {
    uint32_t totalLength;   // 输入数据总元素个数
    uint32_t tileNum;       // 每核切分轮数
};

#endif
