// 传统 TilingKey：Host 设置 key，非模板 Kernel 入口显式选择数据类型。
#pragma once

#include <cstdint>

constexpr uint64_t GELU_TILING_FP16 = 1;
constexpr uint64_t GELU_TILING_FP32 = 2;