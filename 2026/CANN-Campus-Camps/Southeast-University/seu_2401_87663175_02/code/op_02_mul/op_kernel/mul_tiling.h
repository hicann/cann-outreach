#pragma once
#include <cstdint>

namespace mul_config {
// Per-dtype tuning. This avoids forcing FP16 and FP32 to share one
// blockDim sweet spot.
constexpr uint32_t FP16_INPUT_BYTES_PER_CORE = 4096U;
constexpr uint32_t FP32_INPUT_BYTES_PER_CORE = 8192U;

constexpr uint32_t UB_PAD_BYTES = 256U;
constexpr uint32_t Z_UB_ADDR = 64U * 1024U;
}  // namespace mul_config

struct MulTilingData {
    uint32_t length;
};