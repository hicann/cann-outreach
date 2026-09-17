/**
 * @file fp16_utils.h
 * @brief Host 侧 float16 与 float 的相互转换（IEEE 754 binary16）
 *
 * 说明：Host 侧不依赖 CANN 的 half 类型，统一用 uint16_t 承载 float16 的位模式，
 *       避免 Host 编译环境差异带来的类型问题；Kernel 侧仍使用 half 类型。
 */

#ifndef FP16_UTILS_H
#define FP16_UTILS_H

#include <cstdint>
#include <cstring>

/**
 * @brief float16（位模式）转 float
 */
inline float Fp16ToFloat(uint16_t value)
{
    uint32_t sign = (static_cast<uint32_t>(value) & 0x8000u) << 16;
    uint32_t exponent = (static_cast<uint32_t>(value) >> 10) & 0x1Fu;
    uint32_t mantissa = static_cast<uint32_t>(value) & 0x3FFu;
    uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;  // +/-0
        } else {
            // 次正规数：规格化为 float 的正规数
            uint32_t shift = 0;
            uint32_t m = mantissa;
            while ((m & 0x400u) == 0) {
                ++shift;
                m <<= 1;
            }
            m &= 0x3FFu;
            bits = sign | (static_cast<uint32_t>(113 - shift) << 23) | (m << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);  // Inf / NaN
    } else {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

/**
 * @brief float 转 float16（位模式），round to nearest even
 */
inline uint16_t FloatToFp16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t expField = (bits >> 23) & 0xFFu;
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (expField == 0xFFu) {  // Inf / NaN
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
        return static_cast<uint16_t>(sign | 0x7E00u);
    }

    int32_t exponent = static_cast<int32_t>(expField) - 127 + 15;

    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);  // 上溢为 Inf
    }

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);  // 下溢为 0
        }
        mantissa |= 0x800000u;  // 补上隐含的整数位
        uint32_t shift = static_cast<uint32_t>(14 - exponent);
        uint32_t halfMantissa = mantissa >> shift;
        uint32_t remainder = mantissa & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (halfMantissa & 1u) != 0)) {
            ++halfMantissa;
        }
        return static_cast<uint16_t>(sign | halfMantissa);
    }

    uint16_t half = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
    uint32_t remainder = mantissa & 0x1FFFu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u) != 0)) {
        ++half;  // 进位自然溢出到指数位，符合 IEEE 754 语义
    }
    return half;
}

#endif  // FP16_UTILS_H
