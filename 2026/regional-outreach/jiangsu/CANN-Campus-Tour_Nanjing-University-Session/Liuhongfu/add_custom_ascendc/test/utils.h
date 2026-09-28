/**
 * @file utils.h
 * @brief 测试样例公共工具：文件读写、float16 转换、结果比对。
 */
#ifndef ADD_CUSTOM_TEST_UTILS_H
#define ADD_CUSTOM_TEST_UTILS_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/**
 * @brief IEEE 754 binary16 -> float
 *        自实现，避免依赖 CANN 头文件里版本不一的 f16 转换接口。
 */
inline float Fp16ToFloat(uint16_t h)
{
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000U) << 16U;
    uint32_t exp = (h >> 10U) & 0x1FU;
    uint32_t man = h & 0x03FFU;
    uint32_t bits = 0U;

    if (exp == 0U) {
        if (man == 0U) {
            bits = sign;  // +-0
        } else {
            // 次正规数：规格化
            exp = 1U;
            while ((man & 0x0400U) == 0U) {
                man <<= 1U;
                --exp;
            }
            man &= 0x03FFU;
            bits = sign | ((exp + (127U - 15U)) << 23U) | (man << 13U);
        }
    } else if (exp == 0x1FU) {
        bits = sign | 0x7F800000U | (man << 13U);  // Inf / NaN
    } else {
        bits = sign | ((exp + (127U - 15U)) << 23U) | (man << 13U);
    }

    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

/**
 * @brief float -> IEEE 754 binary16（四舍五入到最近偶数）
 */
inline uint16_t FloatToFp16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 16U) & 0x8000U;
    const int32_t exp = static_cast<int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
    const uint32_t man = bits & 0x007FFFFFU;

    if (exp >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00U);  // 溢出为 Inf
    }
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);  // 下溢为 0
        }
        // 次正规数
        const uint32_t m = (man | 0x00800000U) >> static_cast<uint32_t>(1 - exp);
        uint32_t rounded = m >> 13U;
        const uint32_t rem = m & 0x1FFFU;
        if (rem > 0x1000U || (rem == 0x1000U && (rounded & 1U) != 0U)) {
            ++rounded;
        }
        return static_cast<uint16_t>(sign | rounded);
    }

    uint32_t rounded = man >> 13U;
    const uint32_t rem = man & 0x1FFFU;
    uint32_t e = static_cast<uint32_t>(exp);
    if (rem > 0x1000U || (rem == 0x1000U && (rounded & 1U) != 0U)) {
        ++rounded;
        if (rounded == 0x400U) {
            rounded = 0U;
            ++e;
            if (e >= 0x1FU) {
                return static_cast<uint16_t>(sign | 0x7C00U);
            }
        }
    }
    return static_cast<uint16_t>(sign | (e << 10U) | rounded);
}

/**
 * @brief 读取二进制文件到 uint16_t 缓冲（元素个数由文件大小推断）
 */
inline bool ReadBinFile(const std::string& path, std::vector<uint16_t>& out)
{
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr) {
        std::printf("[ERROR] cannot open file: %s\n", path.c_str());
        return false;
    }
    std::fseek(fp, 0, SEEK_END);
    const long size = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if (size < 0 || (size % 2) != 0) {
        std::fclose(fp);
        std::printf("[ERROR] bad file size: %ld\n", size);
        return false;
    }
    out.resize(static_cast<size_t>(size) / 2U);
    const size_t readNum = std::fread(out.data(), 1, static_cast<size_t>(size), fp);
    std::fclose(fp);
    return readNum == static_cast<size_t>(size);
}

/**
 * @brief 写二进制文件
 */
inline bool WriteBinFile(const std::string& path, const std::vector<uint16_t>& data)
{
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        std::printf("[ERROR] cannot create file: %s\n", path.c_str());
        return false;
    }
    const size_t written = std::fwrite(data.data(), 1, data.size() * 2U, fp);
    std::fclose(fp);
    return written == data.size() * 2U;
}

/**
 * @brief 逐元素比对，返回最大绝对误差
 */
inline double CompareFp16(const std::vector<uint16_t>& actual,
                          const std::vector<uint16_t>& expect,
                          size_t& mismatchCount)
{
    mismatchCount = 0U;
    double maxErr = 0.0;
    const size_t n = (actual.size() < expect.size()) ? actual.size() : expect.size();
    for (size_t i = 0; i < n; ++i) {
        const double a = static_cast<double>(Fp16ToFloat(actual[i]));
        const double e = static_cast<double>(Fp16ToFloat(expect[i]));
        const double diff = std::fabs(a - e);
        if (diff > maxErr) {
            maxErr = diff;
        }
        // fp16 的精度约为 2^-11，这里给一个宽松但有意义的绝对误差阈值
        if (diff > 1e-3) {
            if (mismatchCount < 5U) {
                std::printf("[MISMATCH] idx=%zu actual=%f expect=%f diff=%f\n", i, a, e, diff);
            }
            ++mismatchCount;
        }
    }
    if (actual.size() != expect.size()) {
        std::printf("[ERROR] size mismatch: actual=%zu expect=%zu\n", actual.size(), expect.size());
        mismatchCount += 1U;
    }
    return maxErr;
}

#endif  // ADD_CUSTOM_TEST_UTILS_H
