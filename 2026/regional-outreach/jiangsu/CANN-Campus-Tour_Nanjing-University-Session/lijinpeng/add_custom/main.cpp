/**
 * @file main.cpp
 * @brief add_custom 算子 Host 侧主流程（Kernel Launch 直调方式）
 *
 * 流程：
 *  1. aclInit / aclrtSetDevice / aclrtCreateStream 初始化运行环境
 *  2. 读取 input_x.bin、input_y.bin（float16，ND 格式，shape [N2, N1]）到 Host
 *  3. 计算 Tiling 参数并拷贝到 Device
 *  4. 通过 ACLRT_LAUNCH_KERNEL(add_custom) 下发核函数
 *  5. 结果拷回 Host 写入 output_z.bin
 *  6. 若存在 golden_z.bin（由 scripts/gen_data.py 生成），与真值自动比对精度
 *
 * 用法：./add_custom [N2] [N1] [blockDim]
 *   N2       默认 8，shape 第 0 维
 *   N1       默认 2048，shape 第 1 维
 *   blockDim 默认 8，使用的 AI Core 个数
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_add_custom.h"
#include "add_custom_tiling.h"
#include "data_utils.h"
#include "fp16_utils.h"

namespace {
const char* const INPUT_X_PATH = "./input/input_x.bin";
const char* const INPUT_Y_PATH = "./input/input_y.bin";
const char* const GOLDEN_Z_PATH = "./input/golden_z.bin";
const char* const OUTPUT_Z_PATH = "./output/output_z.bin";
// 精度比对门限：float16 的机器精度约为 1e-3，逐元素加法误差限按相对误差 1e-3 控制
constexpr float RELATIVE_ERROR_THRESHOLD = 1e-3f;
constexpr float ABSOLUTE_ERROR_THRESHOLD = 1e-3f;
}  // namespace

#define CHECK_ACL(expr)                                                        \
    do {                                                                       \
        auto ret = (expr);                                                     \
        if (ret != ACL_SUCCESS) {                                              \
            printf("[ERROR] %s failed, error code = %d\n", #expr, ret);        \
            return -1;                                                         \
        }                                                                      \
    } while (0)

/**
 * @brief 向上按 32Byte（16 个 float16）对齐
 * @note 使用 64 位中间量，避免接近 uint32_t 上限时 (value + 15) 溢出
 */
static uint32_t AlignUp16(uint32_t value)
{
    uint64_t aligned = ((static_cast<uint64_t>(value) + ADD_CUSTOM_ALIGN_ELEM - 1) /
                        ADD_CUSTOM_ALIGN_ELEM) * ADD_CUSTOM_ALIGN_ELEM;
    return static_cast<uint32_t>(aligned);
}

/**
 * @brief 计算 Tiling 参数
 * @param shape    shape 数组，本算子为 2 维：[N2, N1]
 * @param dimNum   维度个数
 * @param coreNum  期望使用的核数
 * @param tiling   输出 Tiling 参数
 * @param blockDim 输出实际使用的核数
 */
static void ComputeTiling(const uint32_t* shape, uint32_t dimNum, uint32_t coreNum,
                          AddCustomTiling& tiling, uint32_t& blockDim)
{
    uint64_t totalLength = 1;
    for (uint32_t i = 0; i < dimNum; ++i) {
        totalLength *= shape[i];
    }

    // 数据量较小时降低核数，避免每个核分到的数据太少、流水收益不足
    uint32_t usableCore = static_cast<uint32_t>((totalLength + ADD_CUSTOM_MIN_ELEMS_PER_CORE - 1) /
                                                ADD_CUSTOM_MIN_ELEMS_PER_CORE);
    if (usableCore == 0) {
        usableCore = 1;
    }
    blockDim = (coreNum < usableCore) ? coreNum : usableCore;
    if (blockDim == 0) {
        blockDim = 1;
    }

    // 每个核处理的元素个数，按 32Byte 对齐
    uint32_t blockLength = AlignUp16(static_cast<uint32_t>((totalLength + blockDim - 1) / blockDim));

    // 核内 tile 长度：按每核期望 tile 个数推导，并限制在 [TILE_LENGTH_MIN, TILE_LENGTH_MAX]
    uint32_t tileLength = AlignUp16(blockLength / ADD_CUSTOM_TILE_NUM_TARGET);
    if (tileLength < ADD_CUSTOM_TILE_LENGTH_MIN) {
        tileLength = ADD_CUSTOM_TILE_LENGTH_MIN;
    }
    if (tileLength > ADD_CUSTOM_TILE_LENGTH_MAX) {
        tileLength = ADD_CUSTOM_TILE_LENGTH_MAX;
    }
    // tile 长度不需要超过单核数据量
    if (tileLength > blockLength) {
        tileLength = blockLength;
    }
    uint32_t tileNum = (blockLength + tileLength - 1) / tileLength;

    tiling.totalLength = static_cast<uint32_t>(totalLength);
    tiling.blockNum = blockDim;
    tiling.blockLength = blockLength;
    tiling.tileNum = tileNum;
    tiling.tileLength = tileLength;
    tiling.dimNum = dimNum;
    std::memset(tiling.shape, 0, sizeof(tiling.shape));
    for (uint32_t i = 0; i < dimNum && i < ADD_CUSTOM_MAX_SHAPE_DIM; ++i) {
        tiling.shape[i] = shape[i];
    }
    tiling.reserved = 0;

    printf("[INFO] tiling: totalLength = %u, blockNum = %u, blockLength = %u, "
           "tileNum = %u, tileLength = %u, dimNum = %u\n",
           tiling.totalLength, tiling.blockNum, tiling.blockLength,
           tiling.tileNum, tiling.tileLength, tiling.dimNum);
}

/**
 * @brief 与真值比对精度
 * @return 0 表示通过，-1 表示不通过
 */
static int VerifyResult(const std::vector<uint16_t>& output, const std::vector<uint16_t>& golden,
                        uint32_t totalLength)
{
    uint32_t errorCount = 0;
    float maxRelativeError = 0.0f;
    for (uint32_t i = 0; i < totalLength; ++i) {
        float actual = Fp16ToFloat(output[i]);
        float expect = Fp16ToFloat(golden[i]);
        float absError = std::fabs(actual - expect);
        float relativeError = (std::fabs(expect) > 1e-6f) ? (absError / std::fabs(expect)) : absError;
        if (relativeError > maxRelativeError) {
            maxRelativeError = relativeError;
        }
        if (relativeError > RELATIVE_ERROR_THRESHOLD && absError > ABSOLUTE_ERROR_THRESHOLD) {
            if (errorCount < 5) {
                printf("[ERROR] index %u: actual = %.6f, expect = %.6f\n", i, actual, expect);
            }
            ++errorCount;
        }
    }

    if (errorCount != 0) {
        printf("[ERROR] verification FAILED, error count = %u / %u, max relative error = %.6e\n",
               errorCount, totalLength, maxRelativeError);
        return -1;
    }
    printf("[INFO] verification PASSED, total = %u, max relative error = %.6e\n",
           totalLength, maxRelativeError);
    return 0;
}

int main(int argc, char* argv[])
{
    uint32_t shape[2] = {8, 2048};
    uint32_t coreNum = 8;
    if (argc > 1) {
        shape[0] = static_cast<uint32_t>(std::atoi(argv[1]));
    }
    if (argc > 2) {
        shape[1] = static_cast<uint32_t>(std::atoi(argv[2]));
    }
    if (argc > 3) {
        coreNum = static_cast<uint32_t>(std::atoi(argv[3]));
    }
    if (shape[0] == 0 || shape[1] == 0 || coreNum == 0) {
        printf("[ERROR] invalid arguments: N2 = %u, N1 = %u, blockDim = %u\n", shape[0], shape[1], coreNum);
        return -1;
    }

    // 元素总个数的上限受 Tiling 参数的位宽（uint32_t）约束，超限时明确报错而非静默溢出
    const uint64_t totalLength64 = static_cast<uint64_t>(shape[0]) * static_cast<uint64_t>(shape[1]);
    if (totalLength64 > 0xFFFFFFFFull) {
        printf("[ERROR] shape [%u, %u] 的元素总个数 %llu 超出 uint32_t 上限，"
               "当前 Tiling 参数不支持（如需支持请把 Tiling 中的长度字段改为 uint64_t）\n",
               shape[0], shape[1], static_cast<unsigned long long>(totalLength64));
        return -1;
    }
    const uint32_t totalLength = static_cast<uint32_t>(totalLength64);
    const size_t inputByteSize = static_cast<size_t>(totalLength) * sizeof(uint16_t);
    const size_t outputByteSize = inputByteSize;
    printf("[INFO] add_custom: shape = [%u, %u], totalLength = %u, dtype = float16, format = ND\n",
           shape[0], shape[1], totalLength);

    // 1. 校验输入文件是否存在、大小是否与 shape 一致
    if (!FileExists(INPUT_X_PATH) || !FileExists(INPUT_Y_PATH)) {
        printf("[ERROR] 输入文件不存在（%s 或 %s）。请先执行 "
               "python3 scripts/gen_data.py %u %u 生成输入数据\n",
               INPUT_X_PATH, INPUT_Y_PATH, shape[0], shape[1]);
        return -1;
    }
    size_t xFileSize = GetFileSize(INPUT_X_PATH);
    size_t yFileSize = GetFileSize(INPUT_Y_PATH);
    if (xFileSize != inputByteSize || yFileSize != inputByteSize) {
        printf("[ERROR] input file size mismatch: expect %zu Byte, input_x = %zu Byte, input_y = %zu Byte. "
               "请先执行 python scripts/gen_data.py %u %u 生成输入数据\n",
               inputByteSize, xFileSize, yFileSize, shape[0], shape[1]);
        return -1;
    }

    // 2. 初始化运行环境
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    // 3. 申请 Host / Device 内存
    uint16_t* xHost = nullptr;
    uint16_t* yHost = nullptr;
    uint16_t* zHost = nullptr;
    uint8_t* xDevice = nullptr;
    uint8_t* yDevice = nullptr;
    uint8_t* zDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void**>(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void**>(&yHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void**>(&zHost), outputByteSize));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void**>(&xDevice), inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void**>(&yDevice), inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void**>(&zDevice), outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // 4. 输入数据搬运到 Device
    ReadFile(INPUT_X_PATH, xFileSize, xHost, inputByteSize);
    ReadFile(INPUT_Y_PATH, yFileSize, yHost, inputByteSize);
    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // 5. Tiling 计算并下发
    const size_t tilingSize = sizeof(AddCustomTiling);
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void**>(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    AddCustomTiling tilingData;
    uint32_t blockDim = 1;
    ComputeTiling(shape, 2, coreNum, tilingData, blockDim);
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingSize, &tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // 6. 下发 kernel
    ACLRT_LAUNCH_KERNEL(add_custom)(blockDim, stream, xDevice, yDevice, zDevice, tilingDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // 7. 结果搬运回 Host 并落盘
    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(OUTPUT_Z_PATH, zHost, outputByteSize);

    // 8. 与真值比对（真值由 scripts/gen_data.py 生成，不存在时跳过）
    int ret = 0;
    size_t goldenSize = FileExists(GOLDEN_Z_PATH) ? GetFileSize(GOLDEN_Z_PATH) : 0;
    if (goldenSize == outputByteSize) {
        std::vector<uint16_t> golden(totalLength);
        ReadFile(GOLDEN_Z_PATH, goldenSize, golden.data(), outputByteSize);
        std::vector<uint16_t> output(zHost, zHost + totalLength);
        ret = VerifyResult(output, golden, totalLength);
    } else {
        printf("[WARN] golden file %s not found or size mismatch, skip precision verification. "
               "可执行 python scripts/verify_result.py %u %u 进行独立校验\n",
               GOLDEN_Z_PATH, shape[0], shape[1]);
    }

    // 9. 释放资源
    aclrtFreeHost(xHost);
    aclrtFreeHost(yHost);
    aclrtFreeHost(zHost);
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(zDevice);
    aclrtFree(tilingDevice);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    printf("[INFO] add_custom run finished\n");
    return ret;
}
