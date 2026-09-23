/*!
 * \file abs.h
 * \brief Abs 算子 kernel 类定义
 */

#ifndef ABS_H
#define ABS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "abs_tiling_data.h"
#include "abs_tiling_key.h"

namespace NsAbs {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1; // 每队列 buffer 块数（双缓冲）

template <typename DT_X>
class Abs {
public:
    __aicore__ inline Abs() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AbsTilingData* tilingData)
    {
        // TODO 考生自行补齐
    }

    __aicore__ inline void Process()
    {
        // 循环次数 = tileNum × BUFFER_NUM（tileLength 已按 BUFFER_NUM 折半）
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // TODO 考生自行补齐
    }

    __aicore__ inline void Compute()
    {
        // TODO 考生自行补齐
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        // TODO 考生自行补齐
    }

private:
    // TODO 考生自行补齐
};

} // namespace NsAbs
#endif // ABS_H
