/*!
 * \file add_custom_tiling_data.h
 * \brief add_custom tiling data struct
 *
 * 该结构体由 host 侧 tiling 计算填充，经 GM 传递给 device 侧 kernel，
 * 描述核切分（blockFactor）与 UB 切分（ubFactor）策略。
 */

#ifndef _ADD_CUSTOM_TILING_DATA_H_
#define _ADD_CUSTOM_TILING_DATA_H_

struct AddCustomTilingData {
    int64_t totalNum = 0;    // 全局元素总数
    int64_t blockFactor = 0; // 每个 AI Core 负责的元素数
    int64_t ubFactor = 0;    // 每个 AI Core 内单次处理（UB 块）的元素数
};
#endif
