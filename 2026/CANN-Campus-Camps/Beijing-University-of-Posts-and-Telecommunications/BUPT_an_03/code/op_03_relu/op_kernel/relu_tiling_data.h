/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

struct ReluTilingData {
    int64_t totalNum = 0;     // 鎬诲厓绱犳暟閲?
    int64_t blockFactor = 1;  // 姣忎釜鏍稿鐞嗙殑鍏冪礌鏁伴噺
    int64_t ubFactor = 0;     // 姣忔 UB 寰幆澶勭悊鐨勫厓绱犳暟閲?
};
#endif