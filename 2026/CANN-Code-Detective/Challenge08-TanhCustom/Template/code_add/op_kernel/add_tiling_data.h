/*!
 * \file add_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _ADD_TILING_DATA_H_
#define _ADD_TILING_DATA_H_

struct AddTilingData {
    uint32_t totalLength; // 总元素数量
    uint32_t tileNum;     // 单核内分块数
};
#endif
