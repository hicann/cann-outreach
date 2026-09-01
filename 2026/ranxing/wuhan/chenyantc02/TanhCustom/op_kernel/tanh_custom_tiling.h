/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * ------------------------------------------------------------------------- */
#ifndef TANH_CUSTOM_TILING_H
#define TANH_CUSTOM_TILING_H
#include <cstdint>
struct TanhCustomTilingData {
    // TODO: 考生自行补齐Tiling参数定义，直接定义即可，无需套用宏
    uint32_t totalLength;
    uint32_t tileNum;
    
    void set_totalLength(uint32_t len) {
        totalLength = len;
    }
    
    void set_tileNum(uint32_t num) {
        tileNum = num;
    }

};

#endif 
/*
#include "register/tilingdata_base.h"
#pragma once
#include "tiling/tiling_api.h"

namespace optiling {
    // 使用宏定义tiling数据结构
    BEGIN_TILING_DATA_DEF(TanhCustomTilingData)
        TILING_DATA_FIELD_DEF(uint32_t, totalLength);
        TILING_DATA_FIELD_DEF(uint32_t, tileNum);  // 如果需要的话
    END_TILING_DATA_DEF;

    // 注册tiling数据类
    REGISTER_TILING_DATA_CLASS(TanhCustom, TanhCustomTilingData)
}
*/
