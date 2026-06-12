/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * ------------------------------------------------------------------------- */

#ifndef TANH_CUSTOM_TILING_H
#define TANH_CUSTOM_TILING_H

#include <cstdint>

struct TanhCustomTilingData {
    uint32_t totalLength;
    uint32_t tileNum;
};

#endif
