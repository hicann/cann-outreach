/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 * ------------------------------------------------------------------------- */

#ifndef TANH_CUSTOM_TILING_H
#define TANH_CUSTOM_TILING_H

#include <cstdint>

struct TanhCustomTilingData {
    int64_t totalNum = 0;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
};

#endif
