/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * ... (License header)
 */

/*!
 * \file relu_tiling_data.h
 * \brief ReLU TilingData 结构
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

struct ReluTilingData {
    int64_t totalNum = 0;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
};

#endif // _RELU_TILING_DATA_H_
