/*!
 * \file square_tiling_data.h
 * \brief Square leaderboard tiling data.
 */

#ifndef _SQUARE_TILING_DATA_H_
#define _SQUARE_TILING_DATA_H_

#include <cstdint>

struct SquareTilingData {
    uint32_t totalNum = 0;
    uint32_t blockFactor = 1;      // regular-core element count, 256B aligned
    uint32_t blockNum = 1;
    uint32_t normalRepeats = 0;    // vector repeats for a regular core
    uint32_t lastBlockLength = 0;
    uint32_t lastFullRepeats = 0;
    uint32_t lastTail = 0;
    uint32_t flags = 0;            // bit0: all blocks identical; bit1: last copy is 32B aligned
};

#endif
