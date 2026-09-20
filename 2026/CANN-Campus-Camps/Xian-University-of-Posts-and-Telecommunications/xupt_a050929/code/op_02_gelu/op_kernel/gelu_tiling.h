#pragma once

#include <cstdint>

struct GeluTilingData {
    uint32_t length;            // 杈撳叆鎬诲厓绱犱釜鏁?
    uint32_t smallCoreDataNum;  // 灏忔牳(澶勭悊灏忓潡鏁版嵁鐨勬牳)璐熻矗鐨勫厓绱犱釜鏁?
    uint32_t bigCoreDataNum;    // 澶ф牳(澶勭悊澶у潡鏁版嵁鐨勬牳)璐熻矗鐨勫厓绱犱釜鏁?
    uint32_t finalSmallTileNum; // 灏忔牳闇€瑕佸鐞嗙殑tile涓暟
    uint32_t finalBigTileNum;   // 澶ф牳闇€瑕佸鐞嗙殑tile涓暟
    uint32_t tileDataNum;       // 姣忎釜tile鐨勫厓绱犱釜鏁?
    uint32_t smallTailDataNum;  // 灏忔牳鏈€鍚庝竴鐗噒ile鐨勫厓绱犱釜鏁?
    uint32_t bigTailDataNum;    // 澶ф牳鏈€鍚庝竴鐗噒ile鐨勫厓绱犱釜鏁?
    uint32_t tailBlockNum;      // 澶勭悊澶у潡鏁版嵁鐨勬牳(澶ф牳)涓暟
};