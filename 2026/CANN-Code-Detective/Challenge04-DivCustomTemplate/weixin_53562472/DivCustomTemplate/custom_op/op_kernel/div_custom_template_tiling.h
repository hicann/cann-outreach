/* div_custom_template_tiling.h
 *
 * DivCustomTemplate 算子的 tiling 数据结构定义（host 侧与 kernel 侧共享）。
 * host 侧 TilingFunc 在运行时填写本结构，随启动参数下发；
 * kernel 侧通过 GET_TILING_DATA 解析出同一份数据。
 */

#ifndef DIV_CUSTOM_TEMPLATE_TILING_H
#define DIV_CUSTOM_TEMPLATE_TILING_H
#include <cstdint>

// 输入数据类型编码：host 侧 tiling 写入，kernel 侧据此分发对应的核函数模板实例
constexpr uint32_t DIV_DTYPE_FLOAT16 = 0;
constexpr uint32_t DIV_DTYPE_FLOAT = 1;
// 每个 AI Core 内的 tile 切分数（配合队列双缓冲做流水）
constexpr uint32_t DIV_TILE_NUM = 8;

struct DivCustomTemplateTilingData {
    uint32_t totalLength; // 输入张量的总元素数
    uint32_t tileNum;     // 每个核内的 tile 切分数
    uint32_t dtype;       // 输入数据类型编码（DIV_DTYPE_FLOAT16 / DIV_DTYPE_FLOAT）
};

#endif // DIV_CUSTOM_TEMPLATE_TILING_H
