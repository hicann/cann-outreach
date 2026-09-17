# add_custom 算子

基于 **Ascend C** 实现的逐元素加法算子：`z = x + y`。

## 算子定义

| 项目 | 说明 |
| --- | --- |
| 算子名称 | `add_custom` |
| 功能 | 两个输入张量逐元素相加，输出结果张量 |
| 数学公式 | `z = x + y` |
| 输入 | `x` (float16, ND), `y` (float16, ND) |
| 输出 | `z` (float16, ND) |
| shape | 2 维 `[N2, N1]`，x/y/z 形状一致 |
| 数据类型 | float16 |
| 数据格式 | ND |

## 文件结构

```
add_custom/
├── add_custom.cpp        # device 侧核函数实现 (KernelAdd + add_custom 入口)
├── add_custom_tiling.h   # host 侧算子注册 (输入/输出/属性定义)
├── add_custom_host.cpp   # host 侧 tiling 计算与核函数调用示例
├── CMakeLists.txt        # 构建脚本
└── README.md
```

## 实现说明

- **分块处理**：device 侧按 `TILE_LENGTH = 2048` 个元素分块搬运到 Unified Buffer，在 vector 单元执行 `Add`，再拷回 GlobalMemory；尾部不足一块时按 32B（16 个 fp16）对齐处理。
- **流水优化**：输入/输出队列均采用 double buffer（`BUFFER_NUM = 2`），搬运与计算可重叠。
- **二维 shape**：ND 格式下 `[N2, N1]` 按行主序平铺成一维处理，host 侧传入 `totalLength = N2 * N1`。

## 构建与运行

在配置好 CANN 环境的机器上（`source ${ASCEND_HOME}/ascend-toolkit/set_env.sh`）：

```bash
cmake -B build -DASCEND_HOME=/usr/local/Ascend
cmake --build build
```

## 调用示例

见 `add_custom_host.cpp` 中的 `LaunchAddCustom(x, y, z, n2, n1)`：计算 tiling 后以单核拉起核函数。如需多核并行，可按 `blockIdx` 对 `totalLength` 切分。
