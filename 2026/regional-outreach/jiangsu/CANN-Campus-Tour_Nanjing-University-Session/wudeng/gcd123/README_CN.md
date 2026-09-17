# gcd123 算子

`gcd123` 是基于 Ascend C 框架开发的自定义算子，用于计算两个输入的逐元素最大公约数。

## 算子规格

| 项目 | 内容 |
| --- | --- |
| 算子名 | `gcd123` |
| 输入 | `self`、`other` |
| 输出 | `out` |
| shape | 4 维，例如 `[N4, N3, N2, N1]` |
| shape 约束 | `self` 与 `other` 满足 broadcast 关系，`out` 为 broadcast 后的 shape |
| 数据类型 | `float16` |
| Format | `ND` |

`float16` 的值会先转换为 `int64_t` 后再计算最大公约数，因此输入应当是整数值，且建议落在 float16 能精确表示整数的范围内。

## 文件结构

```text
gcd123/
├── op_host/
│   ├── gcd123.cpp
│   └── gcd123_tiling.h
├── op_kernel/
│   └── gcd123.cpp
├── cmake/
├── scripts/
├── CMakeLists.txt
├── CMakePresets.json
├── build.sh
└── README_CN.md
```

## 编译

在已安装 CANN 的 Linux 环境中：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
bash build.sh
```

也可以使用 CMake preset：

```bash
cmake --preset default
cmake --build build_out --target package -j16
```

本目录中的 `cmake/`、`scripts/`、`CMakeLists.txt`、`CMakePresets.json` 和 `build.sh` 取自 Ascend C 自定义算子示例框架，用于生成算子包。
