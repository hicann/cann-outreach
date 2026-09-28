# Atanh 自定义算子（AscendC / CANN）

反双曲正切算子：`y = atanh(x) = 0.5 * ln((1+x)/(1-x))`，定义域 `x ∈ (-1, 1)`，超出定义域输出 `NaN`。



| 属性      | 值                                      |
| ------- | -------------------------------------- |
| 输入 / 输出 | 1 输入 `x`，1 输出 `y`                      |
| Shape   | 4 维，如 `[N4,N3,N2,N1]`（tiling 按任意维展平处理） |
| 数据类型    | `float16`（half）                        |
| Format  | `ND`                                   |
| 计算单元    | AIV（Vector），多核并行 + 双缓冲流水               |

## 目录结构



```
atanh/

├── op\_info.json               # 算子描述（msopgen 输入，含 dtype/format/impl 信息）

├── CMakeLists.txt             # 构建脚本

├── framework/

│   └── op\_define.cc           # 算子原型：InferShape / InferDataType / OP\_ADD

├── op\_host/

│   ├── atanh\_tiling.h         # Tiling 数据结构定义

│   └── atanh\_tiling.cpp       # Tiling：展平 shape、算 totalLength、定 blockDim

└── op\_kernel/

&#x20;   └── atanh\_kernel.cpp       # Kernel：多核切分 + 双缓冲 + Atanh 计算 + 尾块对齐处理
```

## 实现要点



1. **多核切分**：tiling 将 4 维 shape 展平为 `totalLength`，按可用核数 `blockDim` 均分，

   余数全部归最后一个核。

2. **双缓冲流水**：`TPipe` + `TQue<QuePosition::VECIN/VECOUT, 2>`，搬入 / 计算 / 搬出三级流水重叠。

3. **32B 对齐**：`DataCopy` 的搬运量要求为 32B 整数倍（half 即 16 的倍数），主循环

   `TILE_LENGTH=2048` 天然满足；长度不足一个 tile 的**尾块**改用支持非对齐搬运的

   `DataCopyPad`（`blockLen` 单位字节）。

4. **数学接口**：直接调用 AscendC 数学库 `Atanh(dst, src, count)`；若当前 CANN 版本

   无该接口，kernel 注释中给出了基于 `Muls/Adds/Div/Log` 的公式组合后备实现。

## 构建与部署（在装有 CANN 的昇腾 Linux 环境）

推荐使用 `msopgen` 生成工程骨架，再覆盖为本次实现：



```
\# 1. 设置 CANN 环境

source /usr/local/Ascend/ascend-toolkit/set\_env.sh

\# 2. 用 msopgen 生成空工程骨架（芯片型号按实际修改，如 ascend910b / ascend310p）

msopgen gen -i op\_info.json -c ai\_core-ascend910b -out ./atanh\_project

\# 3. 用本目录的实现文件覆盖生成骨架中的同名文件

\#    framework/op\_define.cc

\#    op\_host/atanh\_tiling.h  op\_host/atanh\_tiling.cpp

\#    op\_kernel/atanh\_kernel.cpp

\# 4. 编译（生成 libops\_info.so / libtiling\_info.so / libatanh\_kernel.so）

cd atanh\_project && bash build.sh
```

不使用 msopgen 时，可尝试本目录 `CMakeLists.txt` 直接构建：



```
cmake -B build && make -C build -j
```

> 注意：kernel 侧需要 CANN 的 ascendc 编译器工具链，请确保工程能正确注入
> `-DTILING_DATA_T=AtanhTilingData`
>
>  编译宏（msopgen 工程会自动从 
>
> `op_info.json`
> 的 
>
> `tiling_data`
>
>  字段生成）。

## 验证



* **数值对比**：用 `numpy.arctanh` / `torch.atanh` 对随机输入（`[-0.99, 0.99]` 均匀分布）

  逐元素对比，float16 下相对误差在 `1e-3` 量级。

* **边界输入**：`x = ±1` 附近应输出接近 `±∞`，`|x| > 1` 输出 `NaN`。

* **调用方式**：可通过 aclnn 单算子接口（`aclnnAtanh`）或 MindStudio 算子调试

  工具加载生成的 `.so` 进行验证。

## 常见问题



* **算子名冲突**：若注册 `Atanh` 与内置算子冲突，将 `op_info.json` 的 `op` 字段、

  `op_define.cc` 中的类名 /`OP_ADD` 参数统一改为 `AtanhCustom` 即可（tiling 注册

  宏 `REGISTER_TILING(atanh, ...)` 同步改为 `atanh_custom`）。

* **编译报 Atanh 未定义**：当前 CANN 版本数学库无该接口，切换 kernel 中注释的

  公式组合实现。