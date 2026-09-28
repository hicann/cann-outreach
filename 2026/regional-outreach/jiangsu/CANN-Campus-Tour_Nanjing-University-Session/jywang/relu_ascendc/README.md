# Relu 算子（AscendC）

- 输入/输出：1 进 1 出，dtype = float16，Format = ND，shape 为 4 维 `[N4, N3, N2, N1]`
- 计算：`y = max(x, 0)`
- 实现方式：host 侧把 4 维 shape 折算成一维总元素量 `totalNum = N4*N3*N2*N1`（ND 格式下数据连续，无需关心维度），kernel 按 tile（2048 元素、双缓冲）从 Global Memory 搬入 UB，用 `Max(x, 0)` 向量指令计算后搬出。

## 目录

| 文件 | 说明 |
|---|---|
| `kernel/relu_custom.cpp` | kernel 侧实现（KernelRelu 类 + `relu_custom` 核函数） |
| `host/relu_tiling.h` | tiling 结构体与 4 维 shape → totalNum 的 tiling 计算 |
| `host/main.cpp` | host 验证程序，与 CPU 参考实现比对 |
| `scripts/build.sh` | 编译/运行脚本 |

## 构建与验证（需装有 CANN 的环境）

```bash
bash scripts/build.sh        # 编译 kernel
bash scripts/build.sh run    # 编译并运行 CPU 模拟验证
```

## 注意事项

- `kernel/relu_custom.cpp` 中 `TILE_SIZE` 与 `host/relu_tiling.h` 中 `tileSize` 必须一致。
- 若总元素量不是 `TILE_SIZE` 整数倍，可在 kernel 的 `Process()` 末尾追加尾块处理（用对齐缓冲 + 有效元素数做 `Max`/`DataCopy`）。
- 本目录下文件无法在 Windows 上编译，仅作源码交付。
