## 作业说明

### Lesson 1
- **课程题目**：补充实现 `mul_custom.asc`，使用 Ascend C 编程实现矢量乘法算子 `mul_custom`。
- **数据类型**：输入与输出均为 `float` 类型。
- **数据形状（Shape）**：输入 Shape 为 `(8, 2048)`，输出 Shape 与输入相同。
- **数据布局（Format）**：ND。
- **实现说明**：采用 8 核并行 + 8 tile/核的分块策略，使用 BUFFER_NUM=2 双缓冲队列。

### Lesson 2
- **课程题目**：根据提供的算子原型文件 `sub_custom_template.json`，使用 `msopgen` 工具生成算子工程，并完成 Ascend C 核函数开发，实现矢量减法算子 `SubCustomTemplate`，计算 `z = x - y`。
- **算子功能**：矢量减法，`z = x - y`。
- **数据类型**：输入 `x`、`y` 和输出 `z` 均支持 `float16` 和 `float32`（原型中已定义）。
- **数据形状（Shape）**：输入 `x` 和 `y` 的形状为 `(8, 2048)`，输出 `z` 形状相同。
- **数据布局（Format）**：ND。
- **实现说明**：8 核并行，QUE_DEPTH=2 双缓冲流水线，搬入/计算/搬出重叠执行。

### Lesson 3
- **课程题目**：算子性能调优。
- **调优目标**：当前该算子的执行耗时约为 **98 µs**（使用 `msprof op` 在真实硬件上测得），尚未达到预期性能，需要进一步调优。在不改变算子功能的前提下，**将总耗时降低至 80 µs 以下**。
- **优化策略**：
  1. **op_kernel**: BUFFER_NUM 1→2，启用双缓冲，搬入/计算/搬出流水并行
  2. **op_host TilingFunc**: 动态 tiling 策略 — BlockDim 最大 40 满核并行，TARGET_TILE_LENGTH=8192 自适应分块，tileNum 大幅减少以降低循环/调度开销，同时加入 ALIGN_SIZE=16 内存对齐

#### 提交步骤
1. 在 `2026/CANN-Summer-Camps` 目录下找到自己学校的目录并进入。
2. 新建一个以你的 GitCode 用户名为名称的文件夹（此处为 `KiritoBP`）。
3. 参考 `chenhao` 目录下每个课程的文件夹结构（`Lesson1`、`Lesson2`、`Lesson3`）。
4. 补全各课程的核心代码。
5. 完成开发后，在对应目录下执行验证命令：
   ```bash
   bash run.sh
   ```
