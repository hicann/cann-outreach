# 东南大学

## 团队信息

- 提交者: 吴尚扬_61524H29
- 身份: 学生
- 单位: 东南大学

## 成员

- 吴尚扬_61524H29 (kk_axilter): 提交者

## 算子: op_10_gelu

基于 Ascend C 实现 GELU 算子。Host 侧将任意维输入展平，按照每核约 512 个元素进行多核切分，并根据 UB 容量动态确定 Tile 大小，所有分块按 32B 对齐；单 Tile 使用单缓冲，多 Tile 启用 Double Buffer，以重叠数据搬运与计算。Kernel 侧针对数据类型采用差异化实现：float16 使用高精度 GELU 接口并动态分配临时空间；float32 使用五阶 Sigmoid 近似计算 GELU，通过区间截断和尾部选择保证数值稳定。输入输出尾块使用 DataCopyPad 处理，兼容非 32B 对齐及任意多维 Shape。
