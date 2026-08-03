# CANN Summer Camp 2026 · 西安交通大学

**提交人**：yuxinzhuo

本目录为课程作业提交，包含 `Lesson1`、`Lesson2`、`Lesson3` 三节课的完成实现。

---

## 目录结构

```
yuxinzhuo/
├── README.md                  # 本说明文档
├── Lesson1/                   # 矢量乘法算子（Ascend C 入门）
│   ├── CMakeLists.txt         # 构建配置，编译 mul_custom.asc -> mul_test
│   ├── mul_custom.asc         # 算子源码（已补全）：z = x * y，float，(8,2048)
│   └── run.sh                 # 构建并运行脚本
├── Lesson2/                   # 自定义减法算子（工程化开发）
│   ├── sub_custom_template.json   # 算子原型定义（msopgen 输入）
│   ├── run.sh                 # 编译-安装-测试一体化脚本
│   ├── custom_op/             # msopgen 生成的算子工程
│   │   ├── op_host/sub_custom_template.cpp        # Host 端 Tiling
│   │   ├── op_kernel/sub_custom_template.cpp      # 核函数（已补全）：z = x - y
│   │   ├── op_kernel/sub_custom_template_tiling.h # Tiling 数据结构
│   │   └── framework/tf_plugin/                   # TensorFlow 插件注册
│   └── test/main.cpp          # aclnn 测试用例
└── Lesson3/                   # 加法算子性能调优
    ├── run.sh                 # 构建 + msprof 性能分析 + 结果解析
    ├── custom_op/             # 调优后的算子工程（add_custom_template）
    │   ├── op_host/add_custom_template.cpp        # Host 端 Tiling
    │   ├── op_kernel/add_custom_template.cpp      # 核函数：z = x + y
    │   ├── op_kernel/add_custom_template_tiling.h # Tiling 数据结构
    │   └── test/              # 算子级独立测试
    └── test/main.cpp          # 顶层 aclnn 测试用例
```

---

## 作业说明

### Lesson 1
- **课程题目**：补充实现 `mul_custom.asc`，使用 Ascend C 编程实现矢量乘法算子 `mul_custom`。
- **数据类型**：输入与输出均为 `float` 类型。
- **数据形状（Shape）**：输入 Shape 为 `(8, 2048)`，输出 Shape 与输入相同。
- **数据布局（Format）**：ND。
- **实现**：采用 tiling 分块 + 双缓冲（`BUFFER_NUM=2`）流水设计，`KernelMul` 内实现 `Init / Process / CopyIn / Compute / CopyOut`；host 端 `kernel_mul` 完成设备内存分配与结果校验。

### Lesson 2
- **课程题目**：根据提供的算子原型文件 `sub_custom_template.json`，使用 `msopgen` 工具生成算子工程，并完成 Ascend C 核函数开发，实现矢量减法算子 `SubCustomTemplate`，计算 `z = x - y`。
- **算子功能**：矢量减法，`z = x - y`。
- **数据类型**：输入 `x`、`y` 和输出 `z` 均支持 `float16` 和 `float32`（原型中已定义）。
- **数据形状（Shape）**：输入 `x` 和 `y` 的形状为 `(8, 2048)`，输出 `z` 形状相同。
- **数据布局（Format）**：ND。
- **实现**：基于 `msopgen` 生成工程，补全 `op_kernel/sub_custom_template.cpp` 中的核函数（`VecSubOperator`，双缓冲队列，按 block/tile 切分）。

### Lesson 3
- **课程题目**：算子性能调优。
- **调优目标**：当前该算子的执行耗时约为 **98 µs**（使用 `msprof op` 在真实硬件上测得），尚未达到预期性能，需要进一步调优。在不改变算子功能的前提下，**将总耗时降低至 80 µs 以下**。
- **实现**：加法算子 `AddCustomTemplate`，结合 `msprof` 性能分析（`run.sh` 自动采集并解析 `OpBasicInfo.csv`），通过调整 block 数与 tile 划分进行调优。

---

## 运行方式

每个 Lesson 目录下均有对应脚本，在配置好 CANN 环境后执行：

```bash
cd Lesson1 && bash run.sh
cd Lesson2 && bash run.sh
cd Lesson3 && bash run.sh
```

---

## 参考来源

本作业参考了同校目录下其他文件夹的结构与实现，具体如下：

| 内容 | 参考文件夹 | 说明 |
|------|-----------|------|
| 整体工程结构（目录组织、run.sh、CMakePresets、op_host、framework、test 等） | `Chenhao` | 课程官方指定的标准模板结构 |
| Lesson1 内核 `mul_custom.asc` | `luohuashuang/Lesson1` | 补全乘法算子的完整实现 |
| Lesson2 内核 `sub_custom_template.cpp` | `balconysam/lesson2` | 补全减法算子的完整实现 |
| Lesson3 | `Chenhao/Lesson3` | 模板本身即为完整加法实现，无需填空 |
