## 作业说明

### Lesson 1
- **课程题目**：补充实现 `mul_custom.asc`，使用 Ascend C 编程实现矢量乘法算子 `mul_custom`。
- **数据类型**：输入与输出均为 `float` 类型。
- **数据形状（Shape）**：输入 Shape 为 `(8, 2048)`，输出 Shape 与输入相同。
- **数据布局（Format）**：ND。

### Lesson 2
- **课程题目**：根据提供的算子原型文件 `sub_custom_template.json`，使用 `msopgen` 工具生成算子工程，并完成 Ascend C 核函数开发，实现矢量减法算子 `SubCustomTemplate`，计算 `z = x - y`。
- **算子功能**：矢量减法，`z = x - y`。
- **数据类型**：输入 `x`、`y` 和输出 `z` 均支持 `float16` 和 `float32`（原型中已定义）。
- **数据形状（Shape）**：输入 `x` 和 `y` 的形状为 `(8, 2048)`，输出 `z` 形状相同。
- **数据布局（Format）**：ND。

### Lesson 3
- **课程题目**：根据提供的算子模板，完成 `AddCustomTemplate` 算子的 Ascend C 核函数开发，实现矢量加法算子，计算 `z = x + y`。
- **算子功能**：矢量加法，`z = x + y`。
- **数据类型**：输入 `x`、`y` 和输出 `z` 均支持 `float16` 和 `float32`。
- **数据形状（Shape）**：输入 `x` 和 `y` 的形状为 `(45, 20480)`，输出 `z` 形状相同。
- **数据布局（Format）**：ND。
- **文件说明**：
  - `run.sh`：一键编译部署和测试脚本
  - `custom_op/`：自定义算子工程目录
    - `op_kernel/add_custom_template.cpp`：Ascend C 核函数实现（矢量加法）
    - `op_kernel/add_custom_template_tiling.h`：Tiling 数据结构定义
    - `op_host/add_custom_template.cpp`：算子注册、形状推导及 Tiling 实现
    - `framework/tf_plugin/`：TensorFlow 插件注册
  - `test/main.cpp`：测试代码，验证算子正确性（期望输出 3.0）

#### 提交步骤
1. 在 `2026/CANN-Summer-Camps` 目录下找到自己学校的目录并进入。
2. 新建一个以你的 GitCode 用户名为名称的文件夹（例如 `Chenhao`）。
3. 参考 `Chenhao` 目录下每个课程的文件夹结构（`Lesson1`、`Lesson2`、`Lesson3`）。
4. 以 `Lesson1` 为例，将 `Chenhao/Lesson1` 复制到自己名字的文件夹中，然后补全 `mul_custom.asc` 中的代码。
5. 完成开发后，在对应目录下执行验证命令：
   ```bash
   bash run.sh

