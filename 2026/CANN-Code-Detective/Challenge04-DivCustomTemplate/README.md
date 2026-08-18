# Challenge 04 · DivCustomTemplate —— 工程化算子开发

## 题目背景

前几期题目中，题目要求直接在单个 `.asc` 文件中完成核函数（kernel）开发，并使用 kernel 直调的方式在 host 侧进行验证。本期将切换为**工程化的算子开发方式**：基于提供的**算子原型文件（JSON）**，使用 **`msopgen` 工具自动生成完整算子工程**，再在生成的工程中完成 Ascend C 核函数与 host 侧 tiling 的实现，最终将算子编译、安装为自定义算子包，并通过 **ACLNN（AscendCL Neural Network）接口**完成调用与精度验证。

本题以矢量除法 `DivCustomTemplate`（`z = x / y`）为例，帮助参与者掌握从"原型定义 → 工程生成 → 算子实现 → 编译安装 → 接口调用"的完整工程化开发流程。


## 算子描述

| 项目 | 说明 |
| :--- | :--- |
| 算子名称 | `DivCustomTemplate` |
| 算子功能 | 矢量除法，`z = x / y` |
| 输入 | `x`、`y`：支持 `float16` 和 `float32`（原型中已定义） |
| 输出 | `z`：支持 `float16` 和 `float32` |
| 数据形状（Shape） | 输入 `x`、`y` 形状为 `(8, 2048)`，输出 `z` 形状相同 |
| 数据布局（Format） | ND |

## 代码骨架

代码骨架位于 [Template/DivCustomTemplate/](./Template/DivCustomTemplate/) 目录，请复制一份并以你的 gitcode 用户名命名，例如：

```bash
cp -r Template/DivCustomTemplate <你的用户名>/DivCustomTemplate
```

骨架中**仅提供**以下内容，算子工程本身需要你自行生成并完成：

- [div_custom_template.json](./Template/DivCustomTemplate/div_custom_template.json) —— 算子原型定义文件（已提供）
- [run.sh](./Template/DivCustomTemplate/run.sh) —— 编译运行脚本（已提供）
- [test/](./Template/DivCustomTemplate/test/) —— 测试代码（已提供）
  - `main.cpp` —— 调用 `aclnnDivCustomTemplate` 接口的测试代码
  - `CMakeLists.txt` —— 测试代码的 CMake 构建脚本（可选，run.sh 默认使用 g++ 直接编译）
- [custom_op/](./Template/DivCustomTemplate/custom_op/) —— 算子工程目录（需要你使用 `msopgen` 工具自行生成）
> **注意**：`custom_op/` 算子工程目录不在骨架中，需要你使用 `msopgen` 工具自行生成。

## 你需要完成的内容

### 第 1 步：使用 msopgen 生成算子工程

在 `Template/DivCustomTemplate` 复制出的目录中，基于原型文件生成算子工程：

```bash
msopgen gen -i div_custom_template.json -c ai_core-ascend910b1 -lan cpp -out ./custom_op
```

> **注意**：`-c` 参数需根据实际芯片的 SoC 版本设置，可通过 `python3 -c "import acl; print(acl.get_soc_name())"` 查询当前环境芯片型号，并参考 [msopgen 文档](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msOT/Operatordevelopmenttools/docs/zh/quick_start/msopgen_quick_start.md) 确认对应的 `ai_core-xxx` 参数。

生成后的工程目录结构如下（以 `DivCustomTemplate` 为例）：

```
custom_op/
├── CMakeLists.txt          # 顶层构建脚本
├── CMakePresets.json       # CMake 预设配置
├── build.sh                # 编译打包脚本
├── framework/              # 框架适配插件（如 tf_plugin）
├── op_host/                # host 侧：算子注册、InferShape、Tiling
│   ├── CMakeLists.txt
│   └── div_custom_template.cpp
└── op_kernel/              # kernel 侧：Ascend C 核函数实现
    ├── CMakeLists.txt
    ├── div_custom_template.cpp
    └── div_custom_template_tiling.h
```

### 第 2 步：完成算子实现

在生成的工程中补充实现，实现方式不唯一以下是参一种考实现：

1. **op_kernel/div_custom_template.cpp** —— 完成核函数 `div_custom_template` 的实现。可参考前几期的 Ascend C 编程范式：
   - 使用 `GlobalTensor<T>` 管理 GM 数据、`LocalTensor<T>` 管理 UB 数据
   - 使用 `TQue<T, BUFFER_NUM>` 管理 UB 队列
   - 使用 `DataCopy` 完成数据搬运
   - 使用 `Div`（或 `Muls`/`Adds` 等标量运算组合）完成 `z = x / y` 逐元素计算
   - 建议补充核函数实现后，添加必要的注释说明实现思路

2. **op_kernel/div_custom_template_tiling.h** —— 确认/完善 tiling 数据结构。

3. **op_host/div_custom_template.cpp** —— 根据算子实际需求完善 tiling 计算。

### 第 3 步：编译、安装并验证

直接执行 `run.sh`，脚本会自动完成"生成工程（若不存在）→ 编译 → 安装 → 编译测试 → 运行测试"全流程：

```bash
bash run.sh
```

预期输出：
```
result is:
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5
test pass
```

> 测试代码中 `x = 1.0`、`y = 2.0`，期望输出 `z = x / y = 0.5`。

## 实现提示

- **核函数签名**：生成的核函数骨架为
  ```cpp
  extern "C" __global__ __aicore__ void div_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z,
      GM_ADDR workspace, GM_ADDR tiling) {
      // TODO: user kernel impl
  }
  ```
  需要结合 `REGISTER_TILING_DEFAULT` / `GET_TILING_DATA` 获取 tiling 数据并实现。


- **数据类型**：原型支持 `float16` 和 `float32`，测试代码默认使用 `float16`。核函数实现时注意保持输入输出类型一致。

## 如何运行

```bash
# 在复制出的参与者目录下执行（如 <你的用户名>/DivCustomTemplate）
# 注意：在线环境一般不用修改脚本，可根据实际情况修改脚本中激活 CANN 环境的路径
bash run.sh
```

若 `custom_op/` 尚未生成或尚未完成核函数实现，`run.sh` 会在生成工程后提示你补全实现，补全后重新执行 `bash run.sh` 即可。

## 评分标准

- ✅ 使用 `msopgen` 生成算子工程并完成 `div_custom_template` 核函数实现，通过 ACLNN 接口验证输出 `test pass`
- ✅ 理解并能在 PR 描述中简要说明算子工程的目录结构和各文件职责
- ✅ 代码风格规范，注释清晰

## 参考学习资料

- [工程化算子开发介绍](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development/03_intermediate_vector_operator_development/03.02_operator_engineering_intro.ipynb)
- [AscendC 算子基础编程](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development/02_AscendC_basic/02.01_chapter_intro.ipynb)
- [msopgen 文档](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msOT/Operatordevelopmenttools/docs/zh/quick_start/msopgen_quick_start.md)

完成学习后，再回到本题目尝试独立实现。
