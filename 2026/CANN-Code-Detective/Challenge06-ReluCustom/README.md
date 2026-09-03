# Challenge 06 · Relu —— 逐元素 ReLU 激活算子

## 题目背景

ReLU（Rectified Linear Unit）是深度学习中最常用的激活函数之一。给定输入张量 `x`，逐元素计算输出 `y = max(0, x)`。

本题为开源仓规范化算子开发：基于 Ascend C 在昇腾 NPU 上完成 ReLU 算子的 Tiling 函数（op_host 侧）与核函数实现（op_kernel 侧）。通过多核切分将输入数据分块搬入 Local Memory（UB），执行 `Relu` 指令逐元素计算，再将结果搬回 Global Memory。

## 算子描述

| 项目 | 说明 |
| :--- | :--- |
| 输入 | `x`：float32 / float16 张量，ND 格式，形状 `(8, 2048)` |
| 输出 | `y`：与 `x` 同 dtype、同形状 |
| 计算 | `y[i] = max(0, x[i])` |

## 涉及知识点

本题覆盖 Ascend C 开源仓算子开发的核心知识点：

- **开源仓工程结构**：`op_host`（算子注册 / 形状推导 / Tiling）、`op_kernel`（核函数）、`tests/ut`（单元测试）、`examples`（aclnn 调用示例）
- **Tiling 函数**（`ReluTilingFunc`）：获取输入 dtype → `ASCENDC_TPL_SEL_PARAM` 按 tilingKey 选择模板实例 → shape 各维相乘得到 `totalLength` → 设置 `tileNum` → `SetBlockDim` 配置启动核数 → 配置 workspace
- **核函数入口**：`__global__ __aicore__` 声明、`REGISTER_TILING_DEFAULT` / `GET_TILING_DATA_WITH_STRUCT` 解析 tiling 数据
- **Vector 编程范式**：`CopyIn → Compute → CopyOut` 三段式流水线
- **多核数据切分**：`blockLength = totalLength / GetBlockNum()`，`SetGlobalBuffer` 按 `GetBlockIdx()` 设置每核 GM 起始地址
- **队列与内存管理**：`TPipe` / `TQue`、`InitBuffer` 分配 UB、`AllocTensor` / `FreeTensor`、`EnQue` / `DeQue`
- **基础指令**：`DataCopy` 数据搬运、`AscendC::Relu` 逐元素计算
- **性能优化**：调整 `tileNum`、`BUFFER_NUM = 2` 实现 Double Buffer 双缓冲

## 代码编写

1. 用 AtomGit 账号登录 [CANNJudge 平台](https://cannjudge.cn/home)
2. 加入《代码侦探》小组，完成 [Lesson06_relu](https://cannjudge.cn/cann_code_detective/challenge06-relucustom) 答案编写，操作详见 [gitcode 讨论 #287](https://gitcode.com/org/cann/discussions/287)
3. 通过平台验证并提交 PR

特别注意：

- ✅ 通过验证集：提交代码后验证通过显示 "PASS"
- ✅ 提交开源仓：填写邮箱、学校/企业代码缩写（后续作为目录，只支持英文、数字、下划线）、学校/企业全称、个人英文标识（gitcode 用户名去掉 @+下划线+题号，例如“wx234234234_06”），按照指引创建令牌并签署 CLA

## 推荐学习路径

1. 先学习 [Ascend C 算子编程基础（第 2 章）](https://gitcode.com/cann/cann-learning-hub/tree/master/tutorials/ascendc_operator_development/02_AscendC_basic)，掌握核函数开发与 Vector 编程范式
2. 再学习 [开源仓算子开发（第 6 章）](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development/06_opensource_repo_operator_intro_and_contribution/06.03_operator_development_based_on_opensource_repo.ipynb)，掌握开源仓工程结构与 Tiling 开发