# TruncateDiv

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A2 训练系列产品 | √ |
| Atlas A3 系列产品 | √ |

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能说明

TruncateDiv 对输入张量 `x1` 和 `x2` 执行逐元素除法，并将商向零方向截断：

$$
y_i = trunc(\frac{x1_i}{x2_i})
$$

例如，`7 / 3` 的结果为 `2`，`-7 / 3` 的结果为 `-2`。输入支持 NumPy 风格双向广播。

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :---: | :--- | :--- | :---: |
| x1 | 输入 | 被除数张量 | FLOAT16、FLOAT、BFLOAT16、INT8、UINT8、INT32 | ND |
| x2 | 输入 | 除数张量，数据类型需与 `x1` 一致 | FLOAT16、FLOAT、BFLOAT16、INT8、UINT8、INT32 | ND |
| y | 输出 | 截断除法结果，数据类型与输入一致，shape 为广播后的 shape | FLOAT16、FLOAT、BFLOAT16、INT8、UINT8、INT32 | ND |

## 约束说明

- `x1` 与 `x2` 的数据类型必须一致。
- 输入维度数不超过 8。
- 输入 shape 必须满足广播关系：从末维开始对齐，每一维相等或其中一维为 1。
- 仅支持 ND 数据格式。
- 除数为 0 时的结果与底层硬件及内置 TBE 算子的行为保持一致，调用方不应依赖该场景的结果。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :---: | :--- | :--- |
| aclnn 调用 | [test_aclnn_truncate_div.cpp](./examples/test_aclnn_truncate_div.cpp) | 通过 [aclnnTruncateDiv](./docs/aclnnTruncateDiv.md) 两段式接口调用 TruncateDiv。 |

在 `ops-math` 仓库根目录执行构建，示例：

```bash
bash build.sh --pkg --op-name truncate_div
```

请确保构建使用的 CANN 版本与目标仓库分支配套，并已正确设置工具链环境。

## 目录结构

```text
TruncateDiv/
├── docs/
│   ├── aclnnTruncateDiv.md
│   └── design.md
├── examples/
├── op_api/
├── op_host/
├── op_kernel/
├── CMakeLists.txt
└── README.md
```

## 贡献说明

| 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| :--- | :---: | :---: | :--- |
| 第10小组 | TruncateDiv | 2026/07 | 新增 TruncateDiv Ascend C 实现、Host 侧适配、aclnn 接口、调用样例与设计文档 |
