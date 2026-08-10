# CANN Code Detective —— 代码侦探

## 活动简介

**CANN Code Detective** 是面向 Ascend C算子开发者的系列编程挑战活动。每一期我们会发布一个算子相关的编程题目，参与者需要根据提供的参考资料学习Ascend C算子开发相关知识，并基于提供的代码骨架完成算子开发任务，并在验证正确性。

活动旨在帮助开发者：
- 学习理解 Ascend C 算子编程范式，由浅入深覆盖算子开发、调试调优等内容

## 活动规划

| 期数 | 主题 | 题目说明 | 难度 | 状态 |
| :--: | :--- | :------ | :--: | :--: |
| 01 | DivCustom | 实现两个输入张量的逐元素除法运算 | ★ | 已发布 |
| 02 | DivBf16Custom | 使用 TBuf 管理 bfloat16 类型转换的临时内存，实现 bfloat16 除法 | ★ | 已发布 |
| 03 | CoshCustom | 实现双曲余弦（cosh）算子 | ★ | - |
| 04 | 待发布 |  |  | 未开始 |
| 05 | 待发布 |  |  | 未开始 |
| 06 | 待发布 |  |  | 未开始 |
| 07 | 待发布 |  |  | 未开始 |
| 08 | 待发布 |  |  | 未开始 |

> 将视情况新增后续期数。

## 目录结构

```
CANN-Code-Detective/
├── README.md                      # 活动总说明（本文件）
├── Challenge01-DivCustom/         # 第 01 期：逐元素除法算子
│   ├── README.md                  # 题目说明
│   └── Template/DivCustom/        # 官方代码骨架（请勿直接修改）
│       ├── CMakeLists.txt         # 构建脚本
│       ├── run.sh                 # 编译运行脚本
│       └── div_custom.asc         # 算子实现骨架（待补充）
│   └── <你的用户名>/DivCustom/    # 你的实现（复制 Template 后改名）
├── Challenge02-DivBf16Custom/     # 第 02 期：bfloat16 除法算子
│   ├── README.md                  # 题目说明
│   └── Template/DivBf16Custom/    # 官方代码骨架（请勿直接修改）
│       ├── CMakeLists.txt         # 构建脚本
│       ├── run.sh                 # 编译运行脚本
│       └── div_bf16_custom.asc    # 算子实现骨架（待补充 Compute等实现）
│   └── <你的用户名>/DivBf16Custom/ # 你的实现（复制 Template 后改名）
├── Challenge03-xxx/              # 第 03 期
│   └── ...
└── ...
```

## 参与方式

1. **获取题目**：进入对应期数目录（如 `Challenge01-DivCustom`），阅读该目录下的 `README.md` 了解题目要求。
2. **复制骨架**：将题目模板目录复制一份，并将目录重命名为你的 gitcode 用户名，例如 `cp -r Template/DivCustom <你的用户名>/DivCustom`。请勿直接修改 Template 目录。
3. **补充实现**：在你的目录中，按照注释提示补充算子实现代码。
4. **本地验证**：参考 README.md 中的运行说明，在NPU 环境编译运行，确保输出 `[Success]` 通过验证。
5. **提交 PR**：将你的实现通过 Pull Request 提交到本仓库，等待评审。提交文件题目模板，不要删除辅助脚本文件，也不要增加编译缓存等冗余文件。

## 提交与评审

- 每期题目对应一个目录，请在**对应期数目录**下新建一个自己gitcode用户名命名的目录存放你的实现代码，比如第一期的目录为 `2026/CANN-Code-Detective/Challenge01-DivCustom/<你的gitcode用户名>/DivCustom/`。
- PR 标题建议格式：标题前添加类似`【代码侦探Challenge01-DivCustom】`标识，方便代码检视，具体格式以每一次活动发布要求为准。
- 评审会关注：功能正确性、代码规范性，通过评审的 PR 将被合并，优秀实现将作为参考样例展示。
- 代码贡献提交PR需要签署CLA，可参考本仓库[README](https://gitcode.com/guest_shao/cann-outreach/blob/master/README.md)获取签署CLA/提交PR的指导说明

## 环境要求

- Atlas A2系列产品（在线环境中算力类型对应：昇腾910B4/昇腾910B3）
- CANN 9.0.0
- 建议直接使用在线开发环境，包括[Hidevlab在线开发](https://hidevlab.huawei.com/online-develop-intro)、[notebook在线环境](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/first_custom_operator&scanFilePath=quick_start/first_custom_operator/first_custom_operator.ipynb)等。

## 常见问题

**Q: 我是新手，不知道从何入手？**
A: 建议先学习参考课程：[AscendC 算子开发](https://gitcode.com/cann/cann-learning-hub/tree/master/tutorials/ascendc_operator_development)，掌握基础后再从第 01 期题目开始挑战。

**Q: 没有 NPU 环境怎么办？**
A: 1、可在[Hidevlab](https://hidevlab.huawei.com/online-develop-intro)申请Atlas A2算力资源，申请中注意填写项目名称为`CANN Code Detective`，申请完成审批后即可使用在线WebIDE中进行开发。**注意：创建环境时算力类型请选择`昇腾910B4`/`昇腾910B3`，镜像选择`CANN9.0.0 Release, Ubuntu22.04`**；2、可以在[notebook在线环境](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/first_custom_operator&scanFilePath=quick_start/first_custom_operator/first_custom_operator.ipynb)中直接开发。

**Q: 可以使用 AI 辅助编程吗？**
A: 鼓励以学习为目的合理使用 AI 工具，但请确保你理解每一行代码的含义，可以通过注释说明你的实现思路。

## 贡献者

感谢每一位参与挑战的开发者。你的每一行代码，都是CANN生态的一份力量。

---

*本活动由 CANN Outreach 团队组织，保留最终解释权。*
