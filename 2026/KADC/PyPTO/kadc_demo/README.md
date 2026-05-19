# KADC Test — PyPTO 算子开发分级练习

基于 **PyPTO 框架**开发华为昇腾 NPU 自定义算子的分级练习与参考答案仓库，按难度递进组织，覆盖从基础逐元素算子到高级稀疏注意力算子的完整学习路径。

## 目录结构

```
kadc_test/
├── 01-beginner/                        # 入门级
│   ├── elewise_ops/                    #   逐元素算子: y = a + |b|
│   │   ├── README.md                   #     题目描述
│   │   ├── add_abs.py                  #     练习骨架（待填写）
│   │   └── add_abs_in_agent.md         #     Agent 辅助描述
│   └── matmul_add_ops/                 #   矩阵算子: y = a @ b^T + c
│       ├── README.md                   #     题目描述
│       ├── matmul_add.py               #     练习骨架（待填写）
│       └── matmul_add_in_agent.md      #     Agent 辅助描述
├── 02_intermediate/                    # 中级
│   └── softmax/                        #   Softmax 算子
│       ├── README.md                   #     题目描述 + API 参考
│       ├── softmax.py                  #     练习骨架（待填写）
│       └── softmax_in_agent.md         #     Agent 辅助描述
├── 03-advanced/                        # 高级
│   └── attention/                      #   DeepSeek V3/V2 稀疏 Flash Attention
│       ├── README.md                   #     题目描述 + API 参考
│       ├── deepseekv32_sparse_flash_attention_quant.py   # 测试入口与 golden 生成
│       ├── sparse_flash_attention_quant_impl.py           # PyPTO kernel 实现（参考答案）
│       └── utils/compare.py            #     精度对比工具
└── kadc_answers/                       # 参考答案（可直接运行的完整 kernel 实现）
    ├── add_abs.py                      #   01 elewise_ops 参考答案
    ├── matmul_add.py                   #   01 matmul_add_ops 参考答案
    ├── softmax.py                      #   02 softmax 参考答案
    └── sparse_flash_attention_quant_impl.py   # 03 attention 参考答案
```

## 难度递进

| 级别 | 算子 | 计算公式 | 核心知识点 | 涉及的 PyPTO API |
|------|------|----------|-----------|-----------------|
| **01 入门** | `add_abs` | `y = a + \|b\|` | 逐元素运算、JIT 编译、动态维度、tiling 配置 | `pypto.frontend.jit`, `pypto.abs`, `pypto.add`, `pypto.set_vec_tile_shapes`, `pypto.loop` |
| **01 入门** | `matmul_add` | `y = a @ b^T + c` | 矩阵乘法、Cube/Vector 协作、BF16 精度 | `pypto.matmul`, `pypto.set_cube_tile_shapes`, `pypto.cast` |
| **02 中级** | `softmax` | `softmax(x) = exp(x-max) / sum(exp(x-max))` | 归约操作 (amax/sum)、逐元素算子链 (sub/exp/div)、全动态形状 | `pypto.amax`, `pypto.exp`, `pypto.sum`, `pypto.div` |
| **03 高级** | Sparse Flash Attention | DeepSeek V3/V2 MLA 稀疏注意力 | 5 层嵌套循环、PagedAttention Gather、INT8 量化/反量化、Flash online softmax、多芯片适配 (910B/950) | 以上全部 + `pypto.loop_unroll`, `pypto.cond`, `pypto.view`, `pypto.assemble`, `pypto.gather_in_ub/l1`, `pypto.maximum` |

## 各级详细说明

### 01-beginner：入门级

从最简单的逐元素算子和矩阵算子入手，掌握 PyPTO 开发的基本范式：

- **JIT 编译**：使用 `@pypto.frontend.jit` 装饰器注册 kernel
- **动态维度**：使用 `pypto.DYNAMIC` 声明运行时变化的维度
- **Tiling 配置**：通过 `set_vec_tile_shapes` / `set_cube_tile_shapes` 控制硬件执行效率
- **循环处理**：使用 `pypto.loop` 对动态维度进行逐 tile 处理

### 02-intermediate：中级

在入门基础上引入归约和数值稳定性处理：

- **数值稳定 Softmax**：减最大值防溢出 → exp → sum → 除法归一化
- **全动态形状**：所有维度均为 `pypto.DYNAMIC`
- **单算子内多步计算**：在同一个 kernel 中串联多个基本操作

### 03-advanced：高级

面向真实大模型场景的完整算子实现：

- **MLA 架构**：Query/Key 拆分为 nope + rope 子向量
- **PagedAttention**：通过 `block_table` 映射不连续 KV Cache，使用 `gather_in_ub` / `gather_in_l1` 稀疏索引
- **KV Cache INT8 量化**：逐 block 量化/反量化，减少显存占用
- **Flash Online Softmax**：跨 tile 增量更新 oi/li/mi，减少中间精度损失
- **多芯片适配**：910B 与 950 使用不同的 TileShape 配置
- **两种推理模式**：Decode (s1=1/2, 标准 softmax) 和 Prefill (s1=256, Flash online softmax)

## 使用方式

### 环境准备

```bash
# 配置 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置设备 ID
export TILE_FWK_DEVICE_ID=0
```

### 练习流程

1. 进入题目目录，阅读 `README.md` 了解算子规格和 API 参考
2. 在对应的 `.py` 骨架文件中填写 PyPTO kernel 实现
3. 运行测试验证精度

### 查看参考答案

```bash
# 直接运行参考答案
python kadc_answers/add_abs.py          # 入门: 逐元素
python kadc_answers/matmul_add.py       # 入门: 矩阵
python kadc_answers/softmax.py          # 中级: Softmax

# 高级: Attention 需在 03-advanced/attention/ 目录下运行
cd 03-advanced/attention/
python deepseekv32_sparse_flash_attention_quant.py
```

## 环境要求

- CANN 工具链（Ascend 910B / 950PR）
- PyPTO 框架
- PyTorch + torch_npu
- `TILE_FWK_DEVICE_ID` 环境变量已设置
