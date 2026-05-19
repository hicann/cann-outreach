# DeepSeek V3/V2 Sparse Flash Attention with Quantization

基于 PyPTO 实现的 DeepSeek V3/V2 MLA（Multi-head Latent Attention）稀疏注意力算子，支持 KV Cache INT8 量化和 PagedAttention，运行于华为昇腾 NPU。

## 算法概述

本算子实现了 DeepSeek V3/V2 架构中的稀疏注意力机制，核心特征如下：

- **MLA 压缩表示**：Query/Key 被拆分为 `nope`（低秩压缩部分，`kv_lora_rank=512`）和 `rope`（旋转位置编码部分，`qk_rope_dim=64`）两个子向量，拼接后参与注意力计算。
- **稀疏 Top-K 选择**：每个 query token 仅关注从 KV Cache 中选出的 `topk` 个 key-value 对（默认 topk=2048），而非全序列。
- **KV Cache INT8 量化**：Key 的 `nope` 部分支持 INT8 逐 block 量化（128 元素为一组），配合 FP32 scale 反量化后参与计算。
- **PagedAttention**：KV Cache 以 block（block_size=128）为粒度管理，通过 `block_table` 映射物理位置，支持不连续内存布局。
- **变长序列**：每个 batch 可有不同的实际序列长度（`actual_seq`）。

## 算子规格

| 项目 | 说明 |
|------|------|
| 算子名称 | sparse_flash_attention_quant |
| 数据类型 | BF16 (query/key_nope/key_rope/output), INT8 (可选 key_nope), FP32 (scales) |
| 精度标准 | rtol=0.005, atol=0.0001 |
| 动态轴 | query_nope/query_rope/topk_indices/block_table/kv_act_seqs 的首维为动态 |
| 推理模式 | Decode (s1=1/2, 标准 softmax) / Prefill (s1=256, Flash online softmax) |

### 输入输出

| 参数 | 方向 | shape | dtype | 说明 |
|------|------|-------|-------|------|
| query_nope | 输入 | (B*S1*N_Q, kv_lora_rank) | BF16 | Query 低秩压缩部分 |
| query_rope | 输入 | (B*S1*N_Q, qk_rope_dim) | BF16 | Query 旋转位置编码部分 |
| key_nope_2d | 输入 | (block_num*block_size, kv_lora_rank) | BF16 / INT8 | Key 低秩压缩部分 (KV Cache) |
| key_rope_2d | 输入 | (block_num*block_size, qk_rope_dim) | BF16 | Key 旋转位置编码部分 (KV Cache) |
| k_nope_scales | 输入 | (block_num*block_size, 4) | FP32 | Key INT8 反量化 scale (每 128 元素一组) |
| topk_indices | 输入 | (B*S1, N_KV*topk) | INT32 | 每个 query token 的 top-k 索引 |
| block_table | 输入 | (B, max_blocknum_perbatch) | INT32 | PagedAttention block 映射表 |
| kv_act_seqs | 输入 | (B,) | INT32 | 每个 batch 的实际 KV 序列长度 |
| attention_out | 输出 | (B, S1, N_Q, kv_lora_rank) | BF16 | 注意力计算结果 |

## 文件结构

```
attention/
├── deepseekv32_sparse_flash_attention_quant.py   # 测试入口与 golden 生成
├── sparse_flash_attention_quant_impl.py           # PyPTO kernel 实现
├── README.md
└── utils/
    └── compare.py                                 # 精度对比工具
```

## 实现版本

| 函数 | 模式 | 算法 | 适用芯片 |
|------|------|------|----------|
| `sparse_flash_attention_quant_d` | Decode | 标准 softmax | 910B |
| `sparse_flash_attention_quant_d_950` | Decode | 标准 softmax | 950 |
| `sparse_flash_attention_quant_p` | Prefill | Flash Attention（online softmax） | 910B |

### Decode 模式（`sparse_flash_attention_quant_compute`）

- 使用标准 softmax 归一化：`softmax = exp(S - max(S)) / sum(exp(S - max(S)))`
- 每次 s2 tile 计算后直接得到归一化结果，写入输出
- 适用于 s1 较小（如 s1=1 或 s1=2）的 decode 场景

### Prefill 模式（`sparse_flash_attention_quant_compute_flash`）

- 使用 Flash Attention 算法的 online softmax：维护 `oi_update`（累加输出）、`li_update`（累加归一化因子）、`mi_update`（累加最大值）三个运行状态
- 跨 s2 tile 增量更新：`mi_new = max(mi, tilda_mij)` → 修正历史累加值 → 归一化
- 仅在最后一个 s2 tile 时做最终归一化，减少中间精度损失
- 适用于 s1 较大（如 s1=256）的 prefill 场景

## 实现要点

### 1. 计算流水线（每个 s2 tile）

```
对每个 batch、每个 s1 token、每个 KV head 组:
  ├─ Sa_V0: Gather — 从 KV Cache 按 topk_indices 搬运 Key/Value 数据
  │   ├─ 若 INT8 量化: Gather INT8 kn + scales → 反量化 → BF16
  │   └─ 若 BF16: 直接 Gather BF16 kn
  ├─ Sa_C1: S = Q × K^T (BF16 → FP32 matmul)
  ├─ Sa_V1: Softmax(S * scale)
  │   ├─ Decode: 标准 softmax (exp-max / sum)
  │   └─ Prefill: Flash online softmax (exp-max, 不除 sum, 累积 oi/li/mi)
  ├─ Sa_C2: O = Softmax × V (BF16 matmul)
  └─ Sa_V2: Flash 归一化更新 (仅 Prefill 模式, 最后一个 tile 时 O = oi / li)
```

### 2. 涉及的 PyPTO API

#### 流程控制

| API | 用途 |
|-----|------|
| [`pypto.frontend.jit`](https://gitcode.com/cann/pypto/blob/master/docs/api/config/pypto-frontend-jit.md) | Kernel JIT 编译装饰器，配置 pass_options / runtime_options |
| [`pypto.loop`](https://gitcode.com/cann/pypto/blob/master/docs/api/controlflow/pypto-loop.md) | 生成硬件级循环（batch / s1 / n_kv / group / s2） |
| [`pypto.loop_unroll`](https://gitcode.com/cann/pypto/blob/master/docs/api/controlflow/pypto-loop_unroll.md) | 循环展开（s2 tile 维度） |
| [`pypto.cond`](https://gitcode.com/cann/pypto/blob/master/docs/api/controlflow/pypto-cond.md) | 条件分支（首个/末个 tile 判断） |

#### 张量构造与视图

| API | 用途 |
|-----|------|
| [`pypto.view`](https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-view.md) | 创建张量视图（切片 topk_indices / block_table / query / key） |
| [`pypto.reshape`](https://gitcode.com/cann/pypto/blob/master/docs/api/tensor/pypto-Tensor-reshape.md) | 张量形状变换（INT8 反量化 reshape 对齐） |
| [`pypto.concat`](https://gitcode.com/cann/pypto/blob/master/docs/api/tensor/pypto-Tensor-concat.md) | 张量拼接（扩展 INT8 列宽用于 reshape） |
| [`pypto.assemble`](https://gitcode.com/cann/pypto/blob/master/docs/api/tensor/pypto-Tensor-assemble.md) | 将子张量写入目标偏移位置（拼接 Key/Query, 写回输出） |

#### 计算算子

| API | 用途 |
|-----|------|
| [`pypto.matmul`](https://gitcode.com/cann/pypto/blob/master/docs/api/tensor/pypto-Tensor-matmul.md) | 矩阵乘法：C1(Q×K^T) 和 C2(Softmax×V) |
| [`pypto.amax`](https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-amax.md) | 沿指定维度求最大值（Softmax 数值稳定：减最大值防溢出） |
| [`pypto.exp`](https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-exp.md) | 逐元素指数运算（Softmax 核心） |
| [`pypto.sum`](https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-sum.md) | 沿指定维度求和（Softmax 归一化因子） |
| [`pypto.maximum`](https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-maximum.md) | 逐元素取最大值（Flash Attention: mi_new = max(mi, tilda_mij)） |
| [`pypto.mul`](https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-mul.md) | 逐元素乘法（scale × S, INT8 反量化, Flash 增量修正） |
| [`pypto.sub`](https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-sub.md) | 逐元素减法（S - max, Flash 增量修正因子） |
| [`pypto.add`](https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-add.md) | 逐元素加法（Flash 增量更新 li_new, oi_new） |
| [`pypto.div`](https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-div.md) | 逐元素除法（Softmax 归一化 / Flash 最终归一化 O = oi / li） |
| [`pypto.cast`](https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-cast.md) | 数据类型转换（INT8→FP16→FP32→BF16 反量化链路） |
| [`pypto.gather`](https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-gather.md) | 按 topk_indices 从 KV Cache 搬运数据（gather_in_ub / gather_in_l1 的底层依赖） |

#### Tiling 与编译配置

| API | 用途 |
|-----|------|
| [`pypto.set_vec_tile_shapes`](https://gitcode.com/cann/pypto/blob/master/docs/api/config/pypto-set_vec_tile_shapes.md) | 设置向量算子 tile 尺寸（Gather / Softmax / Flash 更新） |
| [`pypto.set_cube_tile_shapes`](https://gitcode.com/cann/pypto/blob/master/docs/api/config/pypto-set_cube_tile_shapes.md) | 设置 Cube matmul tile 尺寸（C1: Q×K^T, C2: Softmax×V） |
| [`pypto.set_matrix_size`](https://gitcode.com/cann/pypto/blob/master/docs/api/config/pypto-set_matrix_size.md) | 设置 matmul 矩阵尺寸 [M, K, N] |
| [`pypto.set_semantic_label`](https://gitcode.com/cann/pypto/blob/master/docs/api/config/pypto-set_semantic_label.md) | 设置语义标签（Sa_V0 / Sa_C1 / Sa_V1 / Sa_C2 / Sa_V2） |
| [`pypto.set_pass_options`](https://gitcode.com/cann/pypto/blob/master/docs/api/config/pypto-set_pass_options.md) | 编译期 Pass 选项（BF16 路径 scope 隔离） |

### 3. 关键设计决策

#### 5 层嵌套循环结构

Decode 与 Prefill 共享同一循环骨架：
- **L0 batch** → `pypto.loop`, Decode 可并行 (`parallel=True`), Prefill 串行
- **L1 s1** → `pypto.loop`, query 序列维度
- **L2 n_kv** → `pypto.loop`, KV head 维度 (GQA)
- **L3 group** → `pypto.loop`, GQA group 维度 (N_Q / N_KV = 128)
- **L4 s2** → `pypto.loop_unroll`, KV 序列 tile 维度 (unroll_list={1})

#### INT8 量化路径

Key `nope` 部分按每 128 元素分组量化（512 / 128 = 4 组）：
1. `gather_in_ub` 分别搬运 INT8 kn 和 FP32 scales
2. INT8 → FP16 → FP32 类型提升链
3. 逐组乘 scale 完成反量化
4. 转回 BF16 参与后续计算

#### Flash Online Softmax 增量更新

Prefill 模式跨 s2 tile 维护三个运行状态：
- `oi_update`: 累积注意力输出（未归一化）
- `li_update`: 累积 exp sum, shape=(1, group_tile)
- `mi_update`: 累积 max, shape=(1, group_tile)

每个后续 tile 的修正公式：
```
mi_new = max(mi, tilda_mij)
li_new = exp(mi - mi_new) * li + exp(tilda_mij - mi_new) * tilda_lij
oi_new = exp(mi - mi_new) * oi + exp(tilda_mij - mi_new) * q1
```
仅在最后一个 tile 时做最终归一化 `O = oi / li`。

## Tiling 配置

通过 `SaTileShapeConfig` 控制各级计算的 tile 参数：

```python
@dataclass
class SaTileShapeConfig:
    g_tile: int                   # GQA group tile 大小
    s_kv_tile: int                # KV 序列维度 tile 大小
    gather_vec_tile_shape: list   # Gather 向量算子 tile
    c1_tile_shape: list           # C1（Q×K^T）Cube 算子 tile [M0,M1, K0,K1, N0,N1]
    v1_tile_shape: list           # V1（Softmax）向量算子 tile
    c2_tile_shape: list           # C2（Attn×V）Cube 算子 tile [M0,M1, K0,K1, N0,N1]
    v2_tile_shape: list           # V2（Flash 归一化更新）向量算子 tile（仅 prefill 使用）
```

不同芯片/模式的默认配置：

| 参数 | 910B Decode | 910B Prefill | 950 Decode |
|------|-------------|--------------|------------|
| `g_tile` | 128 | 128 | 128 |
| `s_kv_tile` | 2048 | 2048 | 2048 |
| `gather_vec_tile_shape` | [32, 512] | [32, 512] | [64, 512] |
| `c1_tile_shape` | [128,128,128,128,128,128] | [128,128,128,128,128,128] | [128,128,128,128,64,64] |
| `v1_tile_shape` | [8, 2048] | [8, 2048] | [4, 2048] |
| `c2_tile_shape` | [128,128,128,128,128,128] | [128,128,128,128,128,128] | [128,128,128,128,128,128] |
| `v2_tile_shape` | [64, 256] | [64, 128] | [64, 256] |

## 测试用例

| 用例名 | B | S1 | N_Q | N_KV | 序列长度 | Key 量化 | 模式 | 芯片 |
|--------|---|----|-----|------|----------|----------|------|------|
| `sfa_bf16_b4_s2_seq64K_total_int8_d` | 4 | 2 | 128 | 1 | [65536, 16381, 666, 15] | INT8 | Decode | 910B/950 |
| `sfa_bf16_b4_s2_seq64K_per_int8_d` | 4 | 2 | 128 | 1 | [65536]×4 | INT8 | Decode | 910B/950 |
| `sfa_bf16_b4_s2_seq64K_per_bf16_d` | 4 | 2 | 128 | 1 | [65536]×4 | BF16 | Decode | 910B/950 |
| `sfa_bf16_b1_s256_seq64K_int8_p` | 1 | 256 | 128 | 1 | [65536] | INT8 | Prefill | 910B |
| `sfa_bf16_b4_s2_seq64K_per_int8_d_950` | 4 | 2 | 128 | 1 | [65536]×4 | BF16 | Decode | 950 |

精度容差：`atol=0.0001, rtol=0.005`。

## 运行方式

### 环境准备

```bash
# 配置 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置设备 ID
export TILE_FWK_DEVICE_ID=0
```

### 通过 pytest 运行

```bash
# 运行默认用例
pytest deepseekv32_sparse_flash_attention_quant.py -v

# 运行指定用例
pytest deepseekv32_sparse_flash_attention_quant.py::test_sfa_bf16_b4_s2_seq64k_total_int8_d -v

# 指定芯片类型
pytest deepseekv32_sparse_flash_attention_quant.py -v --soc=910B
```

### 直接运行

```bash
python deepseekv32_sparse_flash_attention_quant.py
```

默认执行 `test_sfa_bf16_b4_s2_seq64k_per_int8_d()`。可编辑 `__main__` 部分切换用例。

## 关键依赖

- `pypto`：PyPTO 算子开发框架
- `pypto.experimental.gather_in_ub` / `gather_in_l1`：PagedAttention block 级稀疏 Gather
- `torch` / `torch_npu`：PyTorch 及昇腾 NPU 后端
- `numpy`：数据生成辅助
- `pytest`：测试框架

## 环境要求

- CANN 工具链（Ascend 910B / 950PR）
- PyPTO 框架
- PyTorch + torch_npu
- 环境变量 `TILE_FWK_DEVICE_ID` 已设置（如 `export TILE_FWK_DEVICE_ID=0`）

## 注意事项

1. **环境要求**：需要可用 NPU 环境（`npu-smi info` 可检测到设备），`TILE_FWK_DEVICE_ID` 环境变量可指定设备编号（默认为 0）。
2. **JIT 编译**：三个 kernel 入口函数通过 `@pypto.frontend.jit` 装饰器注册，首次调用会触发编译。
3. **INT8 量化规则**：Key `nope` 部分按每 128 个元素分组求绝对最大值作为 scale，量化到 `[-128, 127]` 范围。
4. **Gather 算子**：使用 `pypto.experimental.gather_in_ub` / `gather_in_l1` 实现 PagedAttention 的 block 级稀疏索引。
5. **Tiling 调优**：算子性能高度依赖于 `set_vec_tile_shapes` / `set_cube_tile_shapes` 的设置，950 芯片因 UB/L1 容量不同需使用独立的 TileShape 配置。
6. **DYNAMIC Loop**：s2 tile 维度使用 `pypto.loop_unroll(unroll_list={1})`，首次调用时确定循环次数并编译，后续调用可使用更少的迭代次数，但不可超出首次编译时的循环次数。
7. **debug_options**：`sparse_flash_attention_quant_d`（910B Decode）开启了 `runtime_debug_mode` 和 `compile_debug_mode`，正式发布时应移除。
