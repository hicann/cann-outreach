# Gcd Custom Kernel 直调样例 (float16, 4D broadcast)

`gcd_custom` 算子的 Kernel 直调实现示例，计算 `out[i] = gcd(|self[i]|, |other[i]|)`。

支持 4D 张量的 NumPy 风格 broadcast，float16 数据类型，ND Format。

## 算子签名

```
gcd_custom(Tensor self, Tensor other) → Tensor out
```

- **self** (float16): 第一个输入，4D shape，支持 broadcast
- **other** (float16): 第二个输入，4D shape，支持 broadcast
- **out** (float16): 输出，shape = broadcast(self.shape, other.shape)
- **算法**: 欧几里得算法 (Euclidean Algorithm): `gcd(a,b) = gcd(b, a mod b)`

## 目标 4D Broadcast 示例

| self shape | other shape | broadcast 后 |
|-----------|-------------|-------------|
| [2, 1, 8, 16] | [2, 3, 1, 1] | [2, 3, 8, 16] |
| [4, 6, 16, 32] | [4, 6, 16, 32] | [4, 6, 16, 32] |
| [1, 1, 1, 1] | [2, 3, 8, 16] | [2, 3, 8, 16] |
| [4, 1, 8, 16] | [4, 6, 8, 16] | [4, 6, 8, 16] |

## 文件结构

```
├── op_kernel/
│   ├── gcd_custom_tiling.h     Tiling 常量 + GcdTilingData 结构体
│   └── gcd_custom_kernel.asc   KernelGcd 类 + gcd_custom_kernel 核函数
│                                   (欧几里得算法, float32 内部计算, MAX_GCD_ITER=24)
├── op_host/
│   ├── gcd_custom.asc          Host + main 入口（固定 [2,1,8,16]×[2,3,1,1] 测试）
│   └── data_utils.h            数据读写工具
├── op_extension/
│   ├── gcd_custom_torch.cpp    PyTorch host（含 broadcast + 边界修正）
│   ├── register.cpp            TORCH_LIBRARY 注册
│   └── ops.h                   函数声明
├── CMakeLists.txt              双 target：可执行文件 + libgcd_custom_ops.so
├── scripts/
│   ├── gen_data.py             测试数据生成 (4D broadcast)
│   ├── golden.py               numpy 欧几里得参考实现
│   ├── verify_result.py        精度验证 (rtol=1e-2, atol=1e-4)
│   └── test_torch.py           PyTorch 测试（7 个 broadcast 用例）
└── run.sh                      一键运行
```

## 快速开始

### 方式一：直调验证

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash run.sh
```

### 方式二：PyTorch 调用

```bash
bash run.sh --torch
```

或手动：

```python
import torch; import torch_npu
torch.ops.load_library("build/libgcd_custom_ops.so")

self_t = torch.randn(2, 1, 8, 16, dtype=torch.float16).npu()
other_t = torch.randn(2, 3, 1, 1, dtype=torch.float16).npu()
out = torch.ops.npu.gcd_custom(self_t, other_t)
```

## 算法与精度

### 核函数内实现 (Kernel, float32 内部计算)

```
对于每个元素对 (a, b):
  1. a_f32 = cast_f16_to_f32(|self|)    # 提升到 float32 避免除溢
     b_f32 = cast_f16_to_f32(|other|)
  2. for iter in 0..23:                 # MAX_GCD_ITER=24
       b_f32 = max(b_f32, 1e-10)        # 防除零
       r = a_f32 - floor(a_f32/b_f32) * b_f32   # float 取模
       a_f32, b_f32 = b_f32, r          # 欧几里得交换
  3. out = cast_f32_to_f16(a_f32)
```

> **注意**：float16 除法易溢出（65504 ÷ 6e-8 ≈ 1e12 > float16 max 65504），因此内部提升到 float32 计算。

### Host 边界修正 (Torch extension)

Kernel 使用 `Max(b, 1e-10)` 防止除零，但这会使 `gcd(x, 0)` 的收敛结果偏离 `|x|`。Host 层通过 `torch.where` 精确修正：

```python
out = torch.where(other == 0, abs(self), out)   # gcd(x, 0) = |x|
out = torch.where(self == 0, abs(other), out)   # gcd(0, y) = |y|
```

### 精度说明

Kernel 内部使用 float32 计算，最终 cast 回 float16 输出：

| 场景 | 预期误差 | 说明 |
|------|---------|------|
| 常规正数 | 0（精确） | float32 计算无精度损失 |
| 含零输入 | 0 (Host 修正) | 边界值精确 |
| Cast float32→float16 | ~1e-3 relative | float16 尾数仅 11bit 的量化误差 |

- 验证容差: rtol=1e-2, atol=1e-4

## 相关 Skills

- `ascendc-direct-invoke-template` — Kernel 直调工程模板
- `ascendc-abs-operator` — Abs 算子 (单输入 Vector 算子的简单示例)
- `ascendc-tiling-design` — Tiling 设计指南
- `npu-arch` — NPU 架构知识
- `torch-ascendc-op-extension` — PyTorch 对接
