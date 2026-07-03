# op_tester — 通用 Ascend C 算子测试工具

一个声明式的算子精度测试框架：用 **YAML** 描述算子规格（种类、输入输出数量/类型/属性），用 **Python** 写 PyTorch 参考实现（golden），工具自动生成不同 shape 的测试用例，加载算子 `.so` 调用 `torch.ops.npu.<op>` 与 golden 对比，并支持显式指定 shape。

## 特性

- **声明式配置**：算子规格写在 YAML，golden 写在独立 `.py`，复用性强、易批量管理。
- **按算子种类推导多输入 shape 关系**：`elementwise` / `broadcast` / `reduction` / `matmul` / `custom`。
- **自动随机生成 shape**：可配置维度数范围、每维范围、2 的幂偏置、用例数、种子。
- **显式指定 shape**：CLI `--shape` 或 YAML `shapes` 段，可附带属性。
- **属性系统**：支持 `default` / `choice` / `dim` / `random_int` / `random_float` 生成策略（如 reduction 的 `dim`、`keepdim`）。
- **多输出支持**：golden 与 NPU 算子均可返回 tuple/list。
- **多种输入初始化**：`random` / `zeros` / `ones` / `small` / `mixed` / `int`。
- **结果输出**：表格化终端输出 + 可选 JSON 导出。

## 安装

无需安装，直接以模块方式运行（需 `torch`、`torch_npu`、`pyyaml`）：

```bash
python -m op_tester <config.yaml> [options]
```

## 快速开始

以 `abs` 算子为例（假设算子工程已编译产出 `build/libabs_ops.so`）：

```bash
cd CANNBot/hpc_abs_verify/operators/abs
python -m op_tester /path/to/op_tester/configs/examples/abs.yaml
```

工具会：加载 `build/libabs_ops.so` → 注册 `torch.ops.npu.abs` → 自动生成 12 个随机 shape 用例 + 2 个显式 shape 用例 → 对每个用例生成输入、跑 NPU 算子、与 `abs_golden.compute` 对比 → 打印表格。

输出示例：

```
==============================================================================
Op: abs  kind=elementwise  inputs=1  outputs=1
atol=1e-06  rtol=1e-06  cases=14
==============================================================================
case  shapes                                          result         max_diff   time(ms)
------------------------------------------------------------------------------
A1    [4096]                                           PASS          0.000e+00      0.12
A2    [73,128]                                         PASS          0.000e+00      0.08
...
S1    [4096]                                           PASS          0.000e+00      0.05
S2    [512,512]                                        PASS          0.000e+00      0.21
------------------------------------------------------------------------------
Total: 14  Passed: 14  Failed: 0  =>  PASSED
==============================================================================
```

## YAML 配置说明

```yaml
op:
  name: add                 # torch.ops.npu.<name> 中的算子名（可带命名空间 myns.add）
  so: build/libadd_ops.so   # 算子 .so 路径（相对配置文件目录解析）
  kind: elementwise         # elementwise | broadcast | reduction | matmul | custom
  num_inputs: 2
  num_outputs: 1
  dtype: float32            # 默认 dtype（所有输入输出）
  # input_dtypes: [float32, float16]   # 可选：逐输入 dtype
  # output_dtypes: [float32]           # 可选：逐输出 dtype（仅文档用，对比按 golden 实际输出）
  input_init: [random, random]         # 可选：逐输入初始化策略
  attrs:                    # 算子属性（作为 kwargs 传给 npu op 与 golden）
    - {name: alpha, type: float, gen: random_float, low: -1.0, high: 1.0}
    - {name: dim, type: int, gen: dim}              # gen=dim 自动取有效维度
    - {name: keepdim, type: bool, gen: choice, choices: [false, true]}
  atol: 1e-6
  rtol: 1e-6
  golden:
    module: add_golden      # 同目录下的 add_golden.py
    func: compute           # 函数签名: compute(inputs: list[Tensor], attrs: dict) -> Tensor | list[Tensor]

shape_gen:                  # 自动 shape 生成参数
  ndim_min: 1
  ndim_max: 4
  dim_min: 1
  dim_max: 4096
  num_cases: 12
  seed: 0
  power_of_two_bias: 0.3    # 每维以该概率取 2 的幂（覆盖对齐边界）

shapes:                     # 可选：显式指定 shape 用例
  - [[8, 16], [8, 16]]      # 列表形式：每个子列表是一个输入的 shape
  - {shapes: [[4, 8, 16]], attrs: {dim: 1, keepdim: false}}   # 可附带属性
```

### golden 函数

```python
import torch

def compute(inputs, attrs):
    x, y = inputs[0], inputs[1]
    alpha = attrs.get("alpha", 1.0)
    return x + alpha * y
```

- `inputs`：CPU 上的 `torch.Tensor` 列表。
- `attrs`：属性字典（已按生成策略采样）。
- 返回单个 `Tensor` 或 `list/tuple[Tensor]`（多输出）。
- golden 在 CPU 上计算，作为标杆；NPU 算子结果搬回 CPU 后用 `torch.allclose(atol, rtol)` 对比。

## 算子种类与多输入 shape 推导

| kind         | 输入 shape 推导                                         |
|--------------|--------------------------------------------------------|
| elementwise  | 所有输入共享同一随机 shape                             |
| reduction    | 所有输入共享同一随机 shape；`dim` 属性自动取有效维度   |
| broadcast    | 输入 0 为基准 shape，其余输入随机将部分维度置为 1      |
| matmul       | 随机 `(M,K)` 与 `(K,N)`（`num_inputs>2` 时余者取 `(M,N)`） |
| custom       | 同 elementwise（如需特殊关系，在 golden 模块里自定义） |

## CLI 选项

```
python -m op_tester <config.yaml>
  --so PATH              覆盖 .so 路径
  -n, --num-cases N      覆盖自动用例数
  --shape SPEC           追加显式 shape 用例（可重复）
                         SPEC: '8,16;8,16' 或 '[[8,16],[8,16]]'
  --no-auto              只跑 --shape 指定的用例
  --seed N               覆盖随机种子
  --atol F / --rtol F    覆盖容差
  --json PATH            结果导出为 JSON
  -q, --quiet            静默
```

示例：

```bash
# 只跑两个指定 shape
python -m op_tester configs/examples/add.yaml --no-auto \
    --shape '1024,1024;1024,1024' --shape '[[8],[8]]'

# 增加 30 个随机用例并导出结果
python -m op_tester configs/examples/matmul.yaml -n 30 --json result.json
```

## 作为 Python 库使用

```python
from op_tester import load_config, run_tests

spec, config_dir = load_config("configs/examples/add.yaml")
results = run_tests(spec, config_dir, num_cases=20)
print(sum(r.passed for r in results), "/", len(results), "passed")
```

## 目录结构

```
op_tester/
├── __init__.py        # 公开 API
├── __main__.py        # python -m op_tester 入口
├── cli.py             # CLI
├── spec.py            # OpSpec / AttrSpec / ShapeGenConfig 数据类
├── config.py          # YAML 加载 + golden 模块加载
├── shape_gen.py       # shape 与输入张量生成
├── runner.py          # 测试执行与对比
├── README.md
└── configs/examples/  # 示例配置
    ├── abs.yaml, abs_golden.py
    ├── add.yaml, add_golden.py
    ├── sum.yaml, sum_golden.py
    └── matmul.yaml, matmul_golden.py
```

## 适配现有算子工程

现有工程（如 `CANNBot/hpc_abs_verify/operators/abs`）已自带 `scripts/golden.py` 与 `build/libabs_ops.so`，可直接复用：

```yaml
op:
  name: abs
  so: build/libabs_ops.so
  kind: elementwise
  num_inputs: 1
  num_outputs: 1
  dtype: float32
  golden:
    module: scripts/golden   # 复用工程已有 golden
    func: compute_golden
```

> 注意：工程自带 `golden.py` 的函数签名若是 `compute_golden(x1, x2)`（位置参数），可在 YAML 中用自定义 `golden.py` 适配为 `compute(inputs, attrs)` 签名，或直接新写一个薄包装。
