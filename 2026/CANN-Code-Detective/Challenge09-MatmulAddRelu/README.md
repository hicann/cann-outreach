# Challenge 09 · MatmulAddRelu —— Catlass Python 融合算子

题目目录（相对于 `cann-outreach` 仓库根目录）：`2026/CANN-Code-Detective/Challenge09-MatmulAddRelu/`。

## 题目背景

使用 Catlass Python 实现 `Y = ReLU(A @ B + X)`：先由 Cube 计算矩阵乘，再由 Vector 把乘积与 `X` 逐元素相加并执行 ReLU，最后将结果写入 `Y`。本题固定处理一组 `256×256×256` 矩阵。

## 算子描述

```text
Y = ReLU(A @ B + X)
```

| 项目 | 要求 |
| :--- | :--- |
| 输入 | `A(256,256)`、`B(256,256)` 为 FP16；`X(256,256)` 为 FP32 |
| 输出 | `Y(256,256)` 为 FP32，不修改输入 |
| 布局 | 四个矩阵均为连续行优先 |
| 核函数 | `matmul_add_relu_kernel(gm_a, gm_b, gm_x, gm_y)` |
| 启动参数 | `block_num=1` |

因为256x256x256的矩阵乘规模较小，使用`block_num=1` 启动一个计算核心(AICORE)处理。

## 答题准备

进入https://hidevlab.huawei.com/home，按下面步骤获取环境：

![image.png](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/20260928173518453.png)

![image.png](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/20260928173552958.png)

![image.png](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/20260928173618843.png)

![image.png](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/20260928173641478.png)

![image.png](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/20260928173916858.png)

进入环境后拉取一下题库仓库：

```bash
git clone https://gitcode.com/cann/cann-outreach.git
```

开始答题前，先从 `cann-outreach` 仓库根目录执行以下命令，进入题目目录，并将模板复制到以本人 GitCode 用户名命名的个人目录。请将 `your_gitcode_name` 换成自己的用户名：

```bash
cd 2026/CANN-Code-Detective/Challenge09-MatmulAddRelu
cp -r Template PR_code/your_gitcode_name
```

后续请在 `PR_code/your_gitcode_name/matmul_add_relu.py` 中补全代码。

## 待补全任务

L1 装入完整的 A/B；L0 按 K=64 分成四次搬运和累加。L0C 的 `256×256` FP32 结果通过 `SPLIT_M` 分给两个 AIV，各处理 128 行。每个 AIV 将 `X` 分为两个 `64×256` 分片，在存放累加结果的 UB 中原位执行 Add→ReLU，再写回 Y。

在个人目录的 `matmul_add_relu.py` 中补全五处代码（[原始模板参考](Template/matmul_add_relu.py)）：

1. `TODO 1`：GM→L1 搬运 A/B。
2. `TODO 2`：L1→L0A/L0B 搬运当前 K 分片。
3. `TODO 3`：将 K=256 分为四个长度为 64 的分片，每个分片调用一次 `tla.mmad`。首次将当前分片的乘积写入 L0C，覆盖 L0C 原有数据；后三次将其余分片的乘积累加到 L0C。
4. `TODO 4`：使用 `SPLIT_M` 将 L0C 搬入两个 AIV 的 UB。
5. `TODO 5`：Vector 先加 `X`、再做 ReLU，原位写回 UB。

接口参考：

- 数据搬运与分片：[tla.copy](https://catlass.readthedocs.io/zh-cn/latest/4_CATLASS_DSL/api/kernel_api_reference/#copy)、[tla.tile_view](https://catlass.readthedocs.io/zh-cn/latest/4_CATLASS_DSL/api/kernel_api_reference/#tile_view)。
- 矩阵乘累加：[tla.mmad](https://catlass.readthedocs.io/zh-cn/latest/4_CATLASS_DSL/api/kernel_api_reference/#mmad)。
- L0C→UB 分发：[CopyL0C2DstParams 与 L0C2UBMode.SPLIT_M](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/catlass/params.py)。
- Vector 后处理：[Tensor.load](https://catlass.readthedocs.io/zh-cn/latest/4_CATLASS_DSL/api/kernel_api_reference/#tensorload)、[tla.add](https://catlass.readthedocs.io/zh-cn/latest/4_CATLASS_DSL/api/kernel_api_reference/#add)、[tla.max](https://catlass.readthedocs.io/zh-cn/latest/4_CATLASS_DSL/api/kernel_api_reference/#max)、[Tensor.store](https://catlass.readthedocs.io/zh-cn/latest/4_CATLASS_DSL/api/kernel_api_reference/#tensorstore)。
- 核内同步：[tla.mutex](https://catlass.readthedocs.io/zh-cn/latest/4_CATLASS_DSL/api/kernel_api_reference/#mutex)、[mutex.lock](https://catlass.readthedocs.io/zh-cn/latest/4_CATLASS_DSL/api/kernel_api_reference/#mutex_lock)、[mutex.unlock](https://catlass.readthedocs.io/zh-cn/latest/4_CATLASS_DSL/api/kernel_api_reference/#mutex_unlock)。

每处 `TODO` 的上下边界均为独立一行 `# --------------------------------`。请按其中文注释，在两条分隔线之间用代码替换 `pass`；分块、同步和最终 GM 写回框架已给出。ReLU 不能在加 `X` 之前执行。

核内缓冲的访问顺序统一由 `tla.mutex` 的 `lock` / `unlock` 管理。v2.1.0 中 AIC 与 AIV 使用独立的 mutex ID 空间，因此 Cube→Vector 仍通过 `cross_flag` 通知数据就绪，参见[核内与跨核同步的范围说明](https://catlass.readthedocs.io/zh-cn/latest/4_CATLASS_DSL/dsl_development/feature_development/auto_sync_design/)。验证脚本中的 `torch.npu.synchronize()` 用于让主机等待整个核函数执行完成。

每个 AIV 的 UB 使用 128 KiB 存放累加结果、64 KiB 存放 X 分片，合计 192 KiB。

## 环境准备

| 类别 | 需要的组件 |
| :--- | :--- |
| 硬件与系统 | Ascend 950 系列 NPU、配套驱动和固件，以及 CANN 支持的 Linux（`x86_64` 或 `aarch64`） |
| CANN | CANN 9.1.0 或更高版本 |
| Python | Python 3.10～3.13、`pip` |
| Catlass Python 包 | `ascend-catlass-dsl==2.1.0`；其依赖的 `numpy>=1.24,<3` 会由 `pip` 自动安装 |
| 验证 Python 包 | `torch`、`torch-npu`（导入名为 `torch_npu`）和 `PyYAML`（导入名为 `yaml`）；前两者须相互匹配并与 CANN 版本匹配 |

[PyPI 已发布 Catlass Python v2.1.0 wheel](https://pypi.org/project/ascend-catlass-dsl/2.1.0/)。以下以 CANN 9.1.0 为例，安装配套的 PyTorch 2.9.0 和 `torch-npu 2.9.0.post6`；其他 CANN 版本请按[昇腾 PyTorch 版本配套表](https://gitcode.com/Ascend/pytorch/blob/master/docs/zh/release_notes.md)调整。先加载 CANN 环境；如果 CANN 不在示例路径，请改成实际的 `set_env.sh` 路径：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

可以用`uname -a`看下机器CPU架构，下面两个步骤选符合的一个执行(hidevlab环境应该为x86)：

**x86_64：**安装 `torch==2.9.0+cpu`。这里通过[阿里云 PyTorch CPU wheel 镜像](https://mirrors.aliyun.com/pytorch-wheels/cpu/)获取该 wheel；也可使用[PyTorch 官方 CPU wheel 源](https://download.pytorch.org/whl/cpu/torch/)（需能访问该站点）。

```bash
python -m pip install --index-url https://pypi.org/simple --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu/ "torch==2.9.0+cpu"
```

**aarch64：**按[昇腾 PyTorch 安装说明](https://gitcode.com/Ascend/pytorch/tree/v2.9.0-7.3.0)从 PyPI 安装不带 `+cpu` 后缀的 `torch==2.9.0`。

```bash
python -m pip install --index-url https://pypi.org/simple "torch==2.9.0"
```

两种架构完成各自的 `torch` 安装后，都执行以下命令安装其余 Python 包、检查导入并准备 BC 模板：

```bash
python -m pip install \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple \
  "torch-npu==2.9.0.post6" "ascend-catlass-dsl==2.1.0" "PyYAML>=6,<7"
python -c "import torch, torch_npu, catlass; print(torch.__version__, torch_npu.__version__)"
python -m catlass.bc_compile
```

本题验证脚本用 `torch` 生成输入和计算参考结果，用 `torch-npu` 提供 NPU 后端。x86_64 版 `torch-npu 2.9.0.post6` 要求 `torch==2.9.0+cpu`；`+cpu` 表示 PyTorch 包不附带 CUDA 组件，NPU 后端仍由 `torch-npu` 提供。`PyYAML` 提供 `torch_npu` 导入时需要的 `yaml` 模块。`catlass.bc_compile` 会生成 BC（LLVM bitcode）文件，其中包含数据搬运、矩阵乘等底层实现，学员无需修改。如需自行构建 wheel，参见[Catlass Python 官方环境准备与构建](https://catlass.readthedocs.io/zh-cn/latest/4_CATLASS_DSL/dsl_development/build_guide/)。



## 运行验证

补全 `PR_code/your_gitcode_name/matmul_add_relu.py` 后，在 Ascend 950 环境的题目目录下运行：

```bash
python PR_code/your_gitcode_name/test_matmul_add_relu.py --device 0
```

验证脚本只检查固定 `256×256×256` 尺寸，以 PyTorch 的 `torch.relu(A.float() @ B.float() + X)` 为参考，检查精度、有限值和非负性。通过时输出 `PASS`；失败时输出 `FAIL` 并返回非零状态。

## 提交说明

将本人 `PR_code/<GitCode用户名>/` 下补全的代码和验证脚本提交为 PR。保留 `PR_code/README.md` 与 `Template/`。提交和 CLA 操作可参考[代码侦探活动说明](../README.md)与 [GitCode 讨论 #287](https://gitcode.com/org/cann/discussions/287)。[CANNJudge 平台](https://cannjudge.cn/home)为活动入口；本题 README 不列出未经确认的第 09 题专属验证链接。

## 推荐学习资料

本题是固定尺寸的简单 Demo，展示 Cube 矩阵乘与 Vector 后处理的基本流程，未考虑性能优化。进一步优化时，可参考 Catlass Python v2.1.0 中的以下样例：

- [基础 MMAD 矩阵乘](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/examples/end_to_end/basic_mmad/basic_matmul.py)：学习多核分块调度、L1/L0 双缓冲，以及数据搬运与计算的流水重叠。
- Matmul 后处理样例：[Matmul+Add](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/examples/end_to_end/basic_mmad_epilogue/matmul_add_ub.py)、[Matmul+LeakyReLU](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/examples/end_to_end/basic_mmad_epilogue/matmul_leaky_relu.py) 等。前者展示 L0C→UB 后叠加 X，后者展示激活函数后处理。
- [Stream-K 矩阵乘](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/examples/end_to_end/basic_mmad_streamk/README.md)：按 K 维分配工作，并对尾轮任务做负载均衡与结果归约。
