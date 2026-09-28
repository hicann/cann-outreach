# add_custom

Ascend C implementation of elementwise `float16` addition for two equal-shape
2-D ND tensors:

```text
z = x + y
```

The kernel flattens `[N2, N1]` into one contiguous range and uses the Ascend C
vector `Add` instruction. `DataCopyPad` handles a short final range, so the
shape does not need to be padded.

## Layout

```text
op_host/add_custom.cpp          operator schema and tiling
op_kernel/add_custom.cpp        Ascend C device kernel
op_kernel/add_custom_tiling.h   shared tiling data
tests/test_source_contract.py   host-only source check
```

## 运行指南

### 1. 环境要求

需要在 Linux 主机上运行，并安装以下组件：

- CANN Toolkit，包含 Ascend C 编译器和 `ascendc_kernel_cmake`。
- 与目标设备匹配的驱动和固件。
- CMake 3.16 或更高版本。
- Python 3，用于运行源码检查脚本。

先加载 CANN 环境。安装路径按实际情况修改：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

进入算子目录：

```bash
cd add_custom
```

### 2. 运行源码检查

该检查不需要 CANN 或 NPU，只检查算子名称、数据类型、数据格式、二维 shape 约束和 Kernel 关键 API：

```bash
python3 -B tests/test_source_contract.py
```

预期输出：

```text
add_custom source contract: OK
```

### 3. 编译 Ascend C 算子

`SOC_VERSION` 必须改成实际设备型号，例如 `Ascend910B1`、`Ascend310P3`：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1 -DASCEND_CANN_PACKAGE_PATH="$ASCEND_CANN_PACKAGE_PATH"
cmake --build build -j$(nproc)
```

构建目标如下：

- `add_custom_kernel`：编译 Ascend C device Kernel。
- `add_custom_host`：编译 Host 侧算子定义、shape/type 推导和 tiling。

### 4. 在 NPU/框架中执行

本目录提供算子源码和编译入口，没有独立的 `main` 程序。编译成功后，需要将 Host 注册目标和 Kernel 产物接入 ACL、CANN 图执行框架或已有自定义算子工程，再传入两个相同 shape 的二维 `float16` ND 张量：

```text
x: [N2, N1], float16, ND
y: [N2, N1], float16, ND
z = add_custom(x, y)
```

Kernel 使用 `DataCopyPad` 处理非 32 字节对齐的尾部数据，不要求 `N2 * N1` 必须是 16 的倍数。

当前工作区没有 CANN 头文件、编译器或 Ascend 设备，因此本机只能运行第 2 步的源码检查，不能完成真实 NPU 编译和数值验证。
