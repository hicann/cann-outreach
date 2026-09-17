# InplaceRsqrt (Ascend C)

原位计算 `self = 1 / sqrt(self)`。

| 项目 | 说明 |
| --- | --- |
| Shape | 4D `[2,2,4,128]`（`[N4,N3,N2,N1]`） |
| Dtype | float16 (`half`) |
| Format | ND |
| 核心接口 | `AscendC::Rsqrt`（向量化 vrsqrt） |

## 核心 API

```cpp
#include "inplace_rsqrt.h"

// LocalTensor<half> self; 直接修改 self 内存
InplaceRsqrt(self, count);
```

完整核函数见 `inplace_rsqrt.cpp`。

## 编译命令

```bash
cd ops/inplace_rsqrt
export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest

# CPU 调试
bash run.sh -r cpu -v Ascend310P3

# NPU 上板（SOC 请按实际修改）
bash run.sh -r npu -v Ascend310P3

# 或手动 cmake
cmake -B build \
  -DRUN_MODE=npu \
  -DSOC_VERSION=Ascend310P3 \
  -DASCEND_CANN_PACKAGE_PATH=${ASCEND_INSTALL_PATH} \
  -DCMAKE_INSTALL_PREFIX=./out
cmake --build build -j && cmake --install build
```
