# Gcd Custom Kernel 直调 — Ascend C

Ascend C 实现 GCD（最大公约数）算子，**支持 4D broadcast**，数据类型 float16，Format=ND。

## 算子说明

| 项目 | 内容 |
|------|------|
| **功能** | `out = gcd(self, other)` 逐元素最大公约数 |
| **数据类型** | float16 (half) |
| **输入** | self (half), other (half) — 支持 broadcast |
| **输出** | out (half) — broadcast 后的 shape |
| **Shape** | 4 维 `[N4,N3,N2,N1]`，self 和 other 满足 broadcast 关系 |
| **Format** | ND |

## 文件结构

```
├── op_kernel/
│   ├── gcd_custom_tiling.h       Tiling 常量 + host 侧 broadcast 工具函数
│   └── gcd_custom_kernel.asc     Kernel（标准 CopyIn→Compute→CopyOut queue 模式）
├── op_host/
│   ├── gcd_custom.asc            Host（解析 shape → 读原始数据 → broadcast 扩展 → kernel）
│   └── data_utils.h              数据读写工具
├── op_extension/
│   ├── gcd_custom_torch.cpp      PyTorch（expand + contiguous 后调用 kernel）
│   ├── register.cpp              TORCH_LIBRARY 注册
│   └── ops.h                     函数声明
├── scripts/
│   ├── gen_data.py               3 组 broadcast 测试场景生成
│   ├── golden.py                 使用 np.gcd 计算参考值
│   ├── verify_result.py          fp16 精度验证
│   └── test_torch.py             7 个 PyTorch 测试用例
├── CMakeLists.txt                双 target（可执行文件 + .so）
├── run.sh                        一键运行
└── README.md
```

## 架构：Host 预扩展策略

```
原始数据(不等shape)          扩展后数据(等shape)
self=[4,2,1,8] ──Expand──→  self_exp=[4,2,6,8]  ──┐
other=[4,1,6,8]──Expand──→  other_exp=[4,2,6,8] ──┤──→ kernel(gcd) → out=[4,2,6,8]
                                                  └── CopyIn→Compute→CopyOut
```

| 层 | 职责 |
|------|----------|
| **Host** | 解析 8 个 shape 参数 → 计算 broadcast shape → 读原始数据 → **ExpandBroadcast()** 扩展 → 分配 device 内存 → 启动 kernel |
| **Kernel** | 纯 1:1 elementwise GCD，标准 `TQue` queue 模式；**不感知 broadcast** |
| **PyTorch** | `self.expand(outSize).contiguous()` 利用 PyTorch 做 broadcast 扩展 |

## Broadcast 测试场景

| 场景 | self shape | other shape | out shape | broadcast 类型 |
|------|-----------|------------|-----------|---------------|
| 1 | [4,2,1,8] | [4,1,6,8] | [4,2,6,8] | dim=2 单方 |
| 2 | [1,4,4,4] | [2,4,4,4] | [2,4,4,4] | dim=0 单方 |
| 3 | [3,1,5,1] | [1,4,1,6] | [3,4,5,6] | 双向 broadcast |

## 一键运行

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
cd gcd_custom

bash run.sh 1   # [4,2,1,8] + [4,1,6,8] → [4,2,6,8]
bash run.sh 2   # [1,4,4,4] + [2,4,4,4] → [2,4,4,4]
bash run.sh 3   # [3,1,5,1] + [1,4,1,6] → [3,4,5,6]
bash run.sh --torch  # 仅 PyTorch 通路
```

## GCD 算法

逐元素执行：
1. `half → float → int32`（取整，取绝对值）
2. **Euclid 欧几里得算法**：`while(b != 0) { temp = b; b = a % b; a = temp; }`
3. `int32 → float → half` 写回

## Kernel 设计要点

| 设计 | 说明 |
|------|------|
| **数据搬运** | `DataCopyPad` + `TQue<IN/OUT>` queue 双缓冲模式 |
| **逐元素 GCD** | `LocalTensor::GetValue/SetValue` 在 UB 内标量访问 |
| **Broadcast** | **Host 侧** `ExpandBroadcast()` 预扩展；kernel 无需感知 |
| **Block 分配** | 按 `totalLength / blockNum` 平均切分 |
| **Tiling** | TILE_LENGTH=4096，fp16 → 8KB/tile，UB 总占用 24KB |

> ⚠️ 前提：需 CANN 工具链 + `torch_npu` + NPU 环境。NPU 架构默认为 `dav-2201`（Atlas A2）。
