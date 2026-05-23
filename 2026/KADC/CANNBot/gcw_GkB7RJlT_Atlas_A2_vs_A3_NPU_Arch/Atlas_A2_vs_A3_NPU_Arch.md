# Atlas A2 vs A3 NPU 架构区别

## 架构映射

| 代号 | SocVersion | NpuArch | 芯片 |
|------|-----------|---------|------|
| **A2** | ASCEND910B | DAV_2201 | Ascend910B1~B4, B2C |
| **A3** | ASCEND910B (含 Ascend910_93) | DAV_2201 | Ascend910_93 |

**关键点：A2 和 A3 共用同一 NpuArch（DAV_2201），`__NPU_ARCH__=2201`**，指令集和微架构相同。

## 实际差异

A3（Ascend910_93）相对 A2 典型 SKU（如 910B2）的主要区别在 **SKU 级规格**而非架构级：

| 规格 | A2 (910B2) | A3 (910_93) |
|------|-----------|-------------|
| Cube 核数 | 24 | 可能不同（因 SKU 裁剪） |
| 频率 | 1.8 GHz | 可能不同 |
| L2 | 192 MB | 可能不同 |
| Memory | 64 GB | 可能不同 |

> 注意：`Ascend910_93` 运行时 SocVersion 映射到 `ASCEND910B`（非独立枚举值），NpuArch 同为 `DAV_2201`。对算子开发来说，A2 和 A3 **无需条件编译分支**，运行时通过 `GetCoreNumAic()` / `GetCoreMemSize()` 获取实际参数即可。

## 大架构跃迁：A5（DAV_3510）

真正的大架构跃迁发生在 **A5（DAV_3510）**，新增 FP8/MXFP4 数据格式、L0C 翻倍至 256KB、UB 增至 248KB、SIMT/Regbase 编程模型等。

### Buffer 容量对比

| Buffer | A2 (910B2, DAV_2201) | A5 (950PR, DAV_3510) | 用途 |
|--------|:---:|:---:|------|
| L1 | 512 KB | 512 KB | Cube 输入缓存 |
| L0A | 64 KB | 64 KB | Cube 左矩阵操作数 |
| L0B | 64 KB | 64 KB | Cube 右矩阵操作数 |
| L0C | 128 KB | **256 KB** | Cube 输出 |
| UB | 192 KB | **248 KB** | Vector 工作区 |
| L2 | 192 MB | 128 MB (Server) / 112 MB (PCIE) | 跨核共享缓存 |
| BT | 1 KB | **4 KB** | FixPipe Bias 表 |
| SSBuffer | — | **256 KB** | DAV_3510 新增 AIC↔AIV 核间消息通路 |

### 算力对比

| 规格项 | Ascend910B2 (DAV_2201) | Ascend950PR PCIE (DAV_3510) | Ascend950PR Server (DAV_3510) |
|--------|:---:|:---:|:---:|
| CubeCore 核数 | 24 | 28 | 32 |
| 频率 (GHz) | 1.8 | 1.65 | 1.65 |
| Cube 算力 BF16/FP16 | 353T | 378T | 432T |
| Cube 算力 FP8/HiF8/MXFP8 | — | 757T | 865T |
| Cube 算力 MXFP4 | — | 1514T | 1730T |
| Vector 算力 FP16 | 22T | 47T | 54T |
| Memory 容量 (GB) | 64 | 112 | 128 |