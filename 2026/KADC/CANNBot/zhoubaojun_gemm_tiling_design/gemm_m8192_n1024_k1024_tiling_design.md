# GEMM Tiling Design: M=8192, N=1024, K=1024, FP16, DAV_2201

## 1. 硬件约束速查（DAV_2201）

| Buffer | 容量 | 关键约束 |
|--------|------|---------|
| L0A | 64 KB | baseM × baseK × 2 ≤ 65536 |
| L0B | 64 KB | baseN × baseK × 2 ≤ 65536 |
| L0C | 128 KB | db × baseM × baseN × 4 ≤ 131072 |
| L1 | 512 KB | depthA/B × tile × 2 ≤ 524288 |
| UB | 192 KB | 输出暂存 |
| CubeCore | 24 | 1:2 配对 VectorCore |
| MAC atom | 16³ | baseM/baseN/baseK ≥ 16 且为 16 倍数 |

## 2. Tile 尺寸选择

**推荐配置**：`baseM=128, baseN=256, baseK=128, dbL0C=1`

| Buffer | 计算 | 占用 | 上限 | 结果 |
|--------|------|------|------|------|
| L0A | 128×128×2 | 32 KB | 64 KB | ✓ |
| L0B | 256×128×2 | 64 KB | 64 KB | ✓ (刚好) |
| L0C(db=1) | 128×256×4 | 128 KB | 128 KB | ✓ (刚好) |

> dbL0C=2 不满足（需 256KB > 128KB），故 L0C 不做 double buffer。

**L1 滚动深度**：depthA1=2, depthB1=2

| Buffer | 计算 | 占用 | 上限 | 结果 |
|--------|------|------|------|------|
| L1_A | 2×128×128×2 | 64 KB | — | ✓ |
| L1_B | 2×256×128×2 | 128 KB | — | ✓ |
| L1 合计 | — | 192 KB | 512 KB | ✓ (余量充足) |

**对齐检查**：
- M=8192 / 128 = 64 ✓ (整除)
- N=1024 / 256 = 4 ✓ (整除)
- K=1024 / 128 = 8 ✓ (整除，无尾块)

## 3. 多核切分

```
切分维度: M × N 二维切分（K 轴核内迭代，不参与核间切分）
M 方向 tile 数: 64 (8192/128)
N 方向 tile 数: 4  (1024/256)
总 tile 数:     256
可用 CubeCore:  24
```

**调度策略**：蛇形调度（BlockScheduler）

| 项目 | 值 |
|------|------|
| usedCoreNum | 24 |
| 每 core tile 数 | 10~11（256/24，蛇形分配保证负载均衡） |
| 单 tile K 迭代 | 8 步 (1024/128) |
| 单 core 总 MMAD | ~80~88 次 |

**核心数公式**（动态计算，禁止硬编码）：
```
usedCoreNum = min(CeilDiv(M, baseM) × CeilDiv(N, baseN), maxCubeCoreNum)
            = min(64 × 4, 24) = 24
```

## 4. Buffer 规划

### AIC 侧（CubeCore）

| Buffer | 位置 | 大小 | 用途 | Double Buffer |
|--------|------|------|------|:---:|
| L1_A | L1 | 64 KB | A 矩阵滚动窗口 | depth=2 |
| L1_B | L1 | 128 KB | B 矩阵滚动窗口 | depth=2 |
| L0A | L0A | 32 KB | A 分块计算 | 单 buf |
| L0B | L0B | 64 KB | B 分块计算 | 单 buf |
| L0C | L0C | 128 KB | 累加结果 (FP32) | db=1 |

### AIV 侧（VectorCore）— 纯 GEMM 无 eltwise

| Buffer | 大小 | 用途 |
|--------|------|------|
| UB_COut (FP32) | 128 × 256 × 4 = 128 KB | Fixpipe L0C→UB 输出 |
| UB_COut_FP16 | 128 × 256 × 2 = 64 KB | Cast FP32→FP16 暂存 |
| **UB 合计** | **192 KB** | = UB_SIZE ✓ (刚好) |

> 纯 GEMM UB 紧凑：128KB(FP32) + 64KB(FP16) = 192KB。若输出 FP32 则仅需 128KB，余量 64KB。

## 5. 数据流

```
A[M,K] FP16 ── GM → L1(Nz) → L0A → ┐
                                     │ MMAD (16³ atom)
B[K,N] FP16 ── GM → L1(Zn) → L0B → ┤
                                     │ K轴8步累加 → L0C(FP32)
                                     └
                           L0C → UB (Fixpipe, FP32)
                           UB  → Cast FP32→FP16
                           UB  → GM (DataCopy, FP16 output)
```

## 6. 核内计算流程

```
for each (m_tile, n_tile) assigned by BlockScheduler:
    for k_iter = 0 to 7:                      # K=1024, baseK=128, 8步
        GM→L1: A[m_tile × baseM : , k_iter × baseK : ]   # MTE2, depthA1=2 滚动
        GM→L1: B[k_iter × baseK : , n_tile × baseN : ]   # MTE2, depthB1=2 滚动
        L1→L0A: A 分块                                       # MTE1
        L1→L0B: B 分块                                       # MTE1
        MMAD: L0A × L0B → L0C 累加                          # Cube MAC
    
    L0C→UB: Fixpipe (FP32)                                   # CV 直通
    Cast: UB FP32 → UB FP16                                  # Vector
    UB→GM: DataCopy output FP16                              # MTE3
```

## 7. 分支覆盖

| 场景 | 本设计情况 | 处理 |
|------|-----------|------|
| M/N/K 尾块 | 全整除，无尾块 | 无需特殊处理 |
| ODD-M | M=8192 偶数 | 不触发 |
| ODD-N | N=1024, 256整除 | 不触发 |
| 小 shape | — | 减少核数，缩小 baseM/baseN |
| 转置 | 未指定 | Host 侧选 RowMajor/ColumnMajor |
| K 对齐 | K=1024, baseK=128 | FP16 无 64 对齐要求（mxfp8 才需） |

---

**UB 紧凑提示**：FP16 输出时 UB 用量 192KB 刚好等于上限。若后续需融合 eltwise，应缩小 baseN（如 baseN=128）腾出 UB 空间给 stage buffer，或使用 SPLIT_M 将 L0C 拆分给双 AIV。