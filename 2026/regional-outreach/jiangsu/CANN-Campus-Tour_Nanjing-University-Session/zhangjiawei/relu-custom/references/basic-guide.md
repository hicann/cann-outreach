# ReLU 算子开发指南

## 与 Abs 算子的差异

| 项目 | Abs | ReLU |
|------|-----|------|
| 计算 API | `Abs(y, x, count)` | `Duplicate(zeros, 0) → Max(y, x, zeros, count)` |
| 额外 Buffer | 无 | 1 个 zeros TBuf |
| Tiling 除数（单缓冲） | 2×2 = 4 | 3×2 = **6** |
| Tiling 除数（双缓冲） | 4×2 = 8 | 5×2 = **10** |
| 精度策略 | FP16 直算 | FP16 直算 |

## Kernel 计算流水线

```
CopyIn (FP16 → UB)
  → Duplicate(zeros, 0)         填充零常量
  → Max(y, x, zeros, count)     ReLU 核心计算
  → CopyOut (UB → GM)
```

## 检查清单

- [ ] 算子定义：DT_FLOAT16
- [ ] Tiling：bufferDivisor 包含 zeros TBuf（单缓冲 3×2，双缓冲 5×2）
- [ ] Kernel：DoCompute 内 Duplicate + Max
- [ ] 测试覆盖：负值、0、正值、负零、正零
