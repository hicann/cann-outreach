# TruncateMod 算子设计方案

> CANN训练营2026暑期季·西安交通大学专场 | 第21组 | nikankankali

## 1. 需求背景
参考内置TruncateMod的TBE实现，基于Ascend C在NPU上实现。
- TBE参考: `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/`

### 功能定义
`y = x1 - trunc(x1/x2) * x2`, 支持numpy风格广播。

### 参数
| 参数 | 含义 | dtype |
|------|------|-------|
| x1 | 被除数 | float16/float32/bfloat16/int32/int8/uint8 |
| x2 | 除数 | 同上 |
| y | 余数 | 同上 |

## 2. 需求分析
1. 6种dtype支持
2. 广播+动态rank/shape
3. CANNJudge 50/50, 性能≥baseline×95%

## 3. 详细设计
### host侧
- op_def: AutoContiguous+动态rank/shape
- infershape: broadcast shape计算
- tiling: CeilDiv分核+UB切分+双缓冲判断

### kernel侧
- Init: broadcast类型检测(无/标量/模重复/通用)
- CopyIn: 4条搬入路径
- Compute: 5条dtype路径(int32/int8-uint8/bf16/float32/float16)
- Process: CopyIn→Compute→CopyOut流水线

## 4. 硬件

ascend910b ✅

## 5. 约束
维度≤8, x1与x2广播兼容

## 6. 验证
CANNJudge提交ID 112547, 50/50 Pass, 精度100%, 性能达标
