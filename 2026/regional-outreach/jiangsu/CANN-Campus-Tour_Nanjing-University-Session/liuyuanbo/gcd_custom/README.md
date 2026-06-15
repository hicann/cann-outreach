# Gcd Custom Kernel 直调工程

Gcd（最大公约数）算子的 Kernel 直调实现。

## 算子说明

- **算子类型**: Elementwise 逐元素二元算子
- **数学定义**: `out = gcd(self, other)`
- **数据类型**: FP16 (half) → 内部用 float 计算
- **实现算法**: Euclidean 辗转相除法（16 次固定迭代）
- **trunc 实现**: float→int32(TRUNC)→float 截断取整
- **Format**: ND
- **Shape**: 4 维，支持 broadcast

### Broadcast 测试用例

| 索引 | Self shape | Other shape | Out shape | 元素数 |
|------|-----------|-------------|-----------|--------|
| 0 | [1, 4, 1, 128] | [1, 4, 16, 1] | [1, 4, 16, 128] | 8192 |
| 1 | [1, 4, 1, 32] | [1, 1, 16, 1] | [1, 4, 16, 32] | 2048 |
| 2 | [1, 1, 1, 8] | [32, 1, 1, 1] | [32, 1, 1, 8] | 256 |

Broadcast 扩展在 Python 脚本侧完成，kernel 为 1:1 的 elementwise GCD 计算。

## 实现细节

### 数据流

```
self (half) → Cast(TRUNC) → int16 → Abs → Euclidean(×16) → Cast(NONE) → half → out
other(half) → Cast(TRUNC) → int16 → Abs ↗
```

### Euclidean 算法（int16）

```
for iter in 0..15:
    cmp = (b > 0)
    r = a - (a / b) * b     # a % b  (int16 整数除法)
    a = cmp ? b : a          # Select
    b = cmp ? r : b          # Select
```

### Buffer 规划

| Buffer | 类型 | 大小 | 用途 |
|--------|------|------|------|
| inQueueSelf | TQue | DB × 4KB | self 输入 |
| inQueueOther | TQue | DB × 4KB | other 输入 |
| outQueueOut | TQue | DB × 4KB | 输出 |
| tmpBufA | TBuf | 8KB | int16 a 工作区 |
| tmpBufB | TBuf | 8KB | int16 b 工作区 |
| tmpBufQ | TBuf | 8KB | int16 商 |
| tmpBufR | TBuf | 8KB | int16 余数 |
| cmpBuf | TBuf | 4KB | uint8 比较结果 |
| zeroBuf | TBuf | 8KB | int16 零值常量 |

## 快速开始

```bash
cd ops-lab/gcd_custom
source ${ASCEND_HOME_PATH}/set_env.sh

# 一键运行
bash run.sh       # 默认 case 0
bash run.sh 1     # case 1
bash run.sh 2     # case 2

# 跳过编译
bash run.sh --skip-build
```
