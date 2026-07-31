# aclnnTruncateMod

## 函数原型
```cpp
aclnnStatus aclnnTruncateMod(const aclTensor* x1, const aclTensor* x2,
                              const aclTensor* y, aclOpExecutor* executor);
```

## 参数
| 参数 | 方向 | 说明 |
|------|------|------|
| x1 | 输入 | 被除数 tensor, float16/float32/bfloat16/int32/int8/uint8 |
| x2 | 输入 | 除数 tensor, 同x1 |
| y | 输出 | 余数, shape=broadcast(x1,x2) |
| executor | 输入 | 算子执行器 |

## 约束
维度≤8, x1与x2需广播兼容
