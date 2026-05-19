## 用户需求
本文档记录了用户开发算子的需求，请根据描述进行需求检测及分析。

---

## 需求-1
请开发一个 PyPTO 动态算子：
- 算子名称：softmax
- 公式：$ y = \frac{e^{x - \max(x, \text{dim}=-1)}}{\sum e^{x - \max(x, \text{dim}=-1)}} $
- 规格：

| 类型  | shape  | dtype  |
| ------------ | ------------ | ------------ |
| 输入 x| \[batch, seqlen, head, dim\] | float32  |
| 输出 y| \[batch, seqlen, head, dim\] | float32  |

- 精度标准：atol=0.003, rtol=0.003
- 动态轴：所有维度均为动态
