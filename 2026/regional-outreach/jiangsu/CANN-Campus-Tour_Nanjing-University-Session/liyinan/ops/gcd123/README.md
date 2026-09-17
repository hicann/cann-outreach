# gcd123 (Ascend C)

`out = gcd(self, other)`，2 输入 1 输出，支持 broadcast。

| 项目 | 说明 |
| --- | --- |
| 输入 | self, other（float16, ND, 4D） |
| 输出 | out，shape = broadcast(self, other) |
| 样例 | self `[1,4,1,128]` × other `[1,4,16,1]` → out `[1,4,16,128]` |
| 算法 | half→float→abs→Euclidean（16 次向量化迭代）→half |

Broadcast 在 `scripts/gen_data.py` 侧物化，kernel 做等长逐元素 GCD。

```bash
bash run.sh -r npu -v Ascend310P3
```
