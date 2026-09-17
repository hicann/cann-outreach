# Relu (Ascend C)

`y = max(x, 0)`，1 输入 1 输出。

| 项目 | 说明 |
| --- | --- |
| Shape | `[2,2,4,128]`（4D ND） |
| Dtype | float16 |
| 接口 | `AscendC::Relu` |

```bash
bash run.sh -r npu -v Ascend310P3
```
