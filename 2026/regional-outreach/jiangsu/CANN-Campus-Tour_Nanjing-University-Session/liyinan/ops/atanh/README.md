# atanh (Ascend C)

`y = atanh(x)`，1 输入 1 输出。

| 项目 | 说明 |
| --- | --- |
| Shape | `[2,2,4,128]`（4D ND） |
| Dtype | float16 |
| 接口 | `AscendC::Atanh`（带 sharedTmpBuffer） |

输入值域建议落在 `(0.001, 0.99)`（样例数据生成已遵守）。

```bash
bash run.sh -r npu -v Ascend310P3
```
