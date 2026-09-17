# [苏州城市学院] liyinan Ascend C 算子提交

提交者目录：`2026/regional-outreach/jiangsu/CANN-Campus-Tour_Nanjing-University-Session/liyinan`

## 算子一览

| 算子 | 路径 | 输入/输出 | Shape | Dtype | Format |
| --- | --- | --- | --- | --- | --- |
| add_custom | `./` | x, y → z | `[8, 2048]` | float16 | ND |
| atanh | `ops/atanh` | x → y | `[2, 2, 4, 128]` | float16 | ND |
| gcd123 | `ops/gcd123` | self, other → out | 4D broadcast | float16 | ND |
| Relu | `ops/relu` | x → y | `[2, 2, 4, 128]` | float16 | ND |
| InplaceRsqrt | `ops/inplace_rsqrt` | self（原位） | `[2, 2, 4, 128]` | float16 | ND |

## 说明

- 单位：苏州城市学院
- 框架：Ascend C（Kernel 直调工程）
- `gcd123`：self `[1,4,1,128]` × other `[1,4,16,1]` → out `[1,4,16,128]`
- `InplaceRsqrt`：`AscendC::Rsqrt` 向量化原位

## 编译运行

```bash
export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
bash run.sh -r npu -v Ascend310P3
```
