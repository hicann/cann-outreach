# AtanhCustom

四维 `float16` ND 张量的逐元素反双曲正切。数学定义为
`y = 0.5 * ln((1 + x) / (1 - x))`，实数域要求 `|x| < 1`。

构建（Ascend 910B）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
msopgen gen -i AtanhCustom.json -c ai_core-ascend910b -lan cpp -out AtanhCustom
# 用本包 op_host/、op_kernel/ 替换生成目录中的同名源码目录后：
cd AtanhCustom && bash build.sh
```

逻辑算子名为 `AtanhCustom`，kernel 名为 `atanh_custom`。采用双缓冲向量流水，
最后一个 tile 支持非 32 字节对齐长度。

生成四维测试数据：

```bash
python3 scripts/gen_data.py
```

在设备上执行算子并将输出写为 `output.bin` 后验证：

```bash
python3 scripts/verify_result.py output.bin
```

测试输入限定在实数定义域 `[-0.9, 0.9]`。验证容差考虑了 `float16` 下基础
向量指令组合产生的中间舍入误差。
