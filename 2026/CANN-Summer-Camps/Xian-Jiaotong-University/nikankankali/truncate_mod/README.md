# TruncateMod

TruncateMod是二元取模算子: `y = x1 - trunc(x1/x2) * x2`, 基于Ascend C, 适配ascend910b。

## 支持
| 属性 | 值 |
|------|-----|
| dtype | float16/float32/bfloat16/int32/int8/uint8 |
| shape | 任意（≤8维），支持广播 |

## 目录
```
├── op_host/    # 算子定义+infershape+tiling
├── op_kernel/  # kernel入口+类实现
├── op_api/     # aclnn接口
├── examples/   # 调用示例
├── tests/ut/   # 单元测试
├── docs/       # 设计文档+自验报告+API文档
└── README.md
```

## 构建
```bash
bash build.sh -j8
```

## 测试
```bash
cd tests/ut && bash run.sh
```

## 验证
CANNJudge提交ID 112547, 50/50 Pass
