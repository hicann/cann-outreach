## 概述
本样例介绍MatmulLeakyRelu算子的核函数直调方法。

## 目录结构介绍
```
├── KernelLaunch                        // 使用核函数直调的方式调用MatmulLeakyRelu自定义算子。
│   ├── CppExtensions                   // kernel侧的核函数调用程序，通过使用ACLRT_LAUNCH_KERNEL调用宏来完成验证
│   ├── MatmulLeakyReluInvocation       // host侧的核函数调用程序，包含CPU侧和NPU侧两种运行验证方法
│   └── MatmulLeakyReluInvocationAsync  // host侧的核函数调用程序，包含CPU侧和NPU侧两种运行验证方法，使用了Matmul API异步Iterate接口
```

## 编译运行样例算子
针对自定义算子工程，编译运行包含如下步骤：
- 编译自定义算子工程；
- 调用执行自定义算子；

详细操作如下所示。

### 1. 获取源码包
编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)获取源码包。

### 2. 编译运行样例工程
- [CppExtensions样例运行](./CppExtensions/README.md)
- [MatmulLeakyReluInvocation样例运行](./MatmulLeakyReluInvocation/README.md)
- [MatmulLeakyReluInvocationAsync样例运行](./MatmulLeakyReluInvocationAsync/README.md)

## 更新说明
| 时间       | 更新事项                      | 注意事项                                         |
| ---------- | ----------------------------- | ------------------------------------------------ |
| 2024/01/04 | 新增Kernel Launch调用算子样例 | 需要基于社区CANN包7.0.0.alpha003及之后版本运行   |
| 2024/02/23 | 新增pybind11调用算子样例      | 需要基于社区CANN包8.0.RC1.alpha001及之后版本运行 |
| 2024/05/21 | 新增README                    |                                                  |
| 2024/05/25 | 取消TCubeTiling大小硬编码 | 所有样例需要基于社区CANN包8.0.RC2.alpha002及之后版本运行 |
| 2024/06/11 | 取消workspace大小硬编码 |                                        |
| 2024/06/19 | 新增MatmulLeakyRelu异步Iterate接口调用样例 | 本样例仅支持Atlas A2训练系列产品 |
| 2024/12/31 | 样例目录调整 |                                        |

## 已知issue

  暂无
