## 概述
本样例介绍Add算子的核函数直调方法。
## 目录结构介绍
```
├── KernelLaunch                      // 使用核函数直调的方式调用Add自定义算子
│   ├── AddKernelInvocationNeo        // Kernel Launch方式调用核函数样例
│   ├── AddKernelInvocationTilingNeo  // Kernel Launch方式调用核函数样例，带有Tiling
│   └── CppExtensions                 // pybind方式调用核函数样例，带有Tiling
```
## 编译运行样例算子
针对自定义算子工程，编译运行包含如下步骤：
- 编译自定义算子工程；
- 调用执行自定义算子；

详细操作如下所示。
### 1. 获取源码包
编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)获取源码包。
### 2. 编译运行样例工程
- [AddKernelInvocationNeo样例运行](./AddKernelInvocationNeo/README.md)
- [AddKernelInvocationTilingNeo样例运行](./AddKernelInvocationTilingNeo/README.md)
- [CppExtensions样例运行](./CppExtensions/README.md)
## 更新说明
| 时间       | 更新事项                                            | 注意事项                                       |
| ---------- | --------------------------------------------------- | ---------------------------------------------- |
| 2023/10/09 | 新增AddCustomSample/KernelLaunch样例                |                                                |
| 2024/01/04 | 新增AddKernelInvocationNeo样例                      | 需要基于社区CANN包7.0.0.alpha003及之后版本运行 |
| 2024/01/04 | 新增AddKernelInvocationTilingNeo样例                | 需要基于社区CANN包7.0.0.alpha003及之后版本运行 |
| 2024/02/02 | 新增AddCustomSample/KernelLaunch/CppExtensions样例  | 需要基于社区CANN包7.0.0.alpha003及之后版本运行 |
| 2024/05/22 | 更新readme结构                                      | 需要基于社区CANN包7.0.0.alpha003及之后版本运行 |
| 2024/06/06 | AddKernelInvocation样例转维护，不再更新，不推荐使用 |
| 2024/08/11 | 删除AddKernelInvocation样例 |
| 2024/12/31 | 样例目录调整 |   |
