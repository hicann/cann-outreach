# 东南大学

## 团队信息

- 提交者: 王博_71124114
- 身份: 学生
- 单位: 东南大学

## 成员

- 王博_71124114 (wblxr408): 提交者

## 算子: op_03_relu

本 PR 是 CANN 训练营东南大学专场提交的 op_03_relu 算子实现，完整交付了一个面向 ascend910b 的 ReLU 算子，涵盖算子定义（relu_def.cpp）、InferShape（relu_infershape.cpp）、Tiling（relu_tiling.cpp）、AscendC Kernel（relu.cpp/relu.h）以及配套的构建脚本、aclnn 调用示例和基于 gtest 的 Host/Kernel 单元测试，用于实现 y = max(x, 0) 的逐元素计算。

主要改动

新增算子 Host 侧定义与推导逻辑: relu_def.cpp 定义 Relu 算子，输入输出均为 x/y，支持 DT_FLOAT、DT_FLOAT16 及 FORMAT_ND；relu_infershape.cpp 的 InferShapeRelu 将输入 shape 原样复制为输出 shape。
新增 Tiling 逻辑与数据结构: relu_tiling.cpp 依据元素总数、coreNum、UB 大小计算 blockDim/blockFactor/ubFactor（支持双缓冲），并按数据类型（DT_FLOAT16 与其它）通过 SetTilingKey 选择不同调度模式；relu_tiling_data.h 定义 ReluTilingData 结构，relu_tiling_key.h 声明 RELU_TPL_SCH_MODE_0/1 模板参数。
实现算子 Kernel: 模板类 NsRelu::Relu&lt;T&gt; 采用 VECIN/VECOUT 双缓冲队列，通过 CopyIn/Compute/CopyOut 完成搬运、AscendC::Relu 计算与写回；relu.cpp 的入口函数 relu 根据 tiling key 实例化 half 或 float 的 Process() 流程。
新增构建与调用示例: build.sh 提供 cmake 编译、UT 运行（-u）、示例运行（-e）等选项；examples/test_aclnn_relu.cpp 演示通过 aclnnReluGetWorkspaceSize 与 aclnnRelu 两段式接口在 NPU 上调用该算子。
搭建 Host 与 Kernel 单元测试框架: tests/ut 下提供 gtest 基础设施（含从源码以 OLD ABI 编译 gtest 的 BuildGoogleTest.cmake、TilingContextFaker/InferShapeContextFaker 等上下文模拟器），test_relu_tiling.cpp 校验 tiling 结果，test_relu.cpp 通过 ICPU_RUN_KF 运行 Kernel，并配合 gen_data.py/compare_data.py 依据 MERE/MARE 阈值对输出做精度比对。
