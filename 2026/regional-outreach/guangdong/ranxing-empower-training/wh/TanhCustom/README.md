# TanhCustom 自定义算子（AscendC）

基于昇腾 CANN 的 **Tanh（双曲正切）自定义算子**工程，输入/输出均为 FP16、ND 布局，
目标 AI Core：ascend910b。

## 功能

`y = tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)`，逐元素计算；内核采用双缓冲流水，
FP32 中间精度，输出 `CAST_ROUND` 回 FP16。

## 目录结构

```
TanhCustom/
├── build.sh                  # 一键构建脚本
├── CMakeLists.txt            # 顶层 CMake
├── CMakePresets.json         # CMake 预设（ascend910b / Release）
├── op_host/                  # 宿主侧：Tiling、算子注册、形状/类型推导
├── op_kernel/                # AI Core 内核：KernelTanh（双缓冲流水 + tanh 计算）
└── framework/tf_plugin/      # TensorFlow 算子注册插件（可选）
```

## 构建方法

```bash
# 需先安装 CANN 工具链并配置环境变量 ASCEND_HOME_PATH
bash build.sh
# 产物：build_out/custom_opp_almalinux_aarch64.run
```

## 编译产物（构建验证）

本地执行 `bash build.sh` 构建验证通过，生成的算子安装包
`build_out/custom_opp_almalinux_aarch64.run` 内含以下关键产物：

```
packages/vendors/customize/
├── op_api/lib/libcust_opapi.so                    # ACLNN 接口库
├── op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so   # 算子原型库
├── op_impl/ai_core/tbe/kernel/ascend910b/tanh_custom/TanhCustom_*.o  # AI Core 内核二进制
├── op_impl/ai_core/tbe/op_tiling/liboptiling.so   # Tiling 库
├── op_impl/ai_core/tbe/config/ascend910b/...      # 算子配置（ops-info / binary_info）
└── framework/tensorflow/libcust_tf_parsers.so     # TensorFlow 插件
```

- 构建环境：aarch64 / CANN 8.5.0 / cmake
- 目标计算单元：ascend910b（见 `CMakePresets.json`）
- 编译输出无错误，CPack 打包成功（生成 `custom_opp_almalinux_aarch64.run`）

## 提交说明

- 版权声明已采用通用写法（TanhCustom contributors），所有提交者无需修改，可直接提交。
- 提交前请删除本地 `build_out/` 构建产物（已通过 `.gitignore` 排除）。
