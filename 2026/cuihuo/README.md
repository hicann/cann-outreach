# 2026淬火行动

## 活动介绍
为支撑伙伴更好的使用昇腾并基于昇腾AI平台开发更好的产品和方案，助力伙伴市场项目成功，华为将在各区域开展“淬火行动”伙伴能力提升专项活动，赋能伙伴和开发者更深和更全的掌握昇腾软硬件技术，着重培养开发者的动手实测能力。

## 目录结构
```
cuihuo/
├── README.md                             # 本文档
├── jinan/                                # 济南地域活动
│   └── RiddleTan/                        # 开发者gitcode账号
│       └── TanhCustom/                   # 任务名称
├── shanghai/                             # 上海地域活动
```

## 提交规范

### 目录结构要求
每个任务提交目录结构如下：
```
{区域}/
└── {gitcode用户名}/
    └── {任务名称}/
        ├── build.sh                      # 编译脚本
        ├── CMakeLists.txt                # 顶层CMake
        ├── CMakePresets.json             # CMake预设配置
        ├── op_host/                      # Host侧代码
        │   ├── CMakeLists.txt
        │   └── {op_name}.cpp            # Tiling & 算子注册
        ├── op_kernel/                    # Kernel侧代码
        │   ├── CMakeLists.txt
        │   ├── {op_name}.cpp            # Kernel入口
        │   └── {op_name}_tiling.h       # TilingData结构
        └── framework/                   # 框架适配代码（可选）
            └── tf_plugin/                # TensorFlow插件（可选）
                ├── CMakeLists.txt
                └── tensorflow_{op_name}_plugin.cc
```

### 提交步骤
1. 在对应区域目录下创建以 **gitcode用户名** 命名的文件夹
2. 在用户名文件夹下创建以 **任务名称** 命名的文件夹
3. 按照上述目录结构提交算子代码
4. 提交 Pull Request 到本仓库

## 联系方式
- **官方邮箱**：community@cann-community.com
- **技术支持**：support@cann-community.com
- **Issue 反馈**：https://gitcode.com/cann/cann-competitions/issues
