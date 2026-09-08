## 概述
本样例基于MatmulCustom算子工程，介绍了单算子工程、单算子调用、第三方框架调用。

## 目录结构介绍
```
├──FrameworkLaunch             // 使用框架调用的方式调用Matmul自定义算子。
│   ├── AclNNInvocation        // 通过aclnn调用的方式调用MatmulCustom算子工程。
│   ├── MatmulCustomMultiCore  // 多核MatmulCustom算子工程。
│   ├── MatmulCustomSingleCore // 单核MatmulCustom算子工程。
│   └── MatmulCustom.json      // MatmulCustom算子的原型定义json文件。
```

## 算子描述
Matmul高阶API实现了快速的Matmul矩阵乘法的运算操作。

Matmul的计算公式为：
```
C = A * B + Bias
```
- A、B为源操作数，A为左矩阵，形状为\[M, K]；B为右矩阵，形状为\[K, N]。
- C为目的操作数，存放矩阵乘结果的矩阵，形状为\[M, N]。
- Bias为矩阵乘偏置，形状为\[N]。对A*B结果矩阵的每一行都采用该Bias进行偏置。

## 算子规格描述
在框架调用样例中，算子实现<a href="./MatmulCustomMultiCore"> MatmulCustomMultiCore </a>和<a href="./MatmulCustomSingleCore"> MatmulCustomSingleCore </a>支持的shape为：M = 1024, N = 640, K = 256。
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
</tr>
<tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">a</td><td align="center">M * K</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">b</td><td align="center">K * N</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">bias</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_custom</td></tr>
</table>

## 支持的产品型号
本样例支持如下产品型号：
- Atlas 推理系列产品AI Core
- Atlas A2训练系列产品/Atlas 800I A2推理产品

## 算子工程介绍
本样例介绍了多核场景（[MatmulCustomMultiCore](./MatmulCustomMultiCore/)）和单核场景（[MatmulCustomSingleCore](./MatmulCustomSingleCore/)）两种MamMul算子实现。可以根据使用场景，自行选择多核算子工程或单核算子工程，并在编译算子工程时，进入选择的算子实现工程中完成编译和安装。

以多核算子工程为例，算子工程目录MatmulCustomMultiCore包含算子实现的模板文件、编译脚本等，如下所示：

```
├── MatmulCustomMultiCore  // Matmul自定义算子工程
│   ├── cmake
│   ├── framework          // 算子插件实现文件目录，单算子模型文件的生成不依赖算子适配插件，无需关注
│   ├── op_host            // host侧实现文件
│   ├── op_kernel          // kernel侧实现文件
│   ├── scripts            // 自定义算子工程打包相关脚本所在目录
│   ├── CMakeLists.txt     // 算子工程的CMakeLists.txt
│   ├── CMakePresets.json  // 编译配置项
│   └── build.sh           // 编译入口脚本
```
CANN软件包中提供了工程创建工具msopgen，MatmulCustom算子工程可通过MatmulCustom.json自动创建，具体请参考[Ascend C算子开发](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)>工程化算子开发>创建算子工程 章节。

## 编译运行样例算子
针对自定义算子工程，编译运行包含如下步骤：
- 编译自定义算子工程生成算子安装包；
- 安装自定义算子到算子库中；
- 调用执行自定义算子；

详细操作如下所示。
### 1. 获取源码包
编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)获取源码包。

### 2. 编译算子工程<a name="compileop"></a>
编译自定义算子工程，构建生成自定义算子包。可自行选择多核或单核算子实现，这里以多核算子实现工程MatmulCustomMultiCore为例，具体步骤如下：
  - 执行如下命令，切换到算子工程MatmulCustomMultiCore目录。   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/tutorials/MatmulCustomSample/FrameworkLaunch/MatmulCustomMultiCore
    ```

  - 修改CMakePresets.json中ASCEND_CANN_PACKAGE_PATH为CANN软件包安装后的实际路径。

    ```json
    {
        ……
        "configurePresets": [
            {
                    ……
                    "ASCEND_CANN_PACKAGE_PATH": {
                        "type": "PATH",
                        "value": "/usr/local/Ascend/ascend-toolkit/latest"        //请替换为CANN软件包安装后的实际路径。eg:/home/HwHiAiUser/Ascend/ascend-toolkit/latest
                    },
                    ……
            }
        ]
    }
    ```
  - 在算子工程MatmulCustomMultiCore目录下执行如下命令，进行算子工程编译。

    ```bash
    ./build.sh
    ```
编译成功后，会在当前目录下创建build_out目录，并在build_out目录下生成自定义算子安装包custom_opp_\<target os>_\<target architecture>.run，例如“custom_opp_ubuntu_x86_64.run”。

备注：如果要使用dump调试功能，需要移除op_host内和CMakeLists.txt内的Atlas 训练系列产品、Atlas 200/500 A2 推理产品的配置。

### 3. 部署算子包
执行如下命令，在自定义算子安装包所在路径下，安装自定义算子包。

  ```bash
  cd build_out
  ./custom_opp_<target os>_<target architecture>.run
  ```

命令执行成功后，自定义算子包中的相关文件将部署至当前环境的OPP算子库的vendors/customize目录中。

### 4. 配置环境变量

  请根据当前环境上CANN开发套件包的[安装方式](https://hiascend.com/document/redirect/CannCommunityInstSoftware)，选择对应配置环境变量的命令。
  - 默认路径，root用户安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    ```
  - 默认路径，非root用户安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    ```
  - 指定路径install_path，安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=${install_path}/ascend-toolkit/latest
    ```

### 5. 调用执行算子工程
- [aclnn调用MatmulCustom算子工程](./AclNNInvocation/README.md)

## 更新说明
| 时间       | 更新事项                |
| ---------- | ----------------------- |
| 2023/11/02 | 新增AclNNInvocation样例 |
| 2024/05/27 | 更新readme              |
| 2024/12/31 | 样例目录调整 |
| 2025/01/16 | 更新readme              |

## 已知issue
  暂无