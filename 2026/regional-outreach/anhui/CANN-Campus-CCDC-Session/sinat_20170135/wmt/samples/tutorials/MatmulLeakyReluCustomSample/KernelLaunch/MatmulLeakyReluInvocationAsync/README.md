## 目录结构介绍
```
├── MatmulLeakyReluInvocationAsync
│   ├── cmake                               // 编译工程文件
│   ├── scripts
│   │   ├── verify_result.py                // 真值对比文件
│   │   └── gen_data.py                     // 输入数据和真值数据生成脚本文件
│   ├── CMakeLists.txt                      // 编译工程文件
│   ├── data_utils.h                        // 数据读入写出函数
│   ├── main.cpp                            // 主函数，调用算子的应用程序，含CPU域及NPU域调用
│   ├── matmul_leakyrelu_custom_tiling.cpp  // 算子tiling实现
│   ├── matmul_leakyrelu_custom.cpp         // 算子kernel实现
│   └── run.sh                              // 编译运行算子的脚本
```
## 代码实现介绍
本样例中实现的是[m, n, k]固定为[1024, 640, 256]的MatmulLeakyRelu算子。
- kernel实现  
  MatmulLeakyRelu算子的数学表达式为：
  ```
  C = A * B + Bias
  C = C > 0 ? C : C * 0.001
  ```
  其中A的形状为[1024, 256]，B的形状为[256, 640]，C的形状为[1024, 640]，Bias的形状为[640]。具体请参考[matmul_leakyrelu_custom.cpp](./matmul_leakyrelu_custom.cpp)。

  本样例功能与MatmulLeakyReluInvocation相同，唯一区别在于本样例kernel实现调用的是Matmul高阶API的async异步Iterate和GetTensorC接口，接口详细信息可参考[Ascend C 高阶API手册](https://hiascend.com/document/redirect/CannCommunityAscendCHighLevelApi)。本样例代码中的调用方式如下：
  ```cpp
  matmulObj.template Iterate<false>();
  ```

- 调用实现  
  1. CPU侧运行验证主要通过ICPU_RUN_KF CPU调测宏等CPU调测库提供的接口来完成；
  2. NPU侧运行验证主要通过使用ACLRT_LAUNCH_KERNEL内核调用宏来完成。

  应用程序通过ASCENDC_CPU_DEBUG 宏区分代码逻辑运行于CPU侧还是NPU侧。

## 运行样例算子
  - 打开样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/tutorials/MatmulLeakyReluCustomSample/KernelLaunch/MatmulLeakyReluInvocationAsync
    ```
  - 配置环境变量

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
    
  - 样例执行

    ```bash
    bash run.sh -r [RUN_MODE] -v  [SOC_VERSION]
    ```
    - RUN_MODE：编译方式，可选择CPU调试，NPU仿真，NPU上板。支持参数为[cpu / sim / npu]，默认值为npu。
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下参数取值（xxx请替换为具体取值）：
      - Atlas A2训练系列产品/Atlas 800I A2推理产品参数值：AscendxxxB1、AscendxxxB2、AscendxxxB3、AscendxxxB4

    注：本样例仅支持Atlas A2训练系列产品/Atlas 800I A2推理产品。

    示例如下。
    ```bash
    bash run.sh -r npu -v Ascendxxxyy
    ```

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2024/06/19 | 新增本readme |
| 2024/12/31 | 样例目录调整 |