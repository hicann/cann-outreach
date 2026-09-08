## 目录结构介绍
```
├── CppExtensions
│   ├── CMakeLists.txt                      // 编译工程文件
│   ├── matmul_leakyrelu_custom_tiling.cpp  // 算子tiling实现
│   ├── matmul_leakyrelu_custom.cpp         // 算子kernel实现
│   ├── matmul_leakyrelu_custom.py          // 输入数据和真值数据生成脚本文件以及kernel函数入口
│   ├── pybind11.cpp                        // 提供python调用的kernel函数主入口
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

- 调用实现  
  通过PyTorch框架进行模型的训练、推理时，会调用到很多算子进行计算，调用方式也和kernel编译流程相关。对于自定义算子工程，需要使用PyTorch Ascend Adapter中的OP-Plugin算子插件对功能进行扩展，让torch可以直接调用自定义算子包中的算子；对于KernelLaunch开放式算子编程的方式，也可以使用pytorch调用，此样例演示的就是这种算子调用方式。

  pybind11.cpp文件是一个C++的代码示例，使用了pybind11库来将C++代码封装成Python模块。该代码实现中定义了一个名为m的pybind11模块，其中包含一个名为run_matmul_leakyrelu_custom的函数。该函数与my_matmul_leakyrelu::run_matmul_leakyrelu_custom函数相同，用于将C++函数转成Python函数。在函数实现中，通过c10_npu::getCurrentNPUStream() 的函数获取当前NPU上的流，并调用ACLRT_LAUNCH_KERNEL宏启动自定义的Kernel函数matmul_leakyrelu_custom，在NPU上执行算子。

  在matmul_leakyrelu_custom_test.py调用脚本中，通过导入自定义模块matmul_leakyrelu_custom，调用自定义模块matmul_leakyrelu_custom中的run_matmul_leakyrelu_custom函数，在NPU上执行A和B的带Bias的矩阵乘操作，如果结果大于0则取原值，如果小于0则乘以0.001，最后保存在C中。

## 运行样例算子

  - 打开样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/tutorials/MatmulLeakyReluCustomSample/KernelLaunch/CppExtensions
    ```

  - 修改配置
    * 修改CMakeLists.txt内SOC_VERSION为所需产品型号
    * 修改CMakeLists.txt内ASCEND_CANN_PACKAGE_PATH为CANN包的安装路径

    SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下产品型号：
      - Atlas 推理系列产品AI Core
      - Atlas A2训练系列产品/Atlas 800I A2推理产品

  - 样例执行

    ```bash
    rm -rf build
    mkdir build
    cd build
    cmake ..
    make
    python ../matmul_leakyrelu_custom_test.py
    ```

    用户亦可参考run.sh脚本进行编译与运行，Ascendxxxyy请替换为实际的AI处理器型号。
    ```bash
    bash run.sh -v Ascendxxxyy
    ```

## 更新说明
| 时间       | 更新事项     | 注意事项                                         |
| ---------- | ------------ | ------------------------------------------------ |
| 2023/05/21 | 更新本readme |                                                 |
| 2023/05/25 | 取消TCubeTiling大小硬编码 | 需要基于社区CANN包8.0.RC2.alpha002及之后版本运行 |
| 2023/06/11 | 取消workspace大小硬编码 |                                        |
| 2024/12/31 | 样例目录调整 |                                        |