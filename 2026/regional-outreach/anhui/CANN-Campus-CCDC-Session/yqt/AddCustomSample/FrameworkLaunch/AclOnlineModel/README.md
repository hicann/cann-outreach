## 使用aclopCompile模型调用的方式调用AddCustom算子工程
该样例暂不支持Atlas 200/500 A2 推理产品。

## 运行样例算子
### 1.编译算子工程
运行此样例前，请参考[编译算子工程](../README.md#operatorcompile)完成前期准备。
### 2.aclopCompile调用的方式调用样例运行

  - 进入到样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/tutorials/AddCustomSample/FrameworkLaunch/AclOnlineModel
    ```

  - 样例执行

    样例执行过程中会自动生成测试数据，然后编译与运行aclopCompile调用方式的模型调用样例，最后检验运行结果。具体过程可参见run.sh脚本。
    ```bash
    bash run.sh
    ```
## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2024/05/22 | 新增本readme |
| 2024/12/31 | 样例目录调整 |