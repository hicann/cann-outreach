# lesson1 MulCustom

## 内容

本目录完成课程 1 的 `mul_custom.asc` 作业，实现两个 `float` 向量的逐元素乘法：

```text
z[i] = x[i] * y[i]
```

## 文件说明

- `mul_custom.asc`：MulCustom 算子实现与测试入口。
- `CMakeLists.txt`：Ascend C 编译配置。
- `run.sh`：一键编译并运行测试。

## 编译运行

在 CANN 8.5 在线环境中执行：

```bash
bash run.sh
```

`run.sh` 会完成环境加载、CMake 配置、编译和运行：

```bash
source "$ASCEND_TOOLKIT_HOME/set_env.sh"
mkdir -p build
export ASC_DIR="$ASCEND_HOME_PATH/aarch64-linux/tikcpp/ascendc_kernel_cmake/"
cd build
cmake ..
make
./mul_test
```

## 验证结果

已在 CANN 8.5 在线环境中执行 `bash run.sh`，编译和运行均通过。

关键输出：

```text
[Success] Case accuracy is verification passed.
```
