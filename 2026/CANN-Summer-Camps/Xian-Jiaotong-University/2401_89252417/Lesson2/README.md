# SubCustomTemplate 矢量减法算子

本工程由 `sub_custom_template.json` 通过 `msopgen` 生成并补全，实现：

```text
z = x - y
```

- Shape：`(8, 2048)`
- Format：`ND`
- DataType：`float16`、`float32`
- AI Core：8 个，每个 Core 处理一行（2048 个元素）
- Tiling：每行划分为 2 个 Tile
- 流水：`CopyIn -> Compute -> CopyOut`，输入/输出队列采用 Double Buffer

## 关键文件

- `custom_op/op_host/sub_custom_template.cpp`：Shape/DataType 推导、Tiling、TilingKey 设置
- `custom_op/op_kernel/sub_custom_template.cpp`：Ascend C 核函数
- `custom_op/op_kernel/sub_custom_template_tiling.h`：Host 与 Kernel 共用的 Tiling 数据
- `test/main.cpp`：float16 功能测试，输入分别为 1 和 2，期望输出为 -1

## 编译与运行

进入 CANN/Ascend 设备容器，确认已经配置 `ASCEND_TOOLKIT_HOME` 或
`ASCEND_HOME_PATH`，然后执行：

```bash
cd Lesson2
chmod +x run.sh custom_op/build.sh
./run.sh
```

成功时末尾输出：

```text
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
test pass
```

如需单独构建算子，可执行：

```bash
cd Lesson2/custom_op
./build.sh
```
