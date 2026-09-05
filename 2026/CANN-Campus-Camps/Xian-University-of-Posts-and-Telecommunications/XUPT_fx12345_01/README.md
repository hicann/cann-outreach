# 西安邮电大学

## 团队信息

- 提交者: fx12345
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- fx12345 (fx12345): 提交者

## 算子: op_01_sub

1.计算切分参数：从 tiling 取出 `totalLength` 和 `tileNum`，推导 `blockLength`、`tileLength`
2. 绑定全局内存：通过 `SetGlobalBuffer` 将 `xGm/yGm/zGm` 绑定到当前核负责的 GM 地址区间
3. 分配 UB 内存：`pipe.InitBuffer` 为三个队列各分配 `BUFFER_NUM=2` 块、每块 `tileLength × sizeof(T)` 字节的片上内存
