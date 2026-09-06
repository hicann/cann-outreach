# 北京邮电大学

## 团队信息

- 提交者: hyjw6
- 身份: 学生
- 单位: 北京邮电大学

## 成员

- hyjw6 (hyjw6): 提交者

## 算子: op_02_mul

Tiling 策略：
1. 将输入张量展平并划分到 8 个 AIV 核，每核处理 2048 个元素。
2. 每核采用单 Tile、单 Buffer，减少小 Tile 循环和队列调度开销。
3. GM 搬运长度满足 32B 对齐，计算长度满足 256B Vector Repeat 对齐。

Kernel 实现：
1. CopyIn：将 x、y 从 GM 搬入 UB。
2. Compute：使用显式 mask 和 repeatTime 调用 AscendC::Mul。
3. CopyOut：将计算结果从 UB 搬回 GM。
4. FP32 使用 mask=64、repeatTime=32；FP16 使用 mask=128、repeatTime=16。

同时补充输出 Shape 和 DataType 推导，输出 z 的 Shape、DataType 均与输入 x 保持一致。
