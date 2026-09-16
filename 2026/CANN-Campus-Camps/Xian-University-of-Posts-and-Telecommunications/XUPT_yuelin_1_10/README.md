# 西安邮电大学

## 团队信息

- 提交者: yuelin_1
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- yuelin_1 (yuelin_1): 提交者

## 算子: op_10_add

1. 多核均分 + 段内流水
粗粒度并行：length / GetBlockNum()均分到每个 AIV 核，核间零通信（无共享读写，天然无竞争）
细粒度并行：每核内部按 4096 元素切片，双缓冲使搬数与计算重叠 —— 单核吞吐被流水填满
2. 内存策略 ——UB 按队列静态分配
3. 对齐策略
blockLength 向上取整到 32 的倍数 → 每核 GM 起始偏移（coreIdx×blockLength×sizeof (T)）恒为 32B 对齐，满足 DataCopy 与矢量指令的对齐硬约束；尾部不足 32 的元素由 CurNum 在段内钳制（count 可为任意值，对齐只约束地址不约束长度）
4. 边界策略
尾部核：processLength = length − offset（可能非 32 倍数）→ 最后一段 cur = processLength − progress×4096 精确钳制
空核：offset ≥ length → processLength=0 → loopNum=0，Process 零循环直接退出，无脏写
5. 数值策略
直接映射硬件 Add指令，每元素一次运算一次舍入；无中间累加、无重复处理段
