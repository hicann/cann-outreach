# 西安邮电大学

## 团队信息

- 提交者: yuelin_1
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- yuelin_1 (yuelin_1): 提交者

## 算子: op_03_relu

入口分发:relu&lt;schMode&gt; 用 if constexpr按编译期模板参数选择 half/float 实例，零运行时开销；tilingKey 由 host 按输入 dtype 写入
单队列原地：ReLU 是逐元素单输入算子，无需独立输出缓冲，Relu(x, x)直接变换，UB 从 4 份减到 2 份 —— 这是 "更快" 的核心
多核粒度：MIN_SPLIT_THRESHOLD=8192让每核至少 8K 元素，避免小数据下同步开销占比过高；测试点因此用 12 核而非 64 核
尾部安全：coreLength_ = min(blockFactor, totalNum − offset)，越界核直接跳过，段尾用 currentNum钳制，不越界。
