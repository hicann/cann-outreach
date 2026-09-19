# 西安邮电大学

## 团队信息

- 提交者: yuelin_1
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- yuelin_1 (yuelin_1): 提交者

## 算子: op_01_sub

矢量减法算子采用 Host/Device 两层架构，核心思路是：用Host 侧算好 tiling 参数并启动核函数，Device 侧每颗 Vector Core 用 "拷贝 — 计算 — 拷出" 三段流水并行处理自己负责的那一段数据。
Hostrun_kernel：纯 CPU 代码，负责 tiling 计算和以直调语法 sub_custom&lt;&lt;&lt;blockNum, nullptr, stream&gt;&gt;&gt;启动核函数，无需注册算子。
Device 侧sub_custom → KernelSub&lt;T&gt;用 TPipe + TQue 实现双缓冲流水，Sub(z, x, y, len)完成矢量减法。
