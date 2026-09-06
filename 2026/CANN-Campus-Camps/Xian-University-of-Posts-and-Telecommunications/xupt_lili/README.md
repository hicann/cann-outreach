# 西安邮电大学

## 团队信息

- 提交者: lili
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- lili (gcw_ct0G4Vyj): 提交者

## 算子: op_01_sub

基于多核并行 + Double Buffer 流水线实现逐元素减法算子 `z = x - y`。
Tiling 策略：将总元素长度按核数均分得到单 block 长度；单核内继续切分多 tile，采用双缓冲流水线，分三段 CopyIn、Compute、CopyOut 完成数据搬移与矢量减法计算；使用 AscendC::Sub 接口完成 UB 内矢量相减，支持 ND 格式输入输出 tensor，数据类型支持 float32/float16。
