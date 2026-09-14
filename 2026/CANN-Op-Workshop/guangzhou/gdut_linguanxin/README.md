# 广东工业大学

## 团队信息

- 提交者: 林观鑫
- 身份: 学生
- 单位: 广东工业大学

## 成员

- 林观鑫 (gcw_pPraYcIU): 提交者

## 算子: op_01_sub

算子代码简介
 
本代码基于AscendC开发矢量减法算子，实现逐元素运算 \boldsymbol{z=x-y}，分为Host主机侧与Device设备侧两部分。Host侧读取输入张量信息，计算总数据长度、识别数据类型，完成Tiling任务划分，调度多个AICore核并行计算；Device侧采用分块Tile+Double Buffer双缓冲流水线架构，循环执行GM到UB的数据搬运、UB中调用 Sub 接口完成矢量减法运算、结果回写至GM，利用模板泛型同时支持fp16（half）与fp32（float）数据类型，通过多核并行与流水线重叠掩盖数据搬运开销，提升算子运算性能。
