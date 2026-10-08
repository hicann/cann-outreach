# op_01_sub

提交团队: 东北大学
提交者: yzc


## 算子实现介绍

多核矢量减法，z = x - y。输入输出形状 (8, 2048)，ND 格式，支持 float32 与 float16。Host 侧按 32 字节对齐把数据切到多个 Vector Core，前若干核多处理一个对齐块，其余核处理尾块，保证元素不丢、搬运长度对齐。每个核内用 TPipe 双缓冲（BUFFER_NUM=2），单次 tile 长度为 2048：DataCopy 将 x、y 搬入 Local Memory，AscendC::Sub 做逐元素减法，再 DataCopy 写回 z。
