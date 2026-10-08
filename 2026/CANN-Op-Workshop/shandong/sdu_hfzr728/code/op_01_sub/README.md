# op_01_sub

提交团队: 山东大学
提交者: 王观豪


## 算子实现介绍

一、实现要点
在模板要求的位置补齐了四部分：

要求	实现
struct SubCustomTilingData	{ uint32_t totalLength; uint32_t tileNum; }
class KernelSub	模板类，Init / Process / CopyIn / Compute / CopyOut 与模板骨架一致
__global__ __vector__ void sub_custom(...)	实例化 KernelSub&lt;T&gt; 并执行，签名与模板一致
extern "C" void run_kernel(...)	从 TensorGroupInfo 推 tiling，按 dtype 选 half/float 实例并启动
1. 多核切分按 32B 数据块（关键设计）
没有用 blockLength = totalLength / GetBlockNum() 这种按元素均分的写法 —— 那样在 totalLength 不能整除核数时会截断丢数据，而且各核起始地址不保证 32B 对齐。

改为以 32B 数据块为单位均分，余数分给前 remBlocks 个核：

cpp
复制
const uint32_t totalBlocks = (totalLength + ALIGN_ELEM - 1) / ALIGN_ELEM;
const uint32_t baseBlocks  = totalBlocks / blockNum;
const uint32_t remBlocks   = totalBlocks % blockNum;
const uint32_t startBlock  = blockIdx * baseBlocks + (blockIdx &lt; remBlocks ? blockIdx : remBlocks);
const uint32_t start       = startBlock * ALIGN_ELEM;   // 必为 32B 对齐
因为 start 一定是 ALIGN_ELEM 的整数倍，字节偏移必然是 32 的倍数，任何核数下都不会出现非对齐搬运。

2. 单核内 tile 切分
tileLength = coreLength / (tileNum × BUFFER_NUM) 再向下取整到 32B 对齐； 主体循环全部走 DataCopy 快路径，仅当存在长度不足的对齐残块时才走 DataCopyPad。

3. Double Buffer
沿用模板给的 TQue&lt;..., QUEUE_DEPTH=1&gt; + InitBuffer(que, BUFFER_NUM=2, size)。 按 CANN 的约定，双缓冲是在 InitBuffer 的 num 参数上开启，与 TQue 模板 depth 无关。

4. dtype 分派
dtype：0=fp32、1=fp16（取自 TensorInfo.dtype，见 data_utils.h 与 main.asc 注释）。 fp16 走 half 实例，其余走 float 实例 —— 与题目给的 kernel_add.asc 参考实现一致。 纯逐元素矢量减法无累加，fp16 不需要转 fp32 中间计算。

二、一个必须说明的技术判断：没有加 AscendC::InitSocState()
仓库里存在两种 __global__ __vector__ 写法，我做了全仓统计（13 个文件），规律是完全干净的：

写法	是否调用 InitSocState	例子
TPipe / TQue（本题模板所用）	否（0/9 个文件调用）	02.06 sub_custom_template.asc、02.09 sigmoid_custom.asc、本题参考 kernel_add.asc
LocalMemAllocator（静态 Tensor）	是（3/3 个文件调用）	02.05 sub_static.asc、02.09 sigmoid_static_tensor.asc
所以：InitSocState() 是静态 Tensor（LocalMemAllocator）编程范式的要求，对 TPipe/TQue 范式不适用。 本题模板是 TPipe/TQue，且题目自带的参考实现 kernel_add.asc 也没有调用它，因此不加。

（这是我原本最不确定的一点 —— 两种写法在仓库里都叫 __global__ __vector__，只看单个文件很容易加错或漏加。）

三、验证记录
3.1 本地无 NPU / 无 CANN，无法编译或上板
本机没有昇腾硬件、没有 CANN Toolkit、没有 numpy，不能本地编译或运行。 因此我把最可能出错的部分——切分算术——用 Python 完整复刻并做了穷举验证。

3.2 切分逻辑仿真结果（复刻 host+kernel 全部算术）
题目真实规格：

场景	核数	覆盖	非对齐偏移	走慢路径的 tile
fp32 (8,2048), availCore=40	8	16384/16384 ✅	0	0（全走 DataCopy 快路径）
fp16 (8,2048), availCore=40	8	16384/16384 ✅	0	0
fp32 (8,2048), availCore=8	8	16384/16384 ✅	0	0
fp32/fp16, availCore=4/2/1/0	4/2/1/8	全覆盖 ✅	0	0
压力测试：3320 组（n ∈ [1, 20000] 随机与边界、两种 dtype、核数 ∈ {1,2,3,5,8,20,40,64}）

覆盖遗漏：0
元素重复覆盖：0
非 32B 对齐的搬运偏移：0
失败用例：0 / 3320
其中 3080 个 tile 触发了 DataCopyPad 残块路径，说明该分支在非 2 的幂尺寸下是真正需要的，不是死代码。

⚠️ 验证边界：以上验证覆盖的是切分算术的完备性与对齐性，不覆盖 NPU 上的实际编译与数值结果。 真机结论仍需以平台评测为准。

3.3 精度预期
scripts/verify_result.py 对 case0 的判定是 rtol=1e-4, atol=0.1（np.isclose）。 金标准由 scripts/sub.py 用 float64 计算后一次性转回 fp32，输入取值在 [1,10)， 本实现为纯 fp32 逐元素相减，误差量级 ~1e-6，远小于 atol=0.1，精度上无风险。

四、与原模板的差异（有意为之）
位置	模板/参考	本实现	原因
多核切分	按元素 totalLength/blockDim	按 32B 块均分、余数前置	避免截断丢数 + 保证 32B 对齐
核数	硬编码 blockDim = 8	min(8, availableCoreNum, 数据块数)	用上了 run_kernel 传入的 availableCoreNum，且不会超发
循环次数	tileNum × BUFFER_NUM	coreLength / tileLength + 残块	数学等价，但对不能整除的尺寸不丢数据
残块	无	DataCopyPad	非 2 的幂形状下避免 GM 越界读写
对题目给定的 (8,2048) 规格，本实现与参考实现在切分结果上完全一致（每核 2048 元素、每块 128 元素、共 16 块）。
