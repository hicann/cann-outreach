/*
 * -----------------------------------------------------------------------------
 * 文件说明：
 *   TanhCustom 算子的 Kernel 侧实现（运行在 Ascend AI Core 上）
 *
 * 计算逻辑：y = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 *   使用 AscendC 基础 API（Exp, Muls, Add, Sub, Div）在向量单元上逐段计算
 *
 * Pipeline 架构（三阶段流水线 + 双缓冲）：
 *   CopyIn  (MTE2): 将 x 从 Global Memory 搬到 L1 Buffer (inQueueX)
 *   Compute (V):    从 inQueueX 取数 → Exp/Muls/Add/Sub/Div 计算 →
 *                    结果写入 outQueueY
 *   CopyOut (MTE3): 将结果从 outQueueY 搬回 Global Memory
 *
 *   双缓冲：BUFFER_NUM=2，CopyIn(i+1) 与 Compute(i) 可并行执行
 *          循环共 tileNum * BUFFER_NUM 次迭代，每次搬 tileLength 个元素
 * -----------------------------------------------------------------------------
 */

#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

// ---------------------------------------------------------------------------
// BUFFER_NUM: 双缓冲深度
// 每个 TQue 有 2 个 slot，交替使用，实现数据搬运与计算的重叠（乒乓缓冲）
// ---------------------------------------------------------------------------
constexpr int32_t BUFFER_NUM = 2;

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}

    /*
     * ---------------------------------------------------------------------------
     * Init —— 初始化函数
     * ---------------------------------------------------------------------------
     * 功能：
     *   1. 根据 totalLength 和 GetBlockNum() 计算本 Core 负责的元素数
     *   2. 定位本 Core 在 Global Memory 中的输入/输出起始地址
     *   3. 计算每个 pipeline 迭代处理的元素数（tileLength）
     *   4. 初始化 TPipe 中的各个 Buffer（队列 + 临时计算缓冲）
     *
     * 参数：
     *   x, y:       Global Memory 中整个输入/输出张量的起始地址
     *   totalLength: 所有 Core 需要处理的总元素个数
     *   tileNum:     每个 Core 内部的 Tile 数量
     *
     * 地址偏移计算：
     *   blockLength = totalLength / GetBlockNum()    // 每个 Core 分到的元素数
     *   本 Core 起始地址 = 基地址 + blockLength * GetBlockIdx()
     * ---------------------------------------------------------------------------
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // ---- 计算本 Core 处理的元素数 ----
        // totalLength 是所有 Core 的总量，除以 GetBlockNum() 得到每个 Core 的量
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;

        // ---- 计算每次 pipeline 迭代处理的元素数 ----
        // blockLength 除以 (tileNum * BUFFER_NUM) 得到每次搬运的元素数
        // 例如: blockLength=1024, tileNum=8, BUFFER_NUM=2 → tileLength=64
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // ---- 设置 Global Memory 视图 ----
        // 每个 Core 只看到自己负责的那一段数据
        // GetBlockIdx() 返回当前 Core 的编号 [0, GetBlockNum()-1]
        // 通过偏移实现"逻辑切分"，无需物理切分数据
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + this->blockLength * AscendC::GetBlockIdx(),
                            this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + this->blockLength * AscendC::GetBlockIdx(),
                            this->blockLength);

        // ---- 初始化 Pipeline Buffer ----
        // inQueueX:  输入数据队列，BUFFER_NUM 个 slot，每个 slot 大小为 tileLength 个元素
        // outQueueY: 输出数据队列，同理
        // tmpBuf0/1/2: 计算临时缓冲，各 tileLength 个元素
        pipe.InitBuffer(inQueueX,  BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf0,  this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf1,  this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf2,  this->tileLength * sizeof(DTYPE_Y));
    }

    /*
     * ---------------------------------------------------------------------------
     * Process —— 主处理循环
     * ---------------------------------------------------------------------------
     * 循环 tileNum * BUFFER_NUM 次，每次执行 CopyIn → Compute → CopyOut
     * TPipe 框架自动管理双缓冲的同步，保证数据依赖正确
     * ---------------------------------------------------------------------------
     */
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    /*
     * ---------------------------------------------------------------------------
     * CopyIn —— 数据搬入（Global Memory → L1 Buffer）
     * ---------------------------------------------------------------------------
     * 从 GlobalTensor 中读取 tileLength 个元素，放入输入队列 inQueueX
     *
     * 流程：
     *   1. AllocTensor: 从输入队列分配一个空闲 buffer slot
     *   2. DataCopy:    将 xGm[offset] 处的数据 DMA 到该 slot
     *   3. EnQue:       将该 slot 提交到队列，供 Compute 阶段消费
     *
     * progress: 当前迭代编号 [0, tileNum*BUFFER_NUM)
     * offset = progress * tileLength
     * ---------------------------------------------------------------------------
     */
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 从输入队列分配一个 buffer slot
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();

        // 计算本次搬运的起始偏移（以元素为单位）
        uint32_t offset = progress * this->tileLength;

        // Global → Local 数据搬运
        AscendC::DataCopy(xLocal, xGm[offset], this->tileLength);

        // 提交到队列，供 Compute 阶段消费
        inQueueX.EnQue(xLocal);
    }

    /*
     * ---------------------------------------------------------------------------
     * Compute —— 核心计算（向量单元）
     * ---------------------------------------------------------------------------
     * 从输入队列取出数据，使用 AscendC 基础 API 计算 tanh，结果放入输出队列
     *
     * 数学公式：tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
     *
     * 计算步骤（3 个临时 buffer 的使用策略）：
     *   Step 1: tmpBuf0 = Exp(x)           // e^x
     *   Step 2: tmpBuf1 = Muls(x, -1.0)    // -x
     *   Step 3: tmpBuf1 = Exp(tmpBuf1)     // e^(-x)  (复用 tmpBuf1)
     *   Step 4: tmpBuf2 = Sub(tmpBuf0, tmpBuf1)  // 分子: e^x - e^(-x)
     *   Step 5: tmpBuf0 = Add(tmpBuf0, tmpBuf1)  // 分母: e^x + e^(-x)
     *   Step 6: y       = Div(tmpBuf2, tmpBuf0)  // 结果: 分子/分母
     *
     * 流程：
     *   1. DeQue:        从输入队列取出数据
     *   2. Exp/Muls/Add/Sub/Div: 6 步基础运算
     *   3. AllocTensor:  从输出队列分配一个 buffer slot
     *   4. EnQue:        将结果提交到输出队列，供 CopyOut 阶段消费
     *   5. FreeTensor:   释放已消费的输入 slot
     * ---------------------------------------------------------------------------
     */
    __aicore__ inline void Compute(int32_t progress)
    {
        // 从输入队列取出本段数据
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();

        // 从输出队列分配一块 buffer 用于存放结果
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // CANN 9.0: TBuf 不能隐式转换为 LocalTensor，需要通过 .Get<T>() 获取
        AscendC::LocalTensor<DTYPE_X> tmpTensor0 = tmpBuf0.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> tmpTensor1 = tmpBuf1.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> tmpTensor2 = tmpBuf2.Get<DTYPE_Y>();

        // ============================================================
        // Step 1: tmpTensor0 = e^x
        //     Exp 是 AscendC 向量级一元函数，对每个元素计算指数
        // ============================================================
        AscendC::Exp(tmpTensor0, xLocal, this->tileLength);

        // ============================================================
        // Step 2: tmpTensor1 = -x
        //     Muls 是向量-标量乘法，标量为 -1.0 即取相反数
        // ============================================================
        AscendC::Muls(tmpTensor1, xLocal, (DTYPE_X)-1.0, this->tileLength);

        // ============================================================
        // Step 3: tmpTensor1 = e^(-x)   [复用 tmpTensor1，覆盖之前的 -x]
        // ============================================================
        AscendC::Exp(tmpTensor1, tmpTensor1, this->tileLength);

        // ============================================================
        // Step 4: tmpTensor2 = e^x - e^(-x)  → 分子
        //     Sub 是向量级二元减法
        // ============================================================
        AscendC::Sub(tmpTensor2, tmpTensor0, tmpTensor1, this->tileLength);

        // ============================================================
        // Step 5: tmpTensor0 = e^x + e^(-x)  → 分母  [复用 tmpTensor0]
        // ============================================================
        AscendC::Add(tmpTensor0, tmpTensor0, tmpTensor1, this->tileLength);

        // ============================================================
        // Step 6: yLocal = tmpTensor2 / tmpTensor0  → 最终 tanh 结果
        // ============================================================
        AscendC::Div(yLocal, tmpTensor2, tmpTensor0, this->tileLength);

        // 将计算结果提交到输出队列，供 CopyOut 消费
        outQueueY.EnQue<DTYPE_Y>(yLocal);

        // 释放已消费的输入 buffer，使其可被 CopyIn 复用
        inQueueX.FreeTensor(xLocal);
    }

    /*
     * ---------------------------------------------------------------------------
     * CopyOut —— 数据搬出（L1 Buffer → Global Memory）
     * ---------------------------------------------------------------------------
     * 从输出队列取出计算结果，写回 Global Memory
     *
     * 流程：
     *   1. DeQue:     从输出队列取出已计算好的数据
     *   2. DataCopy:  将数据 DMA 到 Global Memory 对应位置
     *   3. FreeTensor: 释放 buffer slot，使其可被 Compute 复用
     * ---------------------------------------------------------------------------
     */
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从输出队列取出已计算好的结果
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();

        // 计算本次写回的起始偏移
        uint32_t offset = progress * this->tileLength;

        // Local → Global 数据写回
        AscendC::DataCopy(yGm[offset], yLocal, this->tileLength);

        // 释放输出 buffer，使其可被 Compute 阶段的 AllocTensor 复用
        outQueueY.FreeTensor(yLocal);
    }

private:
    // -----------------------------------------------------------------------
    // Pipeline 管理对象
    //   TPipe: 管理三阶段流水线的同步和 buffer 生命周期
    // -----------------------------------------------------------------------
    AscendC::TPipe pipe;

    // -----------------------------------------------------------------------
    // 数据队列（TQue = Tensor Queue）
    //   VECIN  → 向量输入方向（CopyIn → Compute）
    //   VECOUT → 向量输出方向（Compute → CopyOut）
    //   BUFFER_NUM=2 → 双缓冲深度
    // -----------------------------------------------------------------------
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    // -----------------------------------------------------------------------
    // 计算临时 Buffer（VECCALC 位置，供向量单元使用）
    //   tmpBuf0: 存放 e^x，复用为分母 e^x + e^(-x)
    //   tmpBuf1: 存放 -x，复用为 e^(-x)
    //   tmpBuf2: 存放分子 e^x - e^(-x)
    // -----------------------------------------------------------------------
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0, tmpBuf1, tmpBuf2;

    // -----------------------------------------------------------------------
    // Global Memory 张量视图
    //   每个 Core 通过 SetGlobalBuffer + GetBlockIdx() 只看到自己的数据段
    //   operator[] 返回从指定偏移开始的子视图
    // -----------------------------------------------------------------------
    AscendC::GlobalTensor<DTYPE_X> xGm;  // 输入 x 在 GM 中的视图
    AscendC::GlobalTensor<DTYPE_Y> yGm;  // 输出 y 在 GM 中的视图

    // -----------------------------------------------------------------------
    // Tiling 参数
    //   blockLength: 本 Core 负责的总元素数（= totalLength / GetBlockNum()）
    //   tileNum:     逻辑 Tile 数量（从 TilingData 传入）
    //   tileLength:  每次 pipeline 迭代处理的元素数
    // -----------------------------------------------------------------------
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

/*
 * ---------------------------------------------------------------------------
 * Kernel 入口函数
 * ---------------------------------------------------------------------------
 * 由 CANN 框架调用，每个 AI Core 执行一次
 *
 * 参数说明：
 *   x, y:      输入/输出张量在 Global Memory 中的基地址
 *   workspace: 工作空间指针（本算子不使用）
 *   tiling:    Tiling 数据指针，通过 GET_TILING_DATA 宏转换为结构体
 *
 * 执行流程：
 *   1. REGISTER_TILING_DEFAULT: 注册 TilingData 类型（编译期宏）
 *   2. GET_TILING_DATA:         将 tiling 裸指针转为 TanhCustomTilingData 结构体
 *   3. op.Init():               初始化 buffer、计算本 Core 偏移
 *   4. op.Process():            执行三阶段流水线循环
 * ---------------------------------------------------------------------------
 */
extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y,
                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    // 注册 TilingData 类型（msopgen 生成的宏，展开为类型注册代码）
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);

    // 将 tiling 裸指针转换为结构化数据
    // 展开后：TanhCustomTilingData tilingData; InitTanhCustomTilingData(tiling, &tilingData);
    GET_TILING_DATA(tilingData, tiling);

    // 实例化算子，传入地址和 tiling 参数
    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);

    // 启动三阶段流水线
    op.Process();
}
