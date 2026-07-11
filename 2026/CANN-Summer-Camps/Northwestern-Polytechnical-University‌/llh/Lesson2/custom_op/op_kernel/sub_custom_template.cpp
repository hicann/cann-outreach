/**
 * @file    sub_custom_template.cpp
 * @brief   SubCustomTemplate 算子 — Device 侧 Kernel (MemBase)
 *
 * ============================================================================
 * 算子开发流程 (AscendC Kernel 直调 → 注册上库)
 * ============================================================================
 *
 *   Step 1  环境检查        → ascendc-env-check: CANN 版本、NPU 状态
 *   Step 2  算子设计        → DESIGN.md: Tiling 策略、API 映射、UB 预算
 *   Step 3  Kernel 开发     → 本文件: CopyIn / Compute / CopyOut 三级流水线
 *   Step 4  Host 开发       → op_host/sub_custom_template.cpp: 算子注册 + TilingFunc
 *   Step 5  测试工程        → test/main.cpp: ACL API 调用 + Golden 对比验证
 *   Step 6  编译部署        → bash run.sh: 编译算子包 → 部署 → msprof profiling
 *   Step 7  性能验收        → 查看 MTE/AIVector 利用率，验证 Task Duration
 *
 * ============================================================================
 * 技术路线: MemBase (LocalTensor + AscendC::Sub)
 * ============================================================================
 *
 *   MemBase 使用 LocalTensor 在 Unified Buffer (UB) 上直接操作，
 *   AscendC::Sub 编译为 Vector 引擎的高效 VSUB 指令。
 *   相比 RegBase (寄存器级 LoadAlign/Sub/StoreAlign)，代码更简洁，
 *   性能差异在 element-wise 算子中可忽略（瓶颈在 MTE 搬数而非计算）。
 *
 * ============================================================================
 * 三级流水线 + 双缓冲
 * ============================================================================
 *
 *            ┌─────────┐    ┌─────────┐    ┌──────────┐
 *   Global   │ CopyIn  │ →  │ Compute │ →  │ CopyOut  │  → Global
 *   Memory   │ (MTE)   │    │(Vector) │    │  (MTE)   │     Memory
 *   (DDR)    │ x,y→UB  │    │ z=x-y   │    │  UB→z    │     (DDR)
 *            └─────────┘    └─────────┘    └──────────┘
 *              inQueueX       outQueueZ
 *              inQueueY
 *
 *   BUFFER_NUM = 2 (双缓冲):
 *     时刻 t:  CopyIn(N+1) ‖ Compute(N) ‖ CopyOut(N-1)
 *     → MTE 搬运引擎与 Vector 计算引擎并行工作，利用率饱和
 *
 *   TPipe/TQue 的信用机制:
 *     AllocTensor 在 Queue 满时阻塞 → 自动背压
 *     EnQue/DeQue 触发依赖解析 → 硬件自动调度流水线重叠
 *
 * ============================================================================
 * Tile 分片计算
 * ============================================================================
 *
 *   totalLength  = shape[0] × shape[1]                       (例: 8×2048 = 16384)
 *   blockDim     = TilingFunc 自适应计算                      (例: 32)
 *   blockLength  = totalLength / blockDim                    (例: 512)
 *   tileLength   = blockLength / tileNum / BUFFER_NUM        (例: 256)
 *   loopCount    = tileNum × BUFFER_NUM                       (例: 2)
 *
 *   内核视角地址: xGm[blockIdx × blockLength + i × tileLength]
 */

#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

// ---------------------------------------------------------------------------
// 队列深度
//   1 → 串行执行, MTE 和 Vector 互等
//   2 → 双缓冲流水线, 搬 tile N+1 与算 tile N 并行
//   3 → 三缓冲, 延迟更大但流水更深 (UB 占用更多)
// ---------------------------------------------------------------------------
constexpr int32_t BUFFER_NUM = 2;

/**
 * @class KernelSubCustomTemplate
 * @brief  Element-wise Sub 算子 Kernel (MemBase)
 *
 * @tparam dtypeX 输入 X 的数据类型
 * @tparam dtypeY 输入 Y 的数据类型
 * @tparam dtypeZ 输出 Z 的数据类型
 *
 * 生命周期: Init() → Process()
 */
template <class dtypeX, class dtypeY, class dtypeZ>
class KernelSubCustomTemplate {
public:
    __aicore__ inline KernelSubCustomTemplate() {}

    /**
     * @brief 初始化 — 数据分片 + Buffer/Queue 配置
     *
     * @param x, y, z     Global Memory 地址
     * @param totalLength  数据总元素数
     * @param tileNum      Host TilingFunc 计算的每核切分轮数
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileNum)
    {
        // --- 分片计算 ---
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum     = tileNum;
        this->tileLength  = this->blockLength / tileNum / BUFFER_NUM;

        // --- Global Buffer 绑定 ---
        // 每核处理 [blockIdx * blockLength, (blockIdx+1) * blockLength)
        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + this->blockLength * AscendC::GetBlockIdx(),
                            this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + this->blockLength * AscendC::GetBlockIdx(),
                            this->blockLength);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + this->blockLength * AscendC::GetBlockIdx(),
                            this->blockLength);

        // --- Queue Buffer 分配 ---
        // 每个 Queue 在 UB 中分配 BUFFER_NUM 个 slot
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dtypeX));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(dtypeY));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(dtypeZ));
    }

    /**
     * @brief 主流程 — 驱动三级流水线循环
     *
     * 循环次数 = tileNum × BUFFER_NUM
     *
     * 硬件流水线时序 (BUFFER_NUM=2):
     *   iter 0: CopyIn(0)                              | 冷启动
     *   iter 1: CopyIn(1) ‖ Compute(0)                  | 启动重叠
     *   ...   : CopyIn(k) ‖ Compute(k-1) ‖ CopyOut(k-2) | 稳态三级满载
     *   末尾  :              Compute(last) ‖ CopyOut(last-1) | 排空
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
    /**
     * @brief CopyIn — MTE 引擎: GM → UB
     *
     * AllocTensor → DataCopy → EnQue 三步:
     *   1. 从 Queue 申请空闲 slot (满则阻塞等待 Compute 释放)
     *   2. MTE DMA 从 DDR 搬运 tileLength 个元素到 UB
     *   3. 放入队列通知下游 Compute
     */
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();

        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    /**
     * @brief Compute — Vector 引擎: z = x - y (UB 上计算)
     *
     * DeQue → AllocTensor → Sub → EnQue → FreeTensor 五步:
     *   1. 等待 CopyIn 填入数据
     *   2. 从输出队列申请 slot
     *   3. Vector 引擎执行逐元素减法
     *   4. 结果放入输出队列
     *   5. 归还输入 slot 给 CopyIn 复用
     */
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();

        // z[i] = x[i] - y[i],  处理 tileLength 个元素
        AscendC::Sub(zLocal, xLocal, yLocal, this->tileLength);

        outQueueZ.EnQue<dtypeZ>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    /**
     * @brief CopyOut — MTE 引擎: UB → GM
     *
     * DeQue → DataCopy → FreeTensor 三步:
     *   1. 等待 Compute 完成结果
     *   2. MTE DMA 从 UB 搬回 DDR
     *   3. 归还输出 slot 给 Compute 复用
     */
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    // ---- 硬件资源 ----
    AscendC::TPipe pipe;  // 流水线管理器, MTE/Vector 指令调度

    // ---- 数据搬运队列 ----
    // VECIN  → Vector 引擎输入侧, 由 MTE 填入数据
    // VECOUT → Vector 引擎输出侧, 由 MTE 搬出结果
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;

    // ---- Global Memory Tensor 描述符 ----
    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeY> yGm;
    AscendC::GlobalTensor<dtypeZ> zGm;

    // ---- 分片参数 ----
    uint32_t blockLength;   // 每核处理元素数 = totalLength / GetBlockNum()
    uint32_t tileNum;       // 每核 tile 轮数 (Host TilingFunc 传入)
    uint32_t tileLength;    // 单 tile 元素数 = blockLength / tileNum / BUFFER_NUM
};

/**
 * @brief Kernel 入口函数 — 每个 AI Core 执行一次
 *
 * REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT:
 *   从 GM tiling buffer 读取 Host 端 TilingFunc 打包的参数。
 *
 * DTYPE_X/DTYPE_Y/DTYPE_Z:
 *   由框架根据 op_host 注册时声明的 dtype 自动展开为 float / half。
 */
__global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(SubCustomTemplateTilingData, tiling_data, tiling);

    KernelSubCustomTemplate<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
