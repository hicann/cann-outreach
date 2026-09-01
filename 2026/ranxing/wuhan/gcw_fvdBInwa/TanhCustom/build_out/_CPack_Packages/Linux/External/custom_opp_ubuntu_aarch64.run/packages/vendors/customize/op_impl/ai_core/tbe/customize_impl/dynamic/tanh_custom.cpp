#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

// 为什么这么写：BUFFER_NUM = 2 代表“双缓冲 (Ping-Pong Buffer)”机制。
// 就像有两个水桶，NPU 的计算单元在处理水桶 A 的数据时，搬运单元可以同时去装水桶 B，
// 这样计算和搬运重叠进行，掩盖了内存读取的延迟，让 NPU 性能跑满。
constexpr int32_t BUFFER_NUM = 2; 

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    
    // 1. 初始化：绑定内存地址，计算分片大小，申请 UB 空间
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // 告诉系统输入 x 和输出 y 在全局内存 (GM) 的起始位置和总长度
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, totalLength);
        
        this->tileNum = tileNum;
        this->blockLength = totalLength / tileNum; // 算出每一块包含多少个元素
        this->tileLength = this->blockLength;
        
        // 在统一内存 (UB) 中为输入和输出队列精确分配空间
        // 大小 = 单块长度 * 数据类型字节数。如果 blockLength 算错太大，这里会报 OOM (内存溢出)
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->blockLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->blockLength * sizeof(DTYPE_Y));
    }

    // 2. 流程控制：启动 CopyIn -> Compute -> CopyOut 的流水线循环
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
    // 3. 数据搬入：从慢速 GM 搬到快速 UB
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 向系统申请一块 UB 本地内存
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        
        // 核心原理：progress / BUFFER_NUM 算出当前正在处理“第几批”数据
        // 乘以 blockLength 得到在 GM 中的起始偏移量 (offset)
        uint32_t offset = (progress / BUFFER_NUM) * blockLength;
        
        // 执行高效硬件搬运指令：从 GM 搬运 blockLength 个元素到 UB
        DataCopy(xLocal, xGm[offset], blockLength);
        
        // 放入队列，通知 Compute 阶段“数据准备好了”
        inQueueX.EnQue(xLocal);
    }

    // 4. 核心计算：在 UB 中使用 NPU 向量单元计算
    __aicore__ inline void Compute(int32_t progress)
    {
        // 从输入队列取出数据
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        // 为输出结果申请 UB 内存
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        
        // 调用 Ascend C 内置的高效 Tanh 硬件指令
        AscendC::Tanh(yLocal, xLocal, blockLength);
        
        // 将结果放入输出队列
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        
        // 【极其重要】释放输入内存！这样下一轮循环的 CopyIn 才能复用这块 UB 空间，避免内存泄漏
        inQueueX.FreeTensor(xLocal);
    }

    // 5. 数据搬出：从快速 UB 搬回慢速 GM 保存
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        uint32_t offset = (progress / BUFFER_NUM) * blockLength;
        
        // 将计算结果写回 GM 的对应位置
        DataCopy(yGm[offset], yLocal, blockLength);
        
        // 释放输出内存，供下一次 Compute 循环复用
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0, tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

// 6. Kernel 入口函数：NPU 真正执行的起点
extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    // 注册并获取 Host 端传下来的 Tiling 计划书
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    
    KernelTanh kernel;
    // 初始化并启动流水线
    kernel.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    kernel.Process();
}