#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"
using namespace AscendC;

// 双缓冲配置：2个缓存槽实现乒乓流水线，隐藏搬运延迟
constexpr uint32_t BUFFER_NUM = 2;

// 模板内核类：一套代码同时支持 float16(half) 和 float32(float)
template<typename T>
class SubCustomTemplateKernel
{
public:
    __aicore__ inline SubCustomTemplateKernel() {}

    // 初始化：绑定全局内存地址，初始化分块参数
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const SubCustomTemplateTilingData& tiling)
    {
        // 绑定输入输出的全局内存（DDR显存）地址与长度
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), tiling.totalLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), tiling.totalLength);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(z), tiling.totalLength);

        // 保存分块参数，计算单块元素个数
        totalLength = tiling.totalLength;
        // 防护：tileNum 为 0 时默认设为 1，避免除零
        tileNum = (tiling.tileNum == 0) ? 1 : tiling.tileNum;
        tileSize = totalLength / tileNum;
        // 最后一块的实际大小，处理非整除时的尾部数据，避免丢失
        lastTileSize = totalLength - tileSize * (tileNum - 1);
    }

    // 主流程：双缓冲流水线执行
    __aicore__ inline void Process()
    {
        // 预取第0块数据到片上缓存，启动流水线
        CopyIn(0);

        // 循环：搬运下一块数据 与 计算当前块 并行执行
        for (int32_t i = 1; i < tileNum; ++i) {
            CopyIn(i);     // 预取下一块输入（占用另一个缓存槽）
            Compute(i-1);  // 计算上一块数据
            CopyOut(i-1);  // 把上一块结果写回全局内存
        }

        // 处理最后一块的计算与输出
        Compute(tileNum - 1);
        CopyOut(tileNum - 1);
    }

private:
    // 数据搬入：全局内存 → 片上高速缓存
    __aicore__ inline void CopyIn(int32_t progress)
    {
        int32_t bufIdx = progress % BUFFER_NUM;
        uint32_t offset = progress * tileSize;
        // 最后一块使用实际剩余长度，避免尾部数据丢失
        uint32_t curSize = (progress == tileNum - 1) ? lastTileSize : tileSize;

        // 批量搬运两个输入的当前分块
        DataCopy(xLocal[bufIdx], xGm[offset], curSize);
        DataCopy(yLocal[bufIdx], yGm[offset], curSize);
    }

    // 核心计算：逐元素矢量减法 z = x - y
    __aicore__ inline void Compute(int32_t progress)
    {
        int32_t bufIdx = progress % BUFFER_NUM;
        // 最后一块使用实际剩余长度，避免尾部数据丢失
        uint32_t curSize = (progress == tileNum - 1) ? lastTileSize : tileSize;
        // Ascend C硬件级矢量减法指令，一次性计算一整块数据
        Sub(zLocal[bufIdx], xLocal[bufIdx], yLocal[bufIdx], curSize);
    }

    // 数据搬出：片上缓存 → 全局内存
    __aicore__ inline void CopyOut(int32_t progress)
    {
        int32_t bufIdx = progress % BUFFER_NUM;
        uint32_t offset = progress * tileSize;
        // 最后一块使用实际剩余长度，避免尾部数据丢失
        uint32_t curSize = (progress == tileNum - 1) ? lastTileSize : tileSize;

        DataCopy(zGm[offset], zLocal[bufIdx], curSize);
    }

private:
    // 全局内存张量（DDR显存，容量大、速度慢）
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;

    // 片上本地缓存张量（UB高速缓存，容量小、速度极快，双缓冲数组）
    LocalTensor<T> xLocal[BUFFER_NUM];
    LocalTensor<T> yLocal[BUFFER_NUM];
    LocalTensor<T> zLocal[BUFFER_NUM];

    uint32_t totalLength; // 总元素数
    uint32_t tileNum;     // 总分块数
    uint32_t tileSize;    // 单个分块的元素数
    uint32_t lastTileSize; // 最后一块的实际元素数（处理非整除边界）
};

// ------------------------------
// 核函数入口：框架调用算子的入口，分别对应两种数据类型
// ------------------------------

// float32 版本入口
extern "C" __global__ __aicore__ void sub_custom_template_float(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, SubCustomTemplateTilingData tiling)
{
    SubCustomTemplateKernel<float> kernel;
    kernel.Init(x, y, z, tiling);
    kernel.Process();
}

// float16 版本入口
extern "C" __global__ __aicore__ void sub_custom_template_half(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, SubCustomTemplateTilingData tiling)
{
    SubCustomTemplateKernel<half> kernel;
    kernel.Init(x, y, z, tiling);
    kernel.Process();
}