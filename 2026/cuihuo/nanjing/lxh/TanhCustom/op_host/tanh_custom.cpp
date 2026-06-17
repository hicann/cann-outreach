/*
 * -----------------------------------------------------------------------------
 * 文件说明：
 *   TanhCustom 算子的 Host 侧实现。
 *   主要负责：
 *     1. Tiling 函数 —— 根据输入 shape 计算分块策略，将参数下发给 Kernel
 *     2. InferShape   —— 推导输出 shape（输出与输入同 shape）
 *     3. InferDataType—— 推导输出数据类型（输出与输入同 dtype）
 *     4. OpDef 注册  —— 将算子注册到 CANN 框架中
 *
 * 算子功能：y = tanh(x)，逐元素（element-wise）激活函数
 * 支持平台：ascend910b（Ascend 910B AI 处理器）
 * 支持类型：DT_FLOAT16（半精度浮点）
 *
 * 架构说明：
 *   Host 侧（本文件）—— 运行在 CPU 上，负责编译期 shape/dtype 推导和 Tiling 策略计算
 *   Kernel 侧（op_kernel/tanh_custom.cpp）—— 运行在 AI Core 上，负责实际计算
 *   Tiling 数据（op_kernel/tanh_custom_tiling.h）—— Host 与 Kernel 共享的参数结构体
 *
 * 数据流：
 *   输入 x (DT_FLOAT16) → Kernel Tanh 计算 → 输出 y (DT_FLOAT16, 与 x 同 shape)
 * -----------------------------------------------------------------------------
 */

#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {

// ---------------------------------------------------------------------------
// Tiling 参数常量
// ---------------------------------------------------------------------------
// BLOCK_DIM: 使用的 AI Core 数量，每个 Core 独立处理一块数据
//            CANN 框架会根据 BlockDim 将输入数据自动切分为对应份数，
//            每个 AI Core 拿到自己的起始地址和长度后并行执行
// TILE_NUM:  每个 Core 内部的分块（Tile）数量，用于双缓冲流水线
//            配合 Kernel 侧的 BUFFER_NUM=2，每个 Core 循环 tileNum * 2 次，
//            实现数据搬运和计算的重叠（乒乓缓冲）
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM  = 8;

/*
 * ---------------------------------------------------------------------------
 * TilingFunc —— Tiling 函数（切分策略计算）
 * ---------------------------------------------------------------------------
 *
 * 函数作用：
 *   在算子图编译阶段，CANN 框架调用此函数，根据输入张量的 shape 信息
 *   计算出最优的数据切分策略（Tiling 参数），并将参数写入 TilingData 结构体，
 *   随 Kernel 二进制一同下发给每个 AI Core。
 *
 * 调用时机：
 *   图编译阶段（Graph Compile），由 CANN 框架自动调用。
 *   每次输入 shape 变化时可能重新触发 Tiling。
 *
 * Tiling 策略详解（针对 Tanh 这类 element-wise 算子）：
 *
 *   假设输入 x 的 shape 为 [2, 1024]，则 totalLength = 2 * 1024 = 2048
 *
 *   第 1 层切分（跨 Core）：
 *     - 将 2048 个元素均匀分给 BLOCK_DIM=8 个 AI Core
 *     - 每个 Core 大约处理 totalLength/BLOCK_DIM = 256 个元素
 *     - 通过 SetBlockDim(8) 告知硬件启动 8 个 Core
 *
 *   第 2 层切分（Core 内部 Tile）：
 *     - 每个 Core 的 256 个元素再切成 TILE_NUM=8 个 Tile
 *     - 每个 Tile 大约处理 256/8 = 32 个元素
 *     - 配合双缓冲（BUFFER_NUM=2），Tile 0 CopyIn 时 Tile 1 可以 Compute
 *     - 这种流水线并行隐藏了数据搬运的延迟
 *
 *   ┌──────────────────────────────────────────────────┐
 *   │  输入数据 (totalLength = 2048)                     │
 *   ├────────┬────────┬────────┬────────┬───┬───────────┤
 *   │ Core 0 │ Core 1 │ Core 2 │ Core 3 │...│  Core 7   │  ← BLOCK_DIM=8
 *   ├──┬──┬──┼──┬──┬──┼──┬──┬──┼──┬──┬──┼───┼──┬──┬─────┤
 *   │T0│T1│..│T0│T1│..│T0│T1│..│T0│T1│..│...│T0│T1│..   │  ← TILE_NUM=8 per Core
 *   └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴───┴──┴──┴─────┘
 *
 * 参数：
 *   context: gert::TilingContext* 指针，提供以下关键方法：
 *     - GetInputShape(idx)    → 获取指定输入索引的 StorageShape
 *     - GetOutputShape(idx)   → 获取指定输出索引的 StorageShape
 *     - SetBlockDim(dim)      → 设置启动的 AI Core 数量
 *     - GetTilingData<T>()    → 获取可写的 TilingData 结构体指针
 *     - SetNeedAtomic(bool)   → 设置是否需要原子清理（Atomic Clean）
 *     - SetTilingKey(key)     → 设置 Tiling 键值
 *
 * 返回值：
 *   ge::GRAPH_SUCCESS:  Tiling 计算成功，参数已写入 Context
 *   ge::GRAPH_FAILED:   Tiling 计算失败（如获取 shape 失败、tiling data 为空等）
 * ---------------------------------------------------------------------------
 */
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // =======================================================================
    // 步骤 1: 获取输入张量的 Shape 信息
    // =======================================================================
    // context->GetInputShape(0) 获取第 0 号输入（即算子定义中的 "x"）的 StorageShape
    //
    // StorageShape 是一个包含两套 shape 的结构体：
    //   - GetOriginShape():  用户指定的逻辑 shape，例如 [batch, channel, height, width]
    //   - GetStorageShape(): 物理存储 shape，当没有 padding 时与 origin shape 相同
    //
    // 对于 Tanh 这种纯逐元素算子，不需要关心存储布局，直接使用 OriginShape 即可
    const gert::StorageShape* x_storage_shape = context->GetInputShape(0);
    if (x_storage_shape == nullptr) {
        // 获取失败：可能是输入索引越界或 Context 状态异常
        return ge::GRAPH_FAILED;
    }

    // 获取原始（逻辑）shape
    const gert::Shape& x_shape = x_storage_shape->GetOriginShape();

    // =======================================================================
    // 步骤 2: 计算输入张量的总元素个数（totalLength）
    // =======================================================================
    // Shape 提供两个关键方法：
    //   - GetDimNum(): 返回维度数量，例如 shape=[2,3,4] 返回 3
    //   - GetDim(i):   返回第 i 维的大小（int64_t 类型），i 从 0 开始
    //
    // 总元素个数 = 所有维度的乘积
    // 例如 shape=[2,3,4] → totalLength = 2 × 3 × 4 = 24
    //
    // 标量（scalar）场景：GetDimNum() == 0，totalLength 保持为 1
    uint32_t totalLength = 1;
    for (size_t i = 0; i < x_shape.GetDimNum(); ++i) {
        totalLength *= static_cast<uint32_t>(x_shape.GetDim(i));
    }

    // =======================================================================
    // 步骤 3: 设置 BlockDim（AI Core 使用数量）
    // =======================================================================
    // SetBlockDim 告知硬件调度器需要启动多少个 AI Core 来并行执行此算子
    //
    // 硬件会根据 BlockDim 自动完成以下工作：
    //   - 将输入数据均匀切分为 BlockDim 份
    //   - 为每个 AI Core 分配一段连续的数据区间
    //   - 每个 AI Core 拿到自己对应的起始地址偏移 + 处理长度
    //
    // BlockDim 的选择策略：
    //   - 太小：AI Core 利用率低，并行度不足
    //   - 太大：每个 Core 处理的数据太少，启动开销占比过高
    //   - 通常根据数据量和 AI Core 总数折中，这里固定为 8
    auto ret = context->SetBlockDim(BLOCK_DIM);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // =======================================================================
    // 步骤 4: Tiling 参数总结
    // =======================================================================
    // 这里只需计算 totalLength 并下发给 Kernel。Kernel 侧的 Init 会自行完成：
    //   - blockLength = totalLength / GetBlockNum()   // 每 Core 元素数
    //   - tileLength  = blockLength / tileNum / BUFFER_NUM  // 每迭代元素数
    //
    // 这意味着 Host 侧不需要关心 per-core 细节，只要把总量和 tile 数告诉 Kernel 即可。
    // 硬件会根据 SetBlockDim(BLOCK_DIM) 自动调度 BLOCK_DIM 个 Core 并行执行。

    // =======================================================================
    // 步骤 5: 获取 TilingData 指针并填充参数
    // =======================================================================
    // context->GetTilingData<T>() 是一个模板方法，返回 T* 类型指针
    //
    // 它的工作原理：
    //   1. 在内部分配一块大小为 sizeof(T) 的内存
    //   2. 返回这块内存的可写指针
    //   3. 编译后，这块数据被嵌入 Kernel 二进制，随 Kernel 下发给 AI Core
    //
    // Kernel 侧通过以下宏获取同一份数据：
    //   GET_TILING_DATA(tilingData, tiling);  // tilingData 即为 TanhCustomTilingData 类型
    //
    // TanhCustomTilingData 结构体定义在 op_kernel/tanh_custom_tiling.h 中：
    //   - totalLength: 总元素个数（所有 Core 合计）
    //   - tileNum:     每个 Core 的 Tile 数量
    //
    // 注意：这里 totalLength 填的是"总元素数"而非 per-core 值，
    //       Kernel 侧的 Init 会通过 GetBlockNum() 自行除以 Core 数量
    auto* tiling_data = context->GetTilingData<TanhCustomTilingData>();
    if (tiling_data == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 填充 TilingData —— 这些字段在 Kernel 侧通过 GET_TILING_DATA 读取
    tiling_data->totalLength = totalLength;
    tiling_data->tileNum     = TILE_NUM;

    // =======================================================================
    // 步骤 6: 配置 Atomic Clean（原子清理）
    // =======================================================================
    // SetNeedAtomic 用于控制 AI Core 的 L1 Buffer 是否需要做原子清理
    //
    // 需要设置为 true 的场景：
    //   - Kernel 中有跨 Core 的原子操作（如 ReduceScatter、AllReduce 等）
    //   - 多个 Core 同时写同一块输出地址
    //
    // 不需要的场景：
    //   - Tanh 是纯逐元素算子，每个 Core 写独立不重叠的输出区域
    //   - 不存在跨 Core 的数据竞争
    //
    // 因此这里设置为 false，可以减少不必要的清理开销
    context->SetNeedAtomic(false);

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {

/*
 * ---------------------------------------------------------------------------
 * InferShape —— 输出 Shape 推导函数
 * ---------------------------------------------------------------------------
 * 对于 Tanh 这种 element-wise 算子，输出 shape 与输入 shape 完全一致。
 * 直接将输入 shape 拷贝给输出即可。
 *
 * 参数：
 *   context: InferShapeContext*，提供 GetInputShape 和 GetOutputShape 接口
 * 返回：
 *   GRAPH_SUCCESS: 推导成功
 * ---------------------------------------------------------------------------
 */
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    // 获取第 0 号输入（x）的 shape
    const gert::Shape* x1_shape = context->GetInputShape(0);
    // 获取第 0 号输出（y）的可写 shape 指针
    gert::Shape* y_shape = context->GetOutputShape(0);
    // 逐元素操作：输出 shape = 输入 shape
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

/*
 * ---------------------------------------------------------------------------
 * InferDataType —— 输出数据类型推导函数
 * ---------------------------------------------------------------------------
 * Tanh 算子的输出类型与输入类型一致。
 * 例如输入是 DT_FLOAT16，输出也必须是 DT_FLOAT16。
 *
 * 参数：
 *   context: InferDataTypeContext*，提供 GetInputDataType 和 SetOutputDataType 接口
 * 返回：
 *   GRAPH_SUCCESS: 推导成功
 * ---------------------------------------------------------------------------
 */
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // 获取第 0 号输入的数据类型
    const auto inputDataType = context->GetInputDataType(0);
    // 将输出类型设置为与输入相同
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {

/*
 * ---------------------------------------------------------------------------
 * TanhCustom —— 算子定义类
 * ---------------------------------------------------------------------------
 * 继承自 OpDef 基类，在构造函数中完成以下注册：
 *   1. 输入/输出规格定义（名称、是否必选、支持的数据类型、Format）
 *   2. Shape 推导函数注册（InferShape）
 *   3. 数据类型推导函数注册（InferDataType）
 *   4. AI Core 配置（Tiling 函数、目标平台）
 *
 * 输入/输出属性说明：
 *   - ParamType(REQUIRED):       此输入/输出为必选
 *   - DataType({DT_FLOAT16}):    支持 float16 数据类型
 *   - Format({FORMAT_ND}):       N 维通用格式（不限定具体维度数）
 *   - UnknownShapeFormat({FORMAT_ND}): 动态 shape 场景下的格式
 *
 * AI Core 配置说明：
 *   - SetTiling(optiling::TilingFunc): 绑定 Tiling 函数
 *   - AddConfig("ascend910b"):         指定目标硬件平台为 Ascend 910B
 * ---------------------------------------------------------------------------
 */
class TanhCustom : public OpDef {
public:
    explicit TanhCustom(const char* name) : OpDef(name)
    {
        // ---- 定义输入 "x" ----
        this->Input("x")
            .ParamType(REQUIRED)                         // 必选输入
            .DataType({ge::DT_FLOAT16})                  // 仅支持 float16
            .Format({ge::FORMAT_ND})                     // N 维通用格式
            .UnknownShapeFormat({ge::FORMAT_ND});        // 动态 shape 也使用 ND 格式

        // ---- 定义输出 "y" ----
        this->Output("y")
            .ParamType(REQUIRED)                         // 必选输出
            .DataType({ge::DT_FLOAT16})                  // 输出类型与输入一致
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // ---- 注册 Shape 推导和数据类型推导函数 ----
        // 这两个函数在图编译阶段被调用，用于静态/动态 shape 场景
        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        // ---- 配置 AI Core 运行参数 ----
        // SetTiling: 绑定 Tiling 函数，框架会在编译期调用它
        // AddConfig: 声明目标硬件平台（Ascend 910B 系列）
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

/*
 * ---------------------------------------------------------------------------
 * OP_ADD 宏 —— 算子注册
 * ---------------------------------------------------------------------------
 * 将 TanhCustom 算子注册到 CANN 算子库中。
 *
 * 根据编译目标不同，该宏展开为不同的注册逻辑：
 *   - OP_PROTO_LIB:  注册 Shape/DataType 推导函数（用于图编译）
 *   - OP_TILING_LIB: 注册 Tiling 函数（生成 Tiling 策略库）
 *   - 默认:          注册到 OpDefFactory（算子定义工厂）
 *
 * 注册后，框架可通过 "TanhCustom" 名称查找到该算子的全部元信息。
 * ---------------------------------------------------------------------------
 */
OP_ADD(TanhCustom);
}
