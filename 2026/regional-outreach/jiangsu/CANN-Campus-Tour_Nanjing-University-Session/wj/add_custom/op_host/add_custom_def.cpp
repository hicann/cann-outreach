/*!
 * \file add_custom_def.cpp
 * \brief add_custom 算子的定义
 *
 * 本文件定义 add_custom 算子的接口，包括输入输出规格、支持的数据类型、
 * 格式以及 AI Core 编译配置。算子定义是算子在 CANN 框架中注册和验证所必需的。
 */
#include "register/op_def_registry.h"

namespace ops {
/*!
 * \brief add_custom 算子类定义
 *
 * 该类定义执行逐元素加法的 add_custom 算子，接收两个输入张量并产生一个相同形状的输出张量。
 *
 * 支持的数据类型: FLOAT16
 * 支持的格式: ND（n 维格式，逻辑 2 维 [N2, N1]）
 */
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char* name) : OpDef(name)
    {
        // 定义输入 x 的规格
        this->Input("x")                                        // 输入 x 名称定义
            .ParamType(REQUIRED)                                // 必选输入参数
            .DataType({ge::DT_FLOAT16})                         // 支持的数据类型：float16
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})             // 支持的格式：n 维格式
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}) // 未确定大小 shape 对应的 format 格式
            .AutoContiguous();                                  // 内存自动连续化
        // 定义输入 y 的规格
        this->Input("y") // 输入 y 名称定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // 定义输出 z 的规格
        this->Output("z") // 输出 z 名称定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // AI Core 编译配置
        OpAICoreConfig aicoreConfig;
        aicoreConfig
            .DynamicCompileStaticFlag(true)                // 启用静态动态编译标志
            .DynamicFormatFlag(false)                      // 禁用动态格式标志
            .DynamicRankSupportFlag(true)                  // 启用动态 rank 支持
            .DynamicShapeSupportFlag(true)                 // 启用动态 shape 支持
            .NeedCheckSupportFlag(false)                   // 禁用检查支持标志
            .PrecisionReduceFlag(true)                     // 启用精度降低标志
            .ExtendCfgInfo("opFile.value", "add_custom"); // 指定的 kernel 入口文件名
        // 为不同 SOC 版本添加 AI Core 配置（按实际硬件保留对应条目）
        this->AICore().AddConfig("ascend910b", aicoreConfig);   // Ascend 910B 芯片配置
        this->AICore().AddConfig("ascend910_93", aicoreConfig); // Ascend 910A 芯片配置
        this->AICore().AddConfig("ascend950", aicoreConfig);    // Ascend 950 芯片配置
    }
};

// 将 AddCustom 算子添加到算子库中
OP_ADD(AddCustom);
} // namespace ops
