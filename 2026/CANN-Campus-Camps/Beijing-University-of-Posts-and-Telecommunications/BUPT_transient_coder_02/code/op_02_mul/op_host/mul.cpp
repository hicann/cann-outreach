// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        // 获取硬件平台信息（AIV核数与UB大小）
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        if (num_cores_aiv <= 0) {
            num_cores_aiv = 8;
        }

        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        if (ub_size <= 0) {
            ub_size = 256 * 1024;
        }

        // 获取算子输入数组信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        if (tensor_x == nullptr) {
            return ge::GRAPH_FAILED;
        }
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        int dtype_size_x = ge::GetSizeByDataType(dtype_x);
        if (dtype_size_x <= 0) {
            dtype_size_x = (dtype_x == ge::DT_FLOAT16) ? 2 : 4;
        }
        // 32字节对齐要求的元素数 (fp32 为 8, fp16 为 16)
        uint32_t align_num = 32 / dtype_size_x;

        const gert::StorageShape* x1_shape = context->GetInputShape(0);
        int32_t totalLength = 1;
        if (x1_shape != nullptr) {
            for (int32_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
                totalLength *= x1_shape->GetStorageShape().GetDim(i);
            }
        } else {
            totalLength = tensor_x->GetShapeSize();
        }

        if (totalLength <= 0) {
            return ge::GRAPH_FAILED;
        }

        // 向上取整计算总 32 字节 Block 数量
        uint32_t totalBlocks = (totalLength + align_num - 1) / align_num;

        // 启动核数：充分利用所有可用 AIV 核，但不超过总块数
        uint32_t coreNum = static_cast<uint32_t>(num_cores_aiv);
        if (totalBlocks < coreNum) {
            coreNum = totalBlocks;
        }
        if (coreNum <= 0) {
            coreNum = 1;
        }

        // 多核均衡切分逻辑（formerNum 核处理 formerBlocks，tailNum 核处理 tailBlocks）
        uint32_t formerBlocks = (totalBlocks + coreNum - 1) / coreNum;
        uint32_t tailBlocks = totalBlocks / coreNum;
        uint32_t formerNum = totalBlocks % coreNum;
        uint32_t tailNum = coreNum - formerNum;

        if (formerNum == 0) {
            formerNum = coreNum;
            formerBlocks = tailBlocks;
            tailNum = 0;
        }

        uint32_t formerLength = formerBlocks * align_num;
        uint32_t tailLength = tailBlocks * align_num;

        // 单 buffer 最大可用空间：UB 的 1/6 (3个队列 * 2个buffer)
        uint32_t maxTileBytes = static_cast<uint32_t>(ub_size / 6);
        uint32_t maxTileLength = maxTileBytes / dtype_size_x;
        maxTileLength = (maxTileLength / align_num) * align_num;
        if (maxTileLength == 0) {
            maxTileLength = align_num;
        }

        // 双缓冲流水线分块：使能 MTE2 搬入与 MTE3 搬出全重叠
        // 若单核数据量 >= 2048 元素，分 4 块流水；若 < 2048 元素，分 2 块流水
        uint32_t splitCount = (formerLength >= 2048) ? 4 : 2;
        uint32_t tileLength = ((formerLength / splitCount) + align_num - 1) / align_num * align_num;
        if (tileLength > maxTileLength) {
            tileLength = maxTileLength;
        }
        if (tileLength == 0) {
            tileLength = align_num;
        }

        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }

        tiling->formerNum = formerNum;
        tiling->formerLength = formerLength;
        tiling->tailNum = tailNum;
        tiling->tailLength = tailLength;
        tiling->tileLength = tileLength;
        tiling->alignNum = align_num;

        context->SetBlockDim(coreNum);

        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace != nullptr) {
            currentWorkspace[0] = 0;
        }
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        if (context == nullptr) {
            return GRAPH_FAILED;
        }
        // 获取第0号输入 x 的 shape
        const gert::Shape* x1_shape = context->GetInputShape(0);
        // 获取第0号输出 z 的 shape 可写指针
        gert::Shape* z_shape = context->GetOutputShape(0);
        if (x1_shape == nullptr || z_shape == nullptr) {
            return GRAPH_FAILED;
        }
        // 逐元素矢量乘算子：输出 shape 与输入 shape 完全一致
        *z_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        if (context == nullptr) {
            return GRAPH_FAILED;
        }
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Mul : public OpDef {
    public:
        explicit Mul(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("z")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Mul);
}  // namespace ops
