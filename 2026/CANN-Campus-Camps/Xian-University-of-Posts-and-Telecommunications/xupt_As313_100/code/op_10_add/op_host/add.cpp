// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/add_tiling.h"
#include "../op_kernel/tiling_key_add.h"

namespace optiling {
constexpr uint32_t ALIGN_BYTES = 512;      // MTE高效搬运的对齐粒度(GM地址与长度均按此对齐)
constexpr uint32_t MIN_ALIGN_BYTES = 32;   // GM搬运的最小对齐要求
constexpr uint32_t BUFFER_NUM = 2;         // 与kernel侧保持一致(双buffer)
constexpr uint32_t TENSOR_NUM = 3;         // x / y / z 三份UB空间
constexpr uint32_t UB_SAFETY_RATIO = 4;    // UB预留1/4余量, 防止超出容量
constexpr uint32_t DEFAULT_UB_ELEM = 8192;

static inline uint32_t AlignUp(uint32_t value, uint32_t align) {
    if (align == 0) {
        return value;
    }
    return (value + align - 1) / align * align;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    // 1. 获取平台信息: AI Core数量与UB大小
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t num_cores_aiv = platform.GetCoreNumAiv();
    uint64_t ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    // 2. 获取输入信息
    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
    if (tensor_x == nullptr || tensor_y == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType dtype_x = tensor_x->GetDataType();                               // 数据类型
    uint32_t dtype_size = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_x));  // 数据类型字长
    if (dtype_size == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t total_length = static_cast<uint32_t>(tensor_x->GetShapeSize());      // 元素个数
    if (total_length == 0) {
        return ge::GRAPH_FAILED;
    }

    // 3. 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 4. 切分粒度: 优先512B, 若不能整除则逐级减半, 保证各核完全均衡、且每段都32B对齐
    uint32_t elem_per_block = MIN_ALIGN_BYTES / dtype_size;  // fp32: 8, fp16: 16
    if (elem_per_block == 0) {
        elem_per_block = 1;
    }
    uint32_t align_elem = ALIGN_BYTES / dtype_size;          // fp32: 128, fp16: 256
    if (align_elem < elem_per_block) {
        align_elem = elem_per_block;
    }
    while (align_elem > elem_per_block && total_length % align_elem != 0) {
        align_elem >>= 1;
    }
    uint32_t units = total_length / align_elem;  // 总的切分单元数
    if (units == 0) {
        units = 1;
    }

    // 5. 核间切分: 单元级负载均衡, 前formerNum个核多分一个单元
    uint32_t core_num = (num_cores_aiv == 0) ? 1 : num_cores_aiv;
    if (core_num > units) {
        core_num = units;  // 不启动空核
    }
    uint32_t unit_per_core = units / core_num;
    uint32_t former_num = units % core_num;                     // 处理大块的核数
    uint32_t former_length = (unit_per_core + 1) * align_elem;  // 大块长度
    uint32_t tail_length = unit_per_core * align_elem;          // 小块长度

    // 6. 核内切分: 由UB容量推导单次可搬运/计算的元素个数, 尽量一次搬完
    uint64_t ub_elem = DEFAULT_UB_ELEM;
    if (ub_size > 0) {
        ub_size = ub_size / UB_SAFETY_RATIO * (UB_SAFETY_RATIO - 1);  // 预留1/4余量
        ub_elem = ub_size / (static_cast<uint64_t>(dtype_size) * TENSOR_NUM * BUFFER_NUM);
    }
    if (ub_elem == 0) {
        ub_elem = align_elem;
    }
    if (ub_elem > former_length) {
        ub_elem = former_length;
    }
    uint32_t tile_length = AlignUp(static_cast<uint32_t>(ub_elem), align_elem);
    if (tile_length > former_length) {
        tile_length = former_length;
    }
    if (tile_length < align_elem) {
        tile_length = align_elem;
    }

    // 7. 写回tiling结构体
    AddTilingData *tiling = context->GetTilingData<AddTilingData>();
    tiling->totalLength = total_length;
    tiling->formerNum = former_num;
    tiling->formerLength = former_length;
    tiling->tailLength = tail_length;
    tiling->tileLength = tile_length;
    tiling->blockDim = core_num;

    // 配置启动核数
    context->SetBlockDim(core_num);
    // 配置workspace大小
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // 输出z的shape与输入x一致
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *z_shape = context->GetOutputShape(0);
        if (x_shape == nullptr || z_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        z_shape->SetDimNum(x_shape->GetDimNum());
        for (size_t i = 0; i < x_shape->GetDimNum(); i++) {
            z_shape->SetDim(i, x_shape->GetDim(i));
        }
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // 输出z的数据类型与输入x一致
        ge::DataType x_dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, x_dtype);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Add : public OpDef {
    public:
        explicit Add(const char *name) : OpDef(name) {
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
    OP_ADD(Add);
}  // namespace ops
