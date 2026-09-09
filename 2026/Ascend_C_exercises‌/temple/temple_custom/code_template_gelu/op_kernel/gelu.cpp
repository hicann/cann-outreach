// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

template <class DT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, ...) {
    	// TODO: 考生自行补齐

    }
    __aicore__ inline void Process() {
    	// TODO: 考生自行补齐

    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // TODO: 考生自行补齐
    }
    __aicore__ inline void Compute(int32_t progress)
    {
    	// TODO: 考生自行补齐
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // TODO: 考生自行补齐
    }

private:
    // Todo: 考生自行补齐
};

template <typename DT_X>
 __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_X> op;
    op.Init(input_x, output, ...); // TODO: 考生自行补齐
    op.Process();
}
