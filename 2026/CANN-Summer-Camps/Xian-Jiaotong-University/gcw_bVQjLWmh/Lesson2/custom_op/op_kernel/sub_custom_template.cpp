#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    uint32_t totalElem = tilingData->size;

    GlobalTensor<float> xGm(x, totalElem);
    GlobalTensor<float> yGm(y, totalElem);
    GlobalTensor<float> zGm(z, totalElem);

    LocalTensor<float> xLm;
    LocalTensor<float> yLm;
    LocalTensor<float> zLm;

    xLm.Init(totalElem);
    yLm.Init(totalElem);
    zLm.Init(totalElem);

    DataCopy(xLm, xGm, totalElem);
    DataCopy(yLm, yGm, totalElem);

    Sub(zLm, xLm, yLm, totalElem);

    DataCopy(zGm, zLm, totalElem);
}
