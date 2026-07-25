#include "sub_custom_template.h"

extern "C" __global__ __aicore__ void sub_custom_template(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(SubCustomTemplateTilingData, tiling_data, tiling);
        SubCustomTemplateVector<DTYPE_X> op;
        op.Init(x, y, z, &tiling_data, &pipe);
        op.Process();
    }
}
