#include "atanh_custom_tiling.h"

#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"

namespace optiling {
namespace {
constexpr uint32_t kMaxTileLength = 4096;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const gert::StorageShape* xShape = context->GetInputShape(0);
  if (xShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const int64_t count = xShape->GetStorageShape().GetShapeSize();
  if (count <= 0 || count > UINT32_MAX) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t total = static_cast<uint32_t>(count);
  const uint32_t tile = std::min(total, kMaxTileLength);
  const uint32_t tileNum = (total + tile - 1) / tile;

  AtanhCustomTilingData data;
  data.set_totalLength(total);
  data.set_tileLength(tile);
  data.set_tileNum(tileNum);
  data.set_lastTileLength(total - (tileNum - 1) * tile);
  data.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
  context->SetBlockDim(1);
  context->SetTilingKey(1);
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* x = context->GetInputShape(0);
  gert::Shape* y = context->GetOutputShape(0);
  if (x == nullptr || y == nullptr || x->GetDimNum() != 4) {
    return GRAPH_FAILED;
  }
  *y = *x;
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class AtanhCustom : public OpDef {
 public:
  explicit AtanhCustom(const char* name) : OpDef(name) {
    this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->SetInferShape(ge::InferShape);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};
OP_ADD(AtanhCustom);
}  // namespace ops
