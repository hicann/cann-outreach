namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  AddCustomTemplateTilingData *tiling = context->GetTilingData<AddCustomTemplateTilingData>();
  uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
  
  // 1. 动态选择核心数：根据数据量选择，但不大于硬件最大核心数（Ascend 910B 为 32？实际为 8 个 AI Core，但可设置多 block，这里设为 8 或 16）
  const uint32_t MAX_CORES = 8;  // 910B 实际可用的 AI Core 数，可根据硬件调整
  uint32_t blockDim = (totalLength < 2048) ? 1 : MAX_CORES;
  context->SetBlockDim(blockDim);

  // 2. 每个 tile 处理固定元素数（可根据 L1/L2 缓存大小调优，推荐 2048 或 4096）
  const uint32_t TILE_SIZE = 2048;
  // 每个核心负责的数据量
  uint32_t perCoreLen = (totalLength + blockDim - 1) / blockDim;
  // 核心的 tile 数，向上取整，并确保至少为 1
  uint32_t tileNum = (perCoreLen + TILE_SIZE - 1) / TILE_SIZE;
  
  tiling->totalLength = totalLength;
  tiling->tileNum = tileNum;      // 每个核心的 tile 数（所有核心相同）
  return ge::GRAPH_SUCCESS;
}
}