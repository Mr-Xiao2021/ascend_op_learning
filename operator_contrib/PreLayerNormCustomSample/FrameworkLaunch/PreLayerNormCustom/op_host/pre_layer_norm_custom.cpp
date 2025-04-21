
#include "pre_layer_norm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 48;

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  PreLayerNormCustomTilingData tiling;
  const float* epsilonAttr = context->GetAttrs()->GetAttrPointer<float>(0);
  tiling.set_epsilon(*epsilonAttr);
  uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
  gert::Shape oriShape = context->GetInputShape(0)->GetOriginShape();
  context->SetBlockDim(BLOCK_DIM);
  uint32_t lastDim = oriShape.GetDim(oriShape.GetDimNum() - 1);
  tiling.set_lastDim(lastDim);
  if (lastDim == 2048) {
    tiling.set_tileNum(415);
  } else {
    tiling.set_tileNum(43);
  }
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetTilingKey(1);
  size_t usrSize = 512 * 4 * 20480 * 1024;
  size_t sysWorksapceSize = 16 * 1024 * 1024;
  size_t* currentWorkSpace = context->GetWorkspaceSizes(1);
  currentWorkSpace[0] = usrSize + sysWorksapceSize;
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* x1_shape = context->GetInputShape(0);
  gert::Shape* y_shape = context->GetOutputShape(0);
  *y_shape = *x1_shape;
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class PreLayerNormCustom : public OpDef {
 public:
  explicit PreLayerNormCustom(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("gamma")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("beta")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("res_out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("epsilon").Float();

    this->SetInferShape(ge::InferShape);

    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(PreLayerNormCustom);
}  // namespace ops
