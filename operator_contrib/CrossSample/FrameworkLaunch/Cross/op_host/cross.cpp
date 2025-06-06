/*
* @author: 孙明志
* @mail: 531483935@qq.com
* @date: 2024-05-27
*/

#include "cross_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    CrossTilingData tiling;
    int64_t numshapes = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    tiling.set_numshapes(numshapes);
    int64_t shape[128];
    for (int k = 0; k < 2; ++k) {
        int64_t *ss = &shape[k * 64];
        const gert::StorageShape* shape = context->GetInputShape(k);
        for (int i = 0; i < shape->GetStorageShape().GetDimNum(); i++) {
            ss[i] = shape->GetStorageShape().GetDim(i);
        }
    }
    tiling.set_shape(shape);
    int64_t dim = *context->GetAttrs()->GetInt(0);
    if (dim < 0) {
        dim = numshapes + dim;
    }
    if (dim < 0) {
        for (int i = 0; i < numshapes; ++i) {
            if (shape[i] == 3 && shape[i + 64] == 3) {
                dim = i;
                break;
            }
        }
    }
    tiling.set_dim(dim);

    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;

}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Cross : public OpDef {
public:
    explicit Cross(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(-65530);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b")
                      .AddConfig("ascend910b");

    }
};

OP_ADD(Cross);
}
