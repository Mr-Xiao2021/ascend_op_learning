
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#include "scatter_sub_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <assert.h>

constexpr uint32_t BLOCK_SIZE = 32;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ScatterSubTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    auto shape = context->GetInputShape(0)->GetOriginShape();
    uint32_t var1stDim = shape.GetDim(0);
    uint32_t lastDim = totalLength / var1stDim;

    // 根据CoerNum设置BlockDim
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNum();
    assert(coreNum != 0);
    // 当前代码仅能运行在单核环境
    coreNum = 1;    
    context->SetBlockDim(coreNum);

    uint64_t ub_size, l1_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    
    // 获取输入的数据类型，计算字节数
    auto dataType = context->GetInputDesc(0)->GetDataType();
    auto sizeOfDataType = 0;
    switch(dataType) {
    case ge::DataType::DT_FLOAT:
        sizeOfDataType = 4;
        break;
    case ge::DataType::DT_FLOAT16:
        sizeOfDataType = 2;
        break;
    case ge::DataType::DT_INT32:
        sizeOfDataType = 4;
        break;
    case ge::DataType::DT_INT8:
        sizeOfDataType = 1;
        break;
    case ge::DataType::DT_BF16:
        sizeOfDataType = 2;
        break;
    default:
        break;
    }

    assert(sizeOfDataType != 0);
   
    // shape对齐的最小段位
    auto alignNum = BLOCK_SIZE / sizeOfDataType;
    uint32_t indicesLength = context->GetInputShape(1)->GetStorageShape().GetShapeSize();

    // 最后一维是对其的情况处理
    uint32_t firstTiling = lastDim;
    if (lastDim * sizeOfDataType % BLOCK_SIZE == 0) {
        // bufferSize是初始化的Buffer数量
        uint32_t bufferSize = 10;

        // Ub大小如果大于Buffer区域大小则直接按照lastDim分块
        // 否则，则按照32B依次减小，知道满足Ub大小
        if (ub_size > bufferSize * lastDim * sizeOfDataType) {
            firstTiling = lastDim;
        } else {
            while(ub_size < bufferSize * firstTiling * sizeOfDataType) {
                firstTiling -= 32 / sizeOfDataType;
            }
        }
    }

    // 最后一维非对其则设置tilingKey从20开始，且firstTiling 失效
    if (lastDim * sizeOfDataType % BLOCK_SIZE != 0) {
        firstTiling = 0;
    }

    // 设置tiling
    // printf("alignNum is %d\n", alignNum);
    // printf("lastDim is %d\n", lastDim);
    // printf("indicesLength is %d\n", indicesLength);
    // printf("var1stDim is %d\n", var1stDim);
    // printf("firstTiling is %d\n", firstTiling);
    tiling.set_alignNum(alignNum);
    tiling.set_lastDim(lastDim);
    tiling.set_indicesLength(indicesLength);
    tiling.set_var1stDim(var1stDim);
    tiling.set_firstTiling(firstTiling);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    // const gert::Shape* x1_shape = context->GetInputShape(0);
    // gert::Shape* y_shape = context->GetOutputShape(0);
    // *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class ScatterSub : public OpDef {
public:
    explicit ScatterSub(const char* name) : OpDef(name)
    {
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("updates")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("use_locking").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b")
                      .AddConfig("ascend910b");

    }
};

OP_ADD(ScatterSub);
}
