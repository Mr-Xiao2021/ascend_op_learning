/**
* @file main.cpp
*
* Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <limits>

#include "acl/acl.h"
#include "op_runner.h"

#include "common.h"

bool g_isDevice = false;
int deviceId = 0;

OperatorDesc CreateOpDesc()
{
    // define operator
    std::vector<int64_t> shape {3, 4, 1024};
    std::vector<int64_t> out_shape {1};
    aclDataType dataType = ACL_FLOAT16;
    aclFormat format = ACL_FORMAT_ND;
    OperatorDesc opDesc;
    opDesc.pOptional= 3;
    opDesc.axesOptional = {};
    opDesc.keepdimOptional = false;
    opDesc.epsilonOptional = 1e-12;
    opDesc.AddInputTensorDesc(dataType, shape.size(), shape.data(), format);
    opDesc.AddOutputTensorDesc(dataType, out_shape.size(), out_shape.data(), format);
    return opDesc;
}

bool SetInputData(OpRunner &runner)
{
    size_t fileSize = 0;
    ReadFile("../input/input_x.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0));
    INFO_LOG("Set input success");
    return true;
}

bool ProcessOutputData(OpRunner &runner)
{
    WriteFile("../output/output.bin", runner.GetOutputBuffer<void>(0), runner.GetOutputSize(0));
    INFO_LOG("Write output success");
    return true;
}

void DestoryResource()
{
    bool flag = false;
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        flag = true;
    }
    INFO_LOG("Reset Device success");
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag) {
        ERROR_LOG("Destory resource failed");
    } else {
        INFO_LOG("Destory resource success");
    }
}

bool InitResource()
{
    std::string output = "../output";
    if (access(output.c_str(), 0) == -1) {
        int ret = mkdir(output.c_str(), 0700);
        if (ret == 0) {
            INFO_LOG("Make output directory successfully");
        }
        else {
            ERROR_LOG("Make output directory fail");
            return false;
        }
    }

    // acl.json is dump or profiling config file
    if (aclInit("../scripts/acl.json") != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. deviceId is %d", deviceId);
        (void)aclFinalize();
        return false;
    }
    INFO_LOG("Set device[%d] success", deviceId);

    // runMode is ACL_HOST which represents app is running in host
    // runMode is ACL_DEVICE which represents app is running in device
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestoryResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);

    return true;
}

bool RunOp()
{
    // create op desc
    OperatorDesc opDesc = CreateOpDesc();

    // create Runner
    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    // Load inputs
    if (!SetInputData(opRunner)) {
        ERROR_LOG("Set input data failed");
        return false;
    }

    // Run op
    if (!opRunner.RunOp()) {
        ERROR_LOG("Run op failed");
        return false;
    }

    // process output data
    if (!ProcessOutputData(opRunner)) {
        ERROR_LOG("Process output data failed");
        return false;
    }

    INFO_LOG("Run op success");
    return true;
}

int main(int argc, char **argv)
{
    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    if (!RunOp()) {
        DestoryResource();
        return FAILED;
    }

    DestoryResource();

    return SUCCESS;
}
