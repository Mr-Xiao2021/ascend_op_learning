## 概述
本样例相比于AclNNInvocation样例工程，简化了工程配置。
## 目录结构介绍
```
├── AclNNInvocationNaive
│   ├── CMakeLists.txt      // 编译规则文件
│   ├── main.cpp            // 单算子调用应用的入口
│   └── run.sh              // 编译运行算子的脚本
```
## 代码实现介绍
完成自定义算子的开发部署后，可以通过单算子调用的方式来验证单算子的功能。main.cpp代码为单算子API执行方式。单算子API执行是基于C语言的API执行算子，无需提供单算子描述文件进行离线模型的转换，直接调用单算子API接口。

自定义算子编译部署后，会自动生成单算子API，可以直接在应用程序中调用。算子API的形式一般定义为“两段式接口”，形如：
   ```cpp
   // 获取算子使用的workspace空间大小
   aclnnStatus aclnnAddCustomGetWorkspaceSize(const aclTensor *x, const aclTensor *y, const alcTensor *out, uint64_t workspaceSize, aclOpExecutor **executor);
   // 执行算子
   aclnnStatus aclnnAddCustom(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream);
   ```
其中aclnnAddCustomGetWorkspaceSize为第一段接口，主要用于计算本次API调用计算过程中需要多少的workspace内存。获取到本次API计算需要的workspace大小之后，按照workspaceSize大小申请Device侧内存，然后调用第二段接口aclnnAddCustom执行计算。具体参考[AscendCL单算子调用](https://hiascend.com/document/redirect/CannCommunityAscendCInVorkSingleOp)>单算子API执行 章节。
## 运行样例算子
### 1. 编译算子工程
运行此样例前，请参考[编译算子工程](../README.md#operatorcompile)完成前期准备。
### 2. aclnn调用样例运行

  - 进入到样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/AddCustomSample/FrameworkLaunchLite/AclNNInvocationNaive
    ```
  - 样例编译文件修改

    将CMakeLists.txt文件内"/usr/local/Ascend/ascend-toolkit/latest"替换为CANN软件包安装后的实际路径。  
    eg:/home/HwHiAiUser/Ascend/ascend-toolkit/latest

    CMakeLists.txt文件内设置"CUST_PKG_PATH"的变量为"../CustomOp/build_out/op_api"，是使用的相对路径，若自定义算子工程编译产物在其他路径，请修改为实际路径。

  - 环境变量配置

    需要设置NPU_HOST_LIB环境变量，以x86为例
    ```bash
    export NPU_HOST_LIB=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/x86_64-linux/lib64
    ```
  - 样例执行

    样例执行过程中会自动生成测试数据，然后编译与运行aclnn样例，最后打印运行结果。
    ```bash
    mkdir -p build
    cd build
    cmake .. && make
    ./execute_add_op
    ```

    用户亦可参考run.sh脚本进行编译与运行。
    ```bash
    bash run.sh -r [RUN_MODE] -v [SOC_VERSION]
    ```
    - RUN_MODE：运行方式，可选择NPU上板和NPU仿真。支持参数为[npu /sim]
    - SOC_VERSION：昇腾AI处理器型号，只支持在RUN_MODE为sim时配置。如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下参数取值（xxx请替换为具体取值）：
      - Atlas 推理系列产品（Ascend 310P处理器）参数值：Ascend310P1、Ascend310P3
      - Atlas 训练系列产品参数值：AscendxxxA、AscendxxxB
      - Atlas A2训练系列产品参数值：AscendxxxB1、AscendxxxB2、AscendxxxB3、AscendxxxB4

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2024/10/21 | 新增本readme |