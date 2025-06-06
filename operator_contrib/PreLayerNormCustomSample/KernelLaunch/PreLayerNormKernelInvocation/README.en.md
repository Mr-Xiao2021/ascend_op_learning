## Directory Structure Introduction
``` 
├── PreLayerNormKernelInvocation
│   ├── cmake                   // Compilation project files
│   ├── scripts
│   │   ├── gen_data.py         // Script for generating input data and ground truth data
│   │   └── verify_result.py    // Ground truth comparison file
│   ├── pre_layer_norm_custom.cpp   // Operator kernel implementation
│   ├── CMakeLists.txt          // Compilation project file
│   ├── data_utils.h            // Data input and output functions
│   ├── main.cpp                // Main function, application that invokes the operator, including CPU and NPU domain calls
│   └── run.sh                  // Script for compiling and running the operator
``` 

## Code Implementation Introduction
This invocation example implements the PreLayerNorm operator with a fixed shape.

- Kernel Implementation   
 The functionality of the PreLayerNorm operator is: PreLayerNorm is a fusion operator of Add and LayerNorm, where the output of the Add operator serves as the first input to the LayerNorm operator. For the input data x and y, after they are added together, the values of Add(x, y) are adjusted according to the coefficients beta and the bias gamma to converge to a fixed interval.

  The calculation logic is: The vector calculation interfaces provided by Ascend C operate on LocalTensor elements. The input data needs to be first moved into on-chip storage, and then the computation interface is used to complete the addition of the two input parameters x and y. After that, the final result is obtained based on the learning coefficient gamma and the bias beta, and then it is moved out to external storage.  

  The implementation process of the PreLayerNorm operator is divided into three basic tasks: CopyIn, Compute, and CopyOut. The CopyIn task is responsible for moving the input Tensor xGm, gammaGm, and betaGm from Global Memory to Local Memory, stored in xLocal, gammaLocal, and betaLocal respectively. The Compute task is responsible for performing operations on xLocal, gammaLocal, and betaLocal, and storing the result in outLocal. The CopyOut task is responsible for moving the output data from outLocal to the output Tensor outGm in Global Memory. For details, please refer to [pre_layer_norm_custom.cpp](./pre_layer_norm_custom.cpp).

- Invocation Implementation  
  1. CPU-side runtime verification is mainly completed through interfaces provided by the CPU debugging library such as the ICPU_RUN_KF CPU debugging macro;  
  2. NPU-side runtime verification is mainly completed using the <<<>>> kernel invocation operator.    
  The application distinguishes between code logic running on the CPU side and the NPU side through the ASCENDC_CPU_DEBUG macro.

## Running the Example Operator
- Open the example directory

  ```bash
  cd ${git_clone_path}/samples/operator_contrib/PreLayerNormCustomSample/KernelLaunch/PreLayerNormKernelInvocation
  ```

- Configure environment variables

  Please select the corresponding command to configure environment variables based on the [installation method](https://hiascend.com/document/redirect/CannCommunityInstSoftware) of the CANN development toolkit package on the current environment.
    - Default path, root user installs CANN software package
      ```bash
      export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
      ```
    - Default path, non-root user installs CANN software package
      ```bash
      export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
      ```
    - Specified path install_path, installs CANN software package
      ```bash
      export ASCEND_INSTALL_PATH=${install_path}/ascend-toolkit/latest
      ```
    
    Configure the simulation mode log file directory, default is sim_log.
    ```bash
    export CAMODEL_LOG_PATH=./sim_log
    ```

- Example execution

  ```bash
  bash run.sh -r [RUN_MODE] -v  [SOC_VERSION] 
  ```
  - RUN_MODE: Compilation method, can choose CPU debugging, NPU simulation, NPU on-board. Supported parameters are [cpu / sim / npu], default is cpu.
  - SOC_VERSION: Ascend AI processor model. If you are unsure of the specific [SOC_VERSION], execute the npu-smi info command on the server with the Ascend AI processor to query it. Add the Ascend information before the "Name" in the query result. For example, if the "Name" corresponds to the value xxxyy, the actual configured [SOC_VERSION] value is Ascendxxxyy. Supported parameter values (replace xxx with the specific value):
    - Atlas A2 training series products parameter values: AscendxxxB1, AscendxxxB2, AscendxxxB3, AscendxxxB4

  Example:
  ```bash
  bash run.sh -r cpu -v Ascend910B4
  ```   

## Update Log
  | Date       | Update Items |
  |------------|--------------|
  | 2023/7/2   | Added this readme |