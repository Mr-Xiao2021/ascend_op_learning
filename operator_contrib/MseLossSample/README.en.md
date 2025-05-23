## MseLoss Custom Operator Example Explanation
This example implements the MseLoss operator using the Ascend C programming language and provides end-to-end implementations for different operator invocation methods.

- [FrameworkLaunch](./FrameworkLaunch/README.en.md): Invokes the MseLoss custom operator using the framework.
  The operator development is completed following the process of project creation -> operator implementation -> compilation and deployment -> operator invocation. The entire process relies on the operator project: the operator kernel function and Tiling implementation are completed based on the project code framework, the operator is compiled and deployed through the project compilation script, and then the operator is invoked either as a single operator or within a third-party framework.

This example includes the following invocation methods:
<table>
    <th>Invocation Method</th><th>Directory</th><th>Description</th>
    <tr>
        <!-- Column occupies 1 cell -->
        <td rowspan='1'><a href="./FrameworkLaunch/README.en.md"> FrameworkLaunch</td><td><a href="./FrameworkLaunch/AclNNInvocation/README.en.md"> AclNNInvocation</td><td>Invokes the MseLoss operator using the aclnn method.</td>
    </tr>
</table>

## Operator Description
The MseLoss operator is a common loss function used in regression tasks in PyTorch. It calculates the Mean Squared Error (MSE) between the predicted values and the target values. However, in practical applications, it differs slightly from the definition. The main difference lies in the parameter settings, where the torch.nn.MSELoss has a reduction parameter. The reduction parameter determines whether and how to reduce the dimensions, with three main options available:

- ‘none’:no reduction will be applied;

- ‘mean’: the sum of the output will be divided by the number of elements in the output;

- ‘sum’: the output will be summed.

## Operator Specification Description
<table>
<tr><th align="center">Operator Type(OpType)</th><th colspan="5" align="center">MSELoss</th></tr>
<tr><td rowspan="3" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">默认值</td></tr>
<tr><td align="center">predict</td><td align="center">-</td><td align="center">fp32, fp16</td><td align="center">ND</td><td align="center">\</td></tr>
<tr><td align="center">label</td><td align="center">-</td><td align="center">fp32, fp16</td><td align="center">ND</td><td align="center">\</td></tr>
<tr><td rowspan="1" align="center">Operator Output</td><td align="center">y</td><td align="center">-</td><td align="center">fp32, fp16</td><td align="center">ND</td><td align="center">\</td></tr>
<tr><td rowspan="1" align="center">attr attribute</td><td align="center">reduction</td><td align="center">\</td><td align="center">string</td><td align="center">\</td><td align="center">mean</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">mse_loss</td></td></tr>
</table>

## Supported Product Models
This example supports the following product models:
- Atlas 200/500 A2 Inference Products
- Atlas A2 training series products / Atlas 800I A2 inference products

## Directory Structure Introduction
```
└── FrameworkLaunch    // Project for invoking the MseLoss custom operator using the framework.
```

## Environment Requirements
Before compiling and running this example, please refer to [《CANN Software Installation Guide》](https://hiascend.com/document/redirect/CannCommunityInstSoftware) to deploy the development and runtime environment.

## Compiling and Running the Example Operator

### 1. Preparation: Obtain the Example Code<a name="codeready"></a>

You can download the source code using one of the following two methods. Please choose one.

- Command line method (long download time, simple steps).

  ```bash
  # In the development environment, execute the following command as a non-root user to download the source repository. git_clone_path is a directory created by the user.
  cd ${git_clone_path}
  git clone https://gitee.com/ascend/samples.git
  ```
  **Note: If you need to switch to another tag version, for example, v0.5.0, you can execute the following command.**
  ```bash
  git checkout v0.5.0
  ```
- Zip package method (short download time, slightly more complex steps).

  **Note: If you need to download the code for another version, please first switch the samples repository branch according to the preconditions.**
  ```bash
  # 1. In the samples repository, select the 【Clone/Download】 dropdown and choose 【Download ZIP】.
  # 2. Upload the ZIP package to a directory of a normal user in the development environment, for example, ${git_clone_path}/ascend-samples-master.zip.
  # 3. In the development environment, execute the following command to unzip the zip package.
  cd ${git_clone_path}
  unzip ascend-samples-master.zip
  ```

### 2. Compile and Run the Example Project
- If using the framework invocation method, please refer to [FrameworkLaunch](./FrameworkLaunch/README.en.md) for compilation and running operations.

## Update Log
  | Date | Update Items |
|----|------|
| 2024/05/24 | New version of readme updated |
| 2024/07/22 | Modified to clone to any directory |
| 2024/07/24 | Modified readme format |