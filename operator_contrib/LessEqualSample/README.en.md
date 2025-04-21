## LessEqual Custom Operator Example Explanation

This example demonstrates the implementation of the LessEqual operator using the Ascend C programming language, and provides end-to-end implementations for different operator invocation methods.

- [FrameworkLaunch](./FrameworkLaunch/README.en.md): Invokes the LessEqual custom operator using the framework.  
  The operator development follows the process of project creation -> operator implementation -> compilation and deployment -> operator invocation. The entire process relies on the operator project: the operator kernel function and Tiling are developed based on the project code framework, the operator is compiled and deployed through the project compilation script, and then the operator is invoked either as a standalone operator or within a third-party framework.

- [KernelLaunch](./KernelLaunch/README.en.md): Invokes the LessEqual custom operator directly using the kernel function.  
  This is the basic kernel function invocation (Kernel Launch) method. After developers complete the development of the operator kernel function and Tiling, they can invoke the operator through the AscendCL runtime interface.

This example includes the following invocation methods:

<table>
    <th>Invocation Method</th><th>Directory</th><th>Description</th>
    <tr>
        <td rowspan='1'><a href="./FrameworkLaunch/README.en.md"> FrameworkLaunch</td><td><a href="./FrameworkLaunch/AclNNInvocation/README.en.md"> AclNNInvocation</td><td>Invokes the LessEqualCustom operator using the aclnn method.</td>
    </tr>
    <tr>
        <td rowspan='1'><a href="./KernelLaunch/README.en.md"> KernelLaunch</td><td><a href="./KernelLaunch/LessEqualKernelInvocation/README.en.md"> LessEqualKernelInvocation</td><td>Host-side kernel function invocation program, including validation methods for both CPU and NPU.</td>
    </tr>
</table>

## Operator Description

The LessEqual operator implements the functionality where the corresponding position in vector y outputs true if vector x1 is less than or equal to x2, otherwise it outputs false.

The corresponding mathematical expression is:
```
y = x1 <= x2
```

## Operator Specification Description

<table>  
<tr><th align="center">Operator Type (OpType)</th><th colspan="4" align="center">LessEqual</th></tr>  
<tr><td rowspan="3" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td align="center">x1</td><td align="center">-</td><td align="center">float32, float16, int32, int8</td><td align="center">ND</td></tr>  
<tr><td align="center">x2</td><td align="center">-</td><td align="center">float32, float16, int32, int8</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">Operator Output</td><td align="center">y</td><td align="center">-</td><td align="center">float32, float16, int32, int8</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">LessEqual</td></tr>
</table>

## Supported Product Models

This example supports the following product models:
- Atlas 200/500 A2 inference products
- Atlas A2 Training Series Products / Atlas 800I A2 Inference Products

## Directory Structure Introduction

```
├── FrameworkLaunch    // Invokes the LessEqual custom operator using the framework.
└── KernelLaunch       // Invokes the LessEqual custom operator directly using the kernel function.
```

## Environment Requirements

Before compiling and running this example, please refer to the [CANN Software Installation Guide](https://hiascend.com/document/redirect/CannCommunityInstSoftware) to deploy the development and runtime environment.

## Compiling and Running the Example Operator

### 1. Preparation: Obtain the Example Code<a name="codeready"></a>

You can download the source code using one of the following two methods. Please choose one:

- Command-line method (long download time, simple steps).

  ```bash
  # In the development environment, execute the following command as a non-root user to download the source code repository. git_clone_path is a directory created by the user.
  cd ${git_clone_path}
  git clone https://gitee.com/ascend/samples.git
  ```
  **Note: If you need to switch to another tag version, for example, v0.5.0, you can execute the following command.**
  ```bash
  git checkout v0.5.0
  ```

- Zip package method (short download time, slightly more complex steps).

  **Note: If you need to download the code for another version, please first switch the samples repository branch according to the prerequisites.**
  ```bash
  # 1. In the samples repository, select the 【Clone/Download】 dropdown and choose 【Download ZIP】.
  # 2. Upload the ZIP package to a directory of a regular user in the development environment, for example, ${git_clone_path}/ascend-samples-master.zip.
  # 3. In the development environment, execute the following command to unzip the zip package.
  cd ${git_clone_path}
  unzip ascend-samples-master.zip
  ```

### 2. Compile and Run the Example Project

- If using the framework invocation method, please refer to [FrameworkLaunch](./FrameworkLaunch/README.en.md) for compilation and running operations.    
- If using the kernel function direct invocation method, please refer to [KernelLaunch](./KernelLaunch/README.en.md) for compilation and running operations.

## Update Log

| Date       | Update Item |
|------------|-------------|
| 2024/05/24 | New readme update |
| 2024/07/22 | Modified to clone to any directory |