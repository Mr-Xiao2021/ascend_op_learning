## Broadcast自定义算子样例说明

本样例通过Ascend C编程语言实现了Broadcast算子，并给出了对应的aclnn端到端实现。

- [FrameworkLaunch](./FrameworkLaunch)：使用框架调用Broadcast自定义算子。

  按照工程创建->算子实现->编译部署>算子调用的流程完成算子开发。整个过程都依赖于算子工程：基于工程代码框架完成算子核函数的开发和Tiling实现，通过工程编译脚本完成算子的编译部署，继而实现单算子调用或第三方框架中的算子调用。

本样例中包含如下调用方式：

<table>
    <th>调用方式</th><th>目录</th><th>描述</th>
    <tr>
        <!-- 列的方向占据7个cell -->
        <td rowspan='2'><a href="./FrameworkLaunch"> FrameworkLaunch</td>
    </tr>
    <tr>
        <td><a href="./FrameworkLaunch/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用Broadcast算子。</td>
    </tr>
    </tr>
<table>

## 算子描述
Broadcast算子实现了将输入数据按照输出shape进行广播的功能，比如A的shape为(2,1)，广播的目标shape为(2,16)，则会将原来的一列扩展为相同的16列, A的shape变为(2,16)。

## 算子规格描述

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Broadcast</td></tr>
</tr>
<tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape_range</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">16 * 1</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">16 * 3</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">broadcast_custom</td></tr>
</table>

## 支持的产品型号

本样例支持如下产品型号：
- Atlas 推理系列产品（Ascend 310P处理器）
- Atlas A2训练系列产品/Atlas 800I A2推理产品

## 目录结构介绍

```
├── FrameworkLaunch    //使用框架调用的方式调用Broadcast自定义算子工程。
```

## 环境要求

编译运行此样例前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

## 编译运行样例算子

### 1. 准备：获取样例代码<a name="codeready"></a>

 可以使用以下两种方式下载，请选择其中一种进行源码准备。

- 命令行方式下载（下载时间较长，但步骤简单）。

  ```bash
  # 开发环境，非root用户命令行中执行以下命令下载源码仓。git_clone_path为用户自己创建的某个目录。
  cd ${git_clone_path}
  git clone https://gitee.com/ascend/samples.git
  ```
- 压缩包方式下载（下载时间较短，但步骤稍微复杂）。

  **注：如果需要下载其它版本代码，请先请根据前置条件说明进行samples仓分支切换。下载压缩包命名跟tag/branch相关，此处以master分支为例，下载的名字将会是samples-master.zip**

  ```bash
  # 1. samples仓右上角选择 【克隆/下载】 下拉框并选择 【下载ZIP】。
  # 2. 将ZIP包上传到开发环境中的普通用户某个目录中，【例如：${git_clone_path}/samples-master.zip】。
  # 3. 开发环境中，执行以下命令，解压zip包。
  cd ${git_clone_path}
  unzip samples-master.zip
  ```

### 2. 编译运行样例工程

- 若使用框架调用的方式，编译运行操作请参见[FrameworkLaunch](./FrameworkLaunch)。

## 更新说明

| 时间       | 更新事项   |
| ---------- | ---------- |
| 2024/10/11 | 新增readme |