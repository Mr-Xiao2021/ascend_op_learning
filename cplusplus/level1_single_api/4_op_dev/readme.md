
# 本目录后续不再演进，Ascend C算子样例开发请移步[operator](../../../operator)
# 自定义算子开发样例

#### 概述

本样例包含了自定义算子的开发样例和功能验证样例。
- 开发样例中包含了TBE自定义算子、AI CPU自定义算子等代码样例，同时提供了对应的编译规则文件，开发者可以基于本样例进行算子工程的编译获得自定义算子安装包，并将该算子包部署到CANN算子库中。

- 功能验证样例提供了基于AscendCL单算子调用方式的算子运行样例，其主要原理为将自定义算子转换为单算子离线模型文件，然后通过AscendCL加载单算子模型文件进行运行。


#### 目录结构与说明
  

| 目录  | 说明  |
|---|---|
| [1_custom_op](./1_custom_op)  | 自定义算子开发样例目录  |
| [2_verify_op](./2_verify_op)  | 自定义算子功能验证目录  |

#### 样例使用方法

1. 参考自定义算子开发样例，您可以了解每一个样例算子，并对算子工程进行编译获得自定义算子包，继而对算子包进行部署；您也可以在样例的基础上追加自己的自定义算子实现代码，进行自定义算子的开发。
2. 完成自定义算子包的部署后，您可以参考自定义算子功能验证样例，进行样例中算子的运行验证；您也可以参考该样例，编写自己的自定义算子验证代码。