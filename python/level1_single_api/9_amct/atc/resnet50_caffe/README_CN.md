# Caffe 框架示例

Caffe 框架 ResNet-50 分类网络模型量化

## 1. 准备模型

通过执行以下命令下载原始模型定义文件 (.prototxt) 与预训练权重文件 (.caffemodel)：

```bash
python3.7.5 ./src/download_models.py --close_certificate_verify
```

> 其中，`--close_certificate_verify` 参数可选，用于关闭证书验证参数，确保模型正常下载。如果模型下载过程中提示认证失败相关信息，则可以增加该参数重新下载。

若执行成功，将会下载 `ResNet-50-deploy.prototxt` 与 `ResNet-50-model.caffemodel` 到该样例根目录的 [model](./model/) 路径下。

> 若用户环境无法连接网络,请先在可连通网络的服务器下载相应文件后上传到 [model](./model/) 路径下：
>
> ```bash
> wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/resnet_50/ResNet-50-deploy.prototxt
> wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/resnet_50/ResNet-50-model.caffemodel
> ```

## 2. 准备校准数据集

校准集用来产生量化因子，保证精度。计算量化参数的过程被称为“校准 (calibration)”。校准过程需要使用一部分图片来针对性计算量化参数，使用一个或多个 batch 对量化后的网络模型进行推理即可完成校准。为了保证量化精度，校准集与测试精度的数据集来源应一致。

该示例提供了一组样例校准集用于量化校准，可切换到该样例根目录下执行以下命令获取校准数据与标签：

```bash
cd data
mkdir image && cd image
wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/amct_acl/classification/calibration.rar
unrar e calibration.rar
```

> 如果环境中没有安装 unrar 工具，可以通过以下命令获取：
>
> ```bash
> sudo apt-get install unrar
> ```

## 3. 执行量化

本章节中量化脚本以Ascend310P3芯片为例（若需要修改芯片类型请参考第5章节）。

切换到该样例根目录下，执行如下命令进行量化：

```bash
bash ./scripts/run_calibration.sh 
```

若出现如下信息则说明量化成功：

```none
ATC run success
```

## 4. 量化结果

量化成功后，在当前目录（若设置了ASCEND_WORK_PATH环境变量，则在设置的目录下的amct_acl文件夹下）生成如下文件：

+ amct_log_用户进程PID_时间戳
  + amct_acl.log: 量化日志文件，记录昇腾模型压缩工具量化过程的日志信息。
+ fusion_result.json: 模型编译过程中使用的融合规则。
+ outputs
  + ResNet-50-model.om: 量化后的模型文件。

## 5. 芯片类型
scripts/run_calibration.sh中atc命令里的soc_version参数为量化后保存模型的芯片类型。
config/compression_opt.config中的soc_version参数为推理使用的芯片类型。

若使用Ascend RC推理芯片，需要配置RC推理芯片的IP地址和端口号等其他配置，请参考config/compression_opt_RC.config文件和相关文档。