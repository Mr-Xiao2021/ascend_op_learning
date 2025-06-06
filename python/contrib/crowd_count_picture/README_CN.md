中文|[English](README.md)

**本样例为大家学习昇腾软件栈提供参考，非商业目的！**

**本README只提供命令行方式运行样例的指导，如需在Mindstudio下运行样例，请参考[Mindstudio运行图片样例wiki](https://gitee.com/ascend/samples/wikis/Mindstudio%E8%BF%90%E8%A1%8C%E5%9B%BE%E7%89%87%E6%A0%B7%E4%BE%8B?sort_id=3164874)。**

## crowd_count_picture 样例    
功能：使用count_person.caffe模型对密集人群进行计数。  
样例输入：密集人群图像  
样例输出：标记出人数的图像。   

### 前置条件
请检查以下条件要求是否满足，如不满足请按照备注进行相应处理。如果CANN版本升级，请同步检查第三方依赖是否需要重新安装（5.0.4及以上版本第三方依赖和5.0.4以下版本有差异，需要重新安装）。
| 条件 | 要求 | 备注 |
|---|---|---|
| CANN版本 | >=5.0.4 | 请参考CANN样例仓介绍中的[安装步骤](https://gitee.com/ascend/samples#%E5%AE%89%E8%A3%85)完成CANN安装，如果CANN低于要求版本请根据[版本说明](https://gitee.com/ascend/samples/blob/master/README_CN.md#%E7%89%88%E6%9C%AC%E8%AF%B4%E6%98%8E)切换samples仓到对应CANN版本 |
| 硬件要求 | Atlas200DK/Atlas 200I DK A2/Atlas300([ai1s](https://support.huaweicloud.com/productdesc-ecs/ecs_01_0047.html#ecs_01_0047__section78423209366))  | 当前已在Atlas200DK、Atlas 200I DK A2和Atlas300测试通过，产品说明请参考[硬件平台](https://ascend.huawei.com/zh/#/hardware/product) ，其他产品可能需要另做适配|
| 第三方依赖 |  opencv，python-acllite | 请参考[第三方依赖安装指导（python样例）](../../environment)选择需要的依赖完成安装 |

### 样例准备

1. 获取源码包。

   可以使用以下两种方式下载，请选择其中一种进行源码准备。   
    - 命令行方式下载（下载时间较长，但步骤简单）。
       ```    
       # 开发环境，非root用户命令行中执行以下命令下载源码仓。    
       cd ${HOME}     
       git clone https://gitee.com/ascend/samples.git
       ```
       **注：如果需要切换到其它tag版本，以v0.5.0为例，可执行以下命令。**
       ```
       git checkout v0.5.0
       ```   
    - 压缩包方式下载（下载时间较短，但步骤稍微复杂）。   
       **注：如果需要下载其它版本代码，请先请根据前置条件说明进行samples仓分支切换。**   
       ``` 
        # 1. samples仓右上角选择 【克隆/下载】 下拉框并选择 【下载ZIP】。    
        # 2. 将ZIP包上传到开发环境中的普通用户家目录中，【例如：${HOME}/ascend-samples-master.zip】。     
        # 3. 开发环境中，执行以下命令，解压zip包。     
        cd ${HOME}    
        unzip ascend-samples-master.zip
        ```

2. 获取此应用中所需要的原始网络模型。
    |  **模型名称**  |  **模型说明**  |  **模型下载路径**  |
    |---|---|---|
    |  count_person.caffe| 	基于caffe的密集人群计数处理。  |  请参考https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/crowdCount/ATC_count_person_caffe_AE](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/crowdCount/ATC_count_person_caffe_AE)原始模型章节，下载原始模型及对应的cfg文件。 |
    ```
    # 为了方便下载，在这里直接给出原始模型下载及模型转换命令,可以直接拷贝执行。也可以参照上表在modelzoo中下载并手工转换，以了解更多细节。     
    cd ${HOME}/samples/python/contrib/crowd_count_picture/model    
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/crowdCount/count_person.caffe.caffemodel
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/crowdCount/count_person.caffe.prototxt
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/crowdCount/insert_op.cfg
    atc --input_shape="blob1:1,3,800,1408" --weight="count_person.caffe.caffemodel" --input_format=NCHW --output="count_person.caffe" --soc_version=Ascend310 --insert_op_conf=insert_op.cfg --framework=0 --model="count_person.caffe.prototxt" 
    **注：如果在310B芯片上进行模型转换，修改参数--soc_version=Ascend310B1即可**
    ```

3. 获取样例需要的测试图片。
    ```
    执行以下命令，进入样例的data文件夹中，下载对应的测试图片。
    cd $HOME/samples/python/contrib/crowd_count_picture/data
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/crowdCount/crowd.jpg
    cd ../src
    ```

### 样例运行

**注1：开发环境与运行环境合一部署，请跳过步骤1，直接执行[步骤2](#step_2)即可。**   

**注2：若在310B芯片上进行推理，需要修改部分代码**
  打开程序主入口文件 main.py
  找到函数'pre_process'，将该函数修改为如下内容保存即可：
  ```
      def pre_process(self, image):
        """
        image preprocess
        """
        image_dvpp = image.copy_to_dvpp()
        yuv_image = self._dvpp.jpegd(image_dvpp)
        return yuv_image
  ```
  再找到'main()'中的images_list相关for循环部分，修改为如下内容保存即可：
  ```
      for image_file in images_list:
        ori_image = cv2.imread(image_file, 1)
        ori_image = cv2.resize(ori_image,(MODEL_WIDTH,MODEL_HEIGHT))
        save_name,ext = os.path.splitext(image_file)
        cv2.imwrite(save_name+'resize'+ext, ori_image)
        image = AclLiteImage(save_name+'resize'+ext)            
        crop_and_paste_image = crowdcount.pre_process(image)
        print("pre process end")
        result = crowdcount.inference([crop_and_paste_image])              
        result_img_encode = crowdcount.post_process(crop_and_paste_image, result, image_file)
        os.remove(save_name+'resize'+ext)
  ```

1. 执行以下命令,将开发环境的 **crowd_count_picture** 目录上传到运行环境中，例如 **/home/HwHiAiUser**，并以HwHiAiUser（运行用户）登录运行环境（Host）。
    ```
    # 【xxx.xxx.xxx.xxx】为运行环境ip，200DK在USB连接时一般为192.168.1.2，300（ai1s）为对应的公网ip。
    scp -r $HOME/samples/python/contrib/crowd_count_picture HwHiAiUser@xxx.xxx.xxx.xxx:/home/HwHiAiUser
    ssh HwHiAiUser@xxx.xxx.xxx.xxx
    cd ${HOME}/crowd_count_picture/src    
    ```

2. <a name="step_2"></a>运行样例。
   ```
   python3.6 main.py ../data/
   ```

### 查看结果

运行完成后，会在样例工程的out目录下生成推理后的图片。

### 常见错误
请参考[常见问题定位](https://gitee.com/ascend/samples/wikis/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E5%AE%9A%E4%BD%8D/%E4%BB%8B%E7%BB%8D)对遇到的错误进行排查。如果wiki中不包含，请在samples仓提issue反馈。