"""
# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
import numpy as np
import dataflow as df
from udf_py.udf_control import UserFunc0
from udf_py.udf_add import UserFunc1

# dataflow初始化参数
options = {
    "ge.exec.deviceId":"0",
    "ge.experiment.data_flow_deploy_info_path":"./config/data_flow_deploy_info.json",
    "ge.socVersion":"Ascend910B"
}
df.init(options)

# 构图
'''
FlowData
   |
FlowNode(ProcessPoint 控制多func的激活)
   |
FlowNode(ProcessPoint 多func)
'''

# 定义输入
data0 = df.FlowData()

# 定义FuncProcessPoint
pp0 = df.FuncProcessPoint(py_func=UserFunc0,workspace_dir="./py_add_ws")

flow_node0 = df.FlowNode(input_num=1, output_num=4)
flow_node0.add_process_point(pp0)

pp1 = df.FuncProcessPoint(py_func=UserFunc1,workspace_dir="./py_add_ws")
flow_node1 = df.FlowNode(input_num=4, output_num=2)
flow_node1.add_process_point(pp1)
# 构建连边关系
flow_node0_out = flow_node0(data0)
flow_node1_out0, flow_node1_out1 = flow_node1(*flow_node0_out)
# 构建图
dag = df.FlowGraph([flow_node1_out0, flow_node1_out1])

# feed data
feed_data0 = np.array([0], dtype=np.int32)

flow_info=df.FlowInfo()
flow_info.start_time = 0
flow_info.end_time = 5
# 激活多func中第一个函数
dag.feed_data({data0:feed_data0}, flow_info)
# 获取第一个函数的输出
result = dag.fetch_data([0])

print("TEST-OUTPUT0:", result)
# 激活多func中第二个函数
feed_data0 = np.array([1], dtype=np.int32)
dag.feed_data({data0:feed_data0}, flow_info)
# 获取第二个函数的输出
result = dag.fetch_data([1])

print("TEST-OUTPUT1:", result)


# 释放dataflow资源
df.finalize()