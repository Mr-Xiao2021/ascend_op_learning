#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import logging
import tensorflow as tf
import numpy as np
import npu_device
from npu_device.compat.v1.npu_init import *
npu_device.compat.enable_v1()

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.flags.DEFINE_string("local_log_dir", "output/train_logs.txt", "Log file path")
FLAGS = tf.compat.v1.flags.FLAGS
atol = 0.001
rtol = 0.001


def config(excute_type):
    if excute_type == 'ai_core':
        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["enable_data_pre_proc"].b = True
        custom_op.parameter_map["mix_compile_mode"].b = True
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["min_group_size"].b = 1

    elif excute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    return session_config


def main(unused_argv):
    shape_params = (8, 2048)
    dtype_params = np.float16
    x_data = np.random.uniform(-2, 2, size=shape_params).astype(dtype_params)
    y_data = np.random.uniform(-2, 2, size=shape_params).astype(dtype_params)

    x = tf.compat.v1.placeholder(dtype_params, shape=shape_params)
    y = tf.compat.v1.placeholder(dtype_params, shape=shape_params)
    out = tf.math.add(x, y)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        result_cpu = session.run(out, feed_dict={x: x_data, y: y_data})
    with tf.compat.v1.Session(config=config('ai_core')) as session:
        print("run npu")
        result_ai_core = session.run(out, feed_dict={x: x_data, y: y_data})

    np.array(result_ai_core).astype(dtype_params)
    np.array(result_cpu).astype(dtype_params)

    print('====================================')
    cmp_result = np.allclose(result_ai_core, result_cpu, atol, rtol)
    print(cmp_result)
    print('====================================')


if __name__ == "__main__":
    tf.compat.v1.app.run()
