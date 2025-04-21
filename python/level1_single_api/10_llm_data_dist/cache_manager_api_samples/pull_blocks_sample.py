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

import argparse
import json
import logging
import time
from llm_datadist import LLMDataDist, LLMRole, LLMConfig, CacheDesc, Cache, DataType, RegisterMemStatus, BlocksCacheKey
import torch
import torch_npu
import torchair

DEVICE_IP_LIST = ['192.168.1.1', '192.168.1.2', '192.168.1.3', '192.168.1.4',
                  '192.168.1.5', '192.168.1.6', '192.168.1.7', '192.168.1.8']
REMOTE_IP_LIST = ['192.168.1.1', '192.168.1.2', '192.168.1.3', '192.168.1.4',
                  '192.168.1.5', '192.168.1.6', '192.168.1.7', '192.168.1.8']

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def init_llm_datadist(role: LLMRole, cluster_id, device_id: int) -> LLMDataDist:
    datadist = LLMDataDist(role, cluster_id)
    llm_config = LLMConfig()
    llm_config.device_id = device_id
    llm_config.enable_cache_manager = True
    llm_options = llm_config.generate_options()
    datadist.init(llm_options)
    return datadist


def link(datadist, device_id):
    rank_table_dict = {
        "server_count": "2",
        "status": "completed",
        "version": "1.0",
        "server_list": [
            {
                "device": [
                    {
                        "device_id": str(device_id),
                        "device_ip": DEVICE_IP_LIST[device_id],
                        "rank_id": "0"
                    }
                ],
                "service_id": "1"
            },
            {
                "device": [
                    {
                        "device_id": str(device_id),
                        "device_ip": REMOTE_IP_LIST[device_id],
                        "rank_id": "1"
                    }
                ],
                "service_id": "2"
            }
        ]
    }
    # 当前展示两个节点cluster id分别为1和2, rank id分别为0和1
    cluster_rank_info = {1: 0, 2: 1}
    rank_table = json.dumps(rank_table_dict)
    comm_id = datadist.link(cluster_rank_info, rank_table)
    while True:
        ret = datadist.query_register_mem_status(comm_id)
        if ret == RegisterMemStatus.OK:
            logging.info('query_register_mem_status ok')
            break
        elif ret == RegisterMemStatus.FAILED:
            logging.info('query_register_mem_status failed')
            raise RuntimeError("link failed")
        logging.info("need check again")
        time.sleep(1)


def run_decoder_sample(datadist, device_id: int):
    cache_manager = datadist.cache_manager
    cache_desc = CacheDesc(num_tensors=1, shape=[2, 1024 * 1024], data_type=DataType.DT_FLOAT)
    tensor = torch.ones(2, 1024 * 1024, dtype=torch.float).npu()
    addr = int(tensor.data_ptr())
    cache = cache_manager.register_blocks_cache(cache_desc, [addr])
    logging.info('[allocate_cache] success')

    link(datadist, device_id)

    # wait prompt prepared
    time.sleep(5)
    cache_manager.pull_blocks(BlocksCacheKey(1, 0), cache, src_blocks=[0, 1], dst_blocks=[0, 1])
    logging.info(f"after pull, tensor={tensor}")

    cache_manager.deallocate_blocks_cache(cache)
    datadist.finalize()


def run_prompt_sample(datadist, device_id: int):
    cache_manager = datadist.cache_manager
    cache_desc = CacheDesc(num_tensors=1, shape=[2, 1024 * 1024], data_type=DataType.DT_FLOAT)
    tensor = torch.ones(2, 1024 * 1024, dtype=torch.float).npu()
    addr = int(tensor.data_ptr())
    cache = cache_manager.register_blocks_cache(cache_desc, [addr], BlocksCacheKey(1, 0))
    logging.info('[allocate_cache] success')

    link(datadist, device_id)
    logging.info('wait for 30 seconds')
    time.sleep(30)
    logging.info('wait ended')
    cache_manager.deallocate_blocks_cache(cache)
    datadist.finalize()
    logging.info('[finalize] success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=int, default=0, help='device id')
    parser.add_argument("--cluster_id", type=int, default=1, help='cluster id')
    args = parser.parse_args()
    if args.cluster_id not in [1, 2]:
        raise RuntimeError("Not supported cluster id")
    if args.device_id not in [0, 1, 2, 3, 4, 5, 6, 7]:
        raise RuntimeError("Not supported device id")
    logging.info(f'Sample start, device_id = {args.device_id}, device_id = {args.device_id}')
    torch.npu.set_device(args.device_id)
    role = LLMRole.PROMPT if args.cluster_id == 1 else LLMRole.DECODER
    datadist = init_llm_datadist(role, args.cluster_id, args.device_id)
    if role == LLMRole.PROMPT:
        run_prompt_sample(datadist, args.device_id)
    else:
        run_decoder_sample(datadist, args.device_id)
    logging.info('Sample end')
