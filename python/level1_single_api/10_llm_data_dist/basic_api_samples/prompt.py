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
import logging
import time
from llm_datadist import LLMDataDist, LLMRole, LLMConfig, CacheDesc, CacheKey, KvCache, DataType
import torch
import torch_npu
import torchair

# 用于被拉取KV的ip与端口
DEVICE_IP_LIST = ['192.168.1.1', '192.168.1.2', '192.168.1.3', '192.168.1.4',
                  '192.168.1.5', '192.168.1.6', '192.168.1.7', '192.168.1.8']
DEVICE_LISTEN_PORT = 26000
DEVICE_ID_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
PADDING_REQ_ID = 2 ** 64 - 1

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def init_llm_datadist(rank_id: int) -> LLMDataDist:
    datadist = LLMDataDist(LLMRole.PROMPT, 0)
    llm_config = LLMConfig()
    listen_ip_info = DEVICE_IP_LIST[rank_id] + ":" + str(DEVICE_LISTEN_PORT)  # ex. 192.168.1.1:26000
    llm_config.listen_ip_info = listen_ip_info
    llm_config.device_id = DEVICE_ID_LIST[rank_id]
    llm_options = llm_config.generate_options()
    datadist.init(llm_options)
    return datadist


def run_with_kv_cache(kv_cache: KvCache):
    kv_tensor_addrs = kv_cache.per_device_tensor_addrs[0]
    logging.info(f'cache_id={kv_cache.cache_id}, kv_tensor_addrs={kv_tensor_addrs}')
    # 使用kv_tensor_addr 构造对应前端框架(如torch)的Tensor
    kv_tensors = torchair.llm_datadist.create_npu_tensors(kv_cache.cache_desc.shape, torch.float16, kv_tensor_addrs)
    # 对kv cache进行赋值,验证Decoder pull_cache的结果
    kv_tensors[0].fill_(1)


def run_prompt_sample(rank_id: int):
    # 1. 初始化llm_datadist
    datadist = init_llm_datadist(rank_id)
    logging.info('[initialize] llm_datadist success')
    # 2. 通过kv_cache_manager分配kv cache
    kv_cache_manager = datadist.kv_cache_manager
    # 描述一个kv cache，管理4个tensor, 每个tensor的batch_size=4, dtype为FP16
    cache_desc = CacheDesc(num_tensors=4, shape=[4, 4, 8], data_type=DataType.DT_FLOAT16)
    # CacheKey用于关联kv cache中的单个batch_index，Decoder可通过同样的CacheKey通过pull_cache接口拉取该batch_index对应的数据
    # 如果kv cache需要被decoder拉取，则在调用allocate_cache时传入CacheKey的列表, 按下标顺序关联各个batch_index
    # 这里, 我们将2个cache_key与该kv cache关联，分别关联batch_index = 0与batch_index = 2
    cache_key_0 = CacheKey(prompt_cluster_id=0, req_id=0, model_id=0)
    # batch_index=1不需要被拉取，但cache_key_2需要关联到batch_index=2, 所以需要通过req_id=PADDING_REQ_ID的cache_key占位
    cache_key_padding = CacheKey(prompt_cluster_id=0, req_id=PADDING_REQ_ID, model_id=0)
    cache_key_2 = CacheKey(prompt_cluster_id=0, req_id=1, model_id=0)
    # 最后一个需要关联的cache_key之后，可以省略padding的cache_key
    kv_cache = kv_cache_manager.allocate_cache(cache_desc, [cache_key_0, cache_key_padding, cache_key_2])
    logging.info('[allocate_cache] success')
    # 操作kv cache，用户可以改写该方法使用kv cache进行推理
    run_with_kv_cache(kv_cache)
    kv_host_tensors = kv_cache_manager.get_cache_tensors(kv_cache, 0)
    logging.info(f'[get_cache_tensor] success, kv_tensor_data = {kv_host_tensors[0].numpy()}')
    # 等待decoder拉取kv cache完成, 这里简单sleep下
    logging.info('wait for 30 seconds')
    time.sleep(30)
    logging.info('wait ended')
    # 如果pull_cache失败，或者decoder没有调用pull_cache，此处需要调用remove_cache_key，确保kv_cache能够得到释放
    # 如果pull_cache成功，这里只是个空操作
    kv_cache_manager.remove_cache_key(cache_key_0)
    kv_cache_manager.remove_cache_key(cache_key_2)
    logging.info('[remove_cache_key] success')
    kv_cache_manager.deallocate_cache(kv_cache)
    logging.info('[deallocate_cache] success')
    datadist.finalize()
    logging.info('[finalize] success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank_id", type=int, default=0, help='rank_id')
    args = parser.parse_args()
    logging.info(f'Sample start, rank_id = {args.rank_id}, device_id = {DEVICE_ID_LIST[args.rank_id]}')
    torch.npu.set_device(DEVICE_ID_LIST[args.rank_id])
    run_prompt_sample(args.rank_id)
    logging.info('Sample end')

