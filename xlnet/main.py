# -*- coding: utf-8 -*-

"""
Copyright 2019 Tae Hwan Jung

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import Timers

from utils import print_args
from utils import print_params_min_max_norm
from utils import print_rank_0
from utils import enable_adlr_autoresume
from utils import check_adlr_autoresume_termination


import os
import data_utils
import argparse
import xlargs
import xlnet
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer

import mpu 
os.environ['CUDA_VISIBLE_DEVICES']='0,1'


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    print("device=",device)
    if args.local_rank is not None:
        device = args.local_rank
        print("local",args.local_rank)
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        init_method=init_method)

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)
        

def print_args(args, writer=None):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)

        if writer:
            writer.add_text(arg, str(getattr(args, arg)))

def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)



def main():
    
    # Arguments.

    
    args = xlargs.get_args()
     # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()
    writer = None
    initialize_distributed(args)
    if torch.distributed.get_rank() == 0:
        print('Pretrain GPT2 model')
        print_args(args, writer)

    # Autoresume.
    print("barrier ahead")
    torch.distributed.barrier()
    print("barrier done")
    
    sp = BertTokenizer.from_pretrained(args.tokenizer)
    model = xlnet.XLNet(n_token=len(sp.vocab), n_layer=6, n_head=4, d_head=8,
                        d_inner=32, d_model=32,
                        dropout=0.1, dropatt=0.1,
                        attn_type="bi", bi_data=args.bi_data,
                        clamp_len=-1, same_length=False,
                        reuse_len=args.reuse_len, mem_len=args.mem_len)
    print(torch.cuda.current_device())
    
    
    model = model.cuda(torch.cuda.current_device())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    for num_epoch in range(args.num_epoch):
        mems = None

        features = data_utils._create_data(sp=sp,
                                           input_paths=args.data,
                                           seq_len=args.seq_len,
                                           reuse_len=args.reuse_len,
                                           bi_data=args.bi_data,
                                           num_predict=args.num_predict,
                                           mask_alpha=args.mask_alpha,
                                           mask_beta=args.mask_beta)

        num_step = 0
        for feature in features:

            permutation = data_utils.make_permute(feature,
                                                  reuse_len=args.reuse_len,
                                                  seq_len=args.seq_len,
                                                  perm_size=args.perm_size,
                                                  num_predict=args.num_predict)
            keys = ['seg_id']
            datatype = torch.int32
            permutations = mpu.broadcast_data(keys,permutation,datatype)
            key = ['input_k', 'target']
            segu = mpu.broadcast_data(key,permutation,torch.int64)
            keyf32 = ['perm_mask','target_mapping', 'input_q','target_mask']
            f32u = mpu.broadcast_data(keyf32,permutation,torch.float32)
            
            #########################################################################3
            # batch size is 1
            inp_k = segu['input_k'].unsqueeze(-1) # [seq_len, 1(=bsz)]
#             print("inp_k, inp_k.type: ", inp_k, type(inp_k), inp_k.dtype)
            seg_id = permutations['seg_id'].unsqueeze(-1) # [seq_len, 1(=bsz)]
#             print("seg_id, seg_id.type: ", seg_id, type(seg_id), seg_id.dtype)
            target = segu['target'].unsqueeze(-1) # [num_predict, 1(=bsz)]
            perm_mask = f32u['perm_mask'].unsqueeze(-1) # [seq_len, seq_len, 1(=bsz)]
#             print("perm_mask: ", perm_mask, type(perm_mask), perm_mask.dtype)
            target_mapping = \
                f32u['target_mapping'].unsqueeze(-1) # [num_predict, seq_len, 1(=bsz)]
#             print("target_mapping: ", target_mapping, type(target_mapping), target_mapping.dtype)
            inp_q = f32u['input_q'].unsqueeze(-1) # [seq_len, 1(=bsz)]
#             print("inp_q: ", inp_q, type(inp_q), inp_q.dtype)
            tgt_mask = f32u['target_mask'].unsqueeze(-1) # [num_predict, 1(=bsz)]
            ###############################################################################

            logits, new_mems = model(inp_k=inp_k, seg_id=seg_id, input_mask=None,
                  mems=mems, perm_mask=perm_mask,
                  target_mapping=target_mapping, inp_q=inp_q)

            #lm_loss = criterion(logits.transpose(1, 2), target).type(torch.float32)
            #####changed loss
            lm_loss = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(),
                                  target)
            tgt_mask_sum = tgt_mask.reshape(-1).sum()
            lm_loss_sum = (lm_loss * tgt_mask).reshape(-1).sum()

            optimizer.zero_grad()
            total_loss = lm_loss_sum / tgt_mask_sum
            print('Number of Epoch: %d in its %d Step' % ((num_epoch + 1), (num_step + 1)),
                  'cost =', '{:.6f}'.format(total_loss))
            num_step += 1

            total_loss.backward()
            optimizer.step()

            mems = new_mems
            
main()