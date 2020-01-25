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

import numpy as np
import mpu
import torch.nn.init as init


import torch
import torch.nn as nn
import torch.nn.functional as F
from mpu import copy_to_model_parallel_region,VocabUtility,get_model_parallel_rank,get_model_parallel_world_size,ColumnParallelLinear ,RowParallelLinear,divide,split_tensor_along_last_dim


class ParallelAttntn(torch.nn.Module):
    
    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob,proj_size,
                 init_method=None, output_layer_init_method=None):
        super(ParallelAttntn, self).__init__()
        ##############
        self.d_head=hidden_size
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Per attention head and per partition values.
        world_size = get_model_parallel_world_size()
        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads,
                                                        world_size)
        # Strided linear layer.
        self.q_proj_weight = nn.Parameter(torch.randn(*proj_size))
        self.k_proj_weight = nn.Parameter(torch.randn(*proj_size))
        self.v_proj_weight = nn.Parameter(torch.randn(*proj_size))
        self.r_proj_weight = nn.Parameter(torch.randn(*proj_size))
        self.proj_o = nn.Parameter(torch.randn(*proj_size))
        self.key = ColumnParallelLinear(hidden_size, hidden_size,
                                                    stride=1,
                                                    gather_output=False,
                                                    init_method=init_method)
        self.val = ColumnParallelLinear(hidden_size, hidden_size,
                                            stride=1,
                                            gather_output=False,
                                            init_method=init_method)
        self.q = ColumnParallelLinear(hidden_size, hidden_size,
                                                    stride=1,
                                                    gather_output=False,
                                                    init_method=init_method)
        self.r = ColumnParallelLinear(hidden_size, hidden_size,
                                                    stride=1,
                                                    gather_output=False,
                                                    init_method=init_method)
        self.g = ColumnParallelLinear(hidden_size, hidden_size,
                                                    stride=1,
                                                    gather_output=False,
                                                    init_method=init_method)
        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)
        self.layer_norm = nn.LayerNorm(proj_size[0])
        # Output.
        self.dense = RowParallelLinear(hidden_size,
                                       hidden_size,
                                       input_is_parallel=True,
                                       init_method=output_layer_init_method)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)
        
    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing."""

        # post-attention projection (back to `d_model`) changed to self.dense instead of proj_o
        attn_vec = self.dense(attn_vec)
        attn_out = torch.einsum('ibnd,hnd->ibh', attn_vec, self.proj_o)

        attn_out = self.output_dropout(attn_out)
        if residual:
            output = self.layer_norm(attn_out + h)
        else:
            output = self.layer_norm(attn_out)

        return output

    def rel_shift(self, x, klen=-1):
        """perform relative shift to form the relative attention score."""
        x_size = x.shape

        x = torch.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
        x = x[1:, 0:, 0:, 0:] # tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
        x = torch.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
        x = x[0:, 0:klen, 0:, 0:] # tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])

        return x
    def head_projection(self, h, name):
        """Project hidden states to a specific head with a 4D-shape."""
        proj_weight = None
        if name == 'q':
            proj_weight = self.q_proj_weight
        elif name == 'k':
            proj_weight = self.k_proj_weight 
        elif name =='v':
            proj_weight = self.v_proj_weight 
        elif name == 'r':
            proj_weight = self.r_proj_weight
        else:
            raise ValueError('Unknown `name` {}.'.format(name))
        h = h.to(torch.cuda.current_device())

        head = torch.einsum('ibh,hnd->ibnd', h, proj_weight)

        return head

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                      r_w_bias, r_r_bias, r_s_bias, attn_mask, scale):

        """Core relative positional attention operations."""

        # content based attention score
        print("QQQQQQQQQQQQ",q_head.shape,r_w_bias.shape,k_head_h.shape)
        ac = torch.einsum('ibnd,jbnd->ijbn', q_head + r_w_bias, k_head_h)

        # position based attention score
        bd = torch.einsum('ibnd,jbnd->ijbn', q_head + r_r_bias, k_head_r)
        bd = self.rel_shift(bd, klen=ac.shape[1])

        # segment based attention score
        if seg_mat is None:
            ef = 0
        else:
            ef = torch.einsum('ibnd,snd->ibns', q_head + r_s_bias, seg_embed)
            seg_mat = seg_mat.to(torch.cuda.current_device())
            ef = torch.einsum('ijbs,ibns->ijbn', seg_mat, ef)

        # merge attention scores and perform masking
        attn_score = (ac + bd + ef) * scale
        if attn_mask is not None:
            # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
            attn_score = attn_score - 1e30 * attn_mask

        # attention probability
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.output_dropout(attn_prob)

        # attention output
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)

        return attn_vec

    
    
    def forward(self, h, g, r, mems, r_w_bias, r_r_bias, seg_mat, r_s_bias,
                            seg_embed, attn_mask_h, attn_mask_g, target_mapping):
        
        scale = 1 / (self.d_head ** 0.5)
        #h,g,r ([512, 1, 32], [85, 1, 32], [1024, 1, 32]
        # content based attention score
        if mems is not None and len(mems.size()) > 1:
            cat = torch.cat([mems, h], dim=0)
        else:
            cat = h
        print(cat.shape,"CAT",self.key.weight.shape)
        # content-based key head
        k_head_h =self.head_projection(cat, 'k')
        k_head_h = self.key(k_head_h)
        
        #v_k_head= torch.einsum('ibh,hnd->ibnd',cat, self.key_value)
        # content-based value head
        v_head_h =  self.head_projection(cat, 'v')
        v_head_h = self.val(v_head_h)

        #k_head,v_head = split_tensor_along_last_dim(v_k_head,2)
        # position-based key head
        k_head_r = self.head_projection(r, 'r')
        k_head_r = self.r(k_head_r)

        ##### h-stream
        # content-stream query head
        q_head_h =self.head_projection(h, 'q')
        q_head_h = self.q(q_head_h)



        # core attention ops
        # hˆ(m)_zt = LayerNorm(h^(m-1)_zt + RelAttn(h^(m-1)_zt + [h~^(m-1), hT(m-1)_z<=t]))
        attn_vec_h = self.rel_attn_core(
            q_head_h, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat, r_w_bias,
            r_r_bias, r_s_bias, attn_mask_h, scale)

        # post processing
        output_h = self.post_attention(h, attn_vec_h)

        ##### g-stream
        # query-stream query head
        q_head_g =  torch.einsum('ibh,hnd->ibnd', g,self.q_proj_weight)#self.head_projection(g, 'q')
        q_head_g = self.g(q_head_g)

        # core attention ops
        # gˆ(m)_zt = LayerNorm(g^(m-1)_zt + RelAttn(g^(m-1)_zt + [h~^(m-1), hT(m-1)_z<=t]))
        if target_mapping is not None:
            q_head_g = torch.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
            attn_vec_g = self.rel_attn_core(
                q_head_g, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat, r_w_bias,
                r_r_bias, r_s_bias, attn_mask_g, scale)
            attn_vec_g = torch.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
        else:
            attn_vec_g = self.rel_attn_core(
                q_head_g, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat, r_w_bias,
                r_r_bias, r_s_bias, attn_mask_g, scale)

        # post processing
        output_g = self.post_attention(g, attn_vec_g)
#         print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
#         print("b4 q_head_g", q_head_g.shape,q_head_g.device)
#         print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")        
#         print("#############################################")
#         print("k_head_h", k_head_h.shape,k_head_h.device)
#         print("v_head_h",v_head_h.shape,v_head_h.device)
#         print("k_head_r", k_head_r.shape,k_head_r.device)
#         print("q_head_h",q_head_h.shape,q_head_h.device)
#         print("output_h", output_h.shape,output_h.device)
#         print("Segembed",seg_embed.shape,seg_embed.device)
#         print("Seg mat",seg_mat.shape,seg_mat.device)    
#         print("afta q_head_g", q_head_g.shape,q_head_g.device)
#         print("attn_vec_g", attn_vec_g.shape,attn_vec_g.device)
#         print("output_g", output_g.shape,output_g.device)
#         print("############################################")
#         print("#############################################")
#         print("cat", cat.shape,cat.device)
# #         print("mems", mems.shape,mems.device)
#         print("h", h.shape,h.device)
#         print("r", r.shape,r.device)
#         print("Segembed",seg_embed.shape,seg_embed.device)
#         print("target_mapping",target_mapping.shape,target_mapping.device)
#         print("############################################")
        
        return output_h, output_g



class XLNetParallelMLP(torch.nn.Module):
    """MLP for XLNet.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Arguments:
        hidden_size: The hidden size of the self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layer initialization. If None,
                                  use `init_method`.
    """
# changed init method from a compulosory to optional arg
    def __init__(self, hidden_size, output_dropout_prob, init_method=None,
                 output_layer_init_method=None):
        super(XLNetParallelMLP, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Project to 4h.
        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, 4*hidden_size,
                                                  gather_output=False,
                                                  init_method=init_method)
        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            4*hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method)
        self.dropout = torch.nn.Dropout(output_dropout_prob)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, hidden_states):
        # [b, s, 4hp]
        
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        
        ####added dropout here  and removed dropout on output changed act to relu
        
        intermediate_parallel = self.relu(intermediate_parallel)
        intermediate_parallel = self.dropout(intermediate_parallel)

        # [b, s, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        
        
        return output


class XLNet(nn.Module):
    """
        Defines a Transformer-XL computation graph with additional
        support for XLNet.

        Args:

        inp_k: int32 Tensor in shape [len, bsz], the input token IDs.
        seg_id: int32 Tensor in shape [len, bsz], the input segment IDs.
        input_mask: float32 Tensor in shape [len, bsz], the input mask.
          0 for real tokens and 1 for padding.
        mems: a list of float32 Tensors in shape [mem_len, bsz, d_model], memory
          from previous batches. The length of the list equals n_layer.
          If None, no memory is used.
        perm_mask: float32 Tensor in shape [len, len, bsz].
          If perm_mask[i, j, k] = 0, i attend to j in batch k;
          if perm_mask[i, j, k] = 1, i does not attend to j in batch k.
          If None, each position attends to all the others.
        target_mapping: float32 Tensor in shape [num_predict, len, bsz].
          If target_mapping[i, j, k] = 1, the i-th predict in batch k is
          on the j-th token.
          Only used during pretraining for partial prediction.
          Set to None during finetuning.
        inp_q: float32 Tensor in shape [len, bsz].
          1 for tokens with losses and 0 for tokens without losses.
          Only used during pretraining for two-stream attention.
          Set to None during finetuning.

        n_layer: int, the number of layers.
        d_model: int, the hidden size.
        n_head: int, the number of attention heads.
        d_head: int, the dimension size of each attention head.
        d_inner: int, the hidden size in feed-forward layers.
        ff_activation: str, "relu" or "gelu".
        n_token: int, the vocab size.

        dropout: float, dropout rate.
        dropatt: float, dropout rate on attention probabilities.

        mem_len: int, the number of tokens to cache.
        reuse_len: int, the number of tokens in the currect batch to be cached
          and reused in the future.
        bi_data: bool, whether to use bidirectional input pipeline.
          Usually set to True during pretraining and False during finetuning.
        clamp_len: int, clamp all relative distances larger than clamp_len.
          -1 means no clamping.

      """
    def __init__(self, n_token, n_layer, n_head, d_head, d_inner, d_model, dropout, dropatt,
                 attn_type, bi_data, clamp_len, same_length, reuse_len, mem_len):
        super(XLNet, self).__init__()
        ######changeXX
        #torch.cuda.set_device(torch.cuda.current_device())
        ###########################3
        self.n_token = n_token
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.d_model = d_model
        self.dropout = dropout
        self.dropatt = dropatt
        self.attn_type = attn_type
        self.bi_data = bi_data
        self.clamp_len = clamp_len
        self.same_length = same_length
        self.reuse_len = reuse_len
        self.mem_len = mem_len

        self.embedding = mpu.VocabParallelEmbedding(n_token, d_model)
        self.Dropout = nn.Dropout(p=dropout)
        self.DropAttn = nn.Dropout(p=dropatt)
        world_size = get_model_parallel_world_size()
        self.dhead_size_per_partition = divide(self.d_head, world_size)

        self.r_w_bias = nn.Parameter(torch.randn(self.n_layer,
                                                  self.n_head,self.dhead_size_per_partition))
        self.r_r_bias = nn.Parameter(torch.randn(self.n_layer,
                                                  self.n_head,self.dhead_size_per_partition))

        ##### Segment embedding
        self.r_s_bias = nn.Parameter(torch.randn(self.n_layer,
                                                  self.n_head,self.dhead_size_per_partition))

        self.seg_embed = nn.Parameter(torch.randn(self.n_layer, 2,
                                                   self.n_head, self.dhead_size_per_partition))
        
        self.mask_emb = nn.Parameter(torch.randn(1, 1, d_model))

        # post-attention projection (back to `d_model`)
        self.proj_o = nn.Parameter(torch.randn(self.d_model,
                                                self.n_head, self.d_head))

        #### Project hidden states to a specific head with a 4D-shape.
        self.q_proj_weight = nn.Parameter(torch.randn(self.d_model,
                                                       self.n_head, self.d_head))
        self.k_proj_weight = nn.Parameter(torch.randn(self.d_model,
                                                       self.n_head, self.d_head))
        self.v_proj_weight = nn.Parameter(torch.randn(self.d_model,
                                                       self.n_head, self.d_head))
        self.r_proj_weight = nn.Parameter(torch.randn(self.d_model,
                                                       self.n_head, self.d_head))
        
        hidden_size=(self.d_model,self.n_head ,self.d_head)
        self.two_stream_rel_attn = ParallelAttntn(self.d_head,self.n_head,dropatt,dropout,hidden_size)
        self.layer_norm = nn.LayerNorm(d_model)
        #change
        self.positionwise_ffn = XLNetParallelMLP(
            d_model,
            dropout)

        self.conv1 = nn.Linear(d_model, d_inner)
        self.conv2 = nn.Linear(d_inner, d_model)
        self.relu = nn.ReLU(inplace=True)
        ##changes : parallelised bias
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.n_token, get_model_parallel_rank(),
                get_model_parallel_world_size())
        self.n_token_per_partition = self.vocab_end_index - \
                                            self.vocab_start_index
        self.softmax_b = nn.Parameter(torch.zeros(self.n_token_per_partition))
                



    def gelu(self, x):
        """Gaussian Error Linear Unit.

        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415
        Args:
          x: float Tensor to perform activation.

        Returns:
          `x` with the GELU activation applied.
        """
        cdf = 0.5 * (1.0 + torch.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))))
        return x * cdf



#     def positionwise_ffn(self, inp, activation_type='relu'):

#         """Position-wise Feed-forward Network."""
#         output = self.conv1(inp)
#         output = self.Dropout(output)
#         if activation_type == 'relu':
#             output = self.relu(output)
#         elif activation_type == 'gelu':
#             output = self.gelu(output)
#         else:
#             raise ValueError('Unsupported activation type {}'.format(activation_type))

#         output = self.layer_norm(output + inp)
#         return output




    def _create_mask(self, qlen, mlen, dtype, same_length=False):
        """create causal attention mask."""
        # [[0,1,1],
        #  [0,0,1],
        #  [0,0,0]]
        attn_mask = torch.ones([qlen, qlen], dtype=dtype)
        mask_u = torch.triu(attn_mask) # Upper triangular part.
        mask_dia = torch.tril(attn_mask) & torch.triu(attn_mask) # Diagonal. Figure 2(c)
        attn_mask_pad = torch.zeros([qlen, mlen], dtype=dtype)
        ret = torch.cat([attn_mask_pad, mask_u - mask_dia], dim=1) # [qlen, mlen]
        if same_length:
            # [[0,1,1],
            #  [1,0,1],
            #  [1,1,0]]
            mask_l = torch.tril(attn_mask) # Lower triangular part.
            ret = torch.cat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], dim=1)

        return ret.type(dtype=torch.float32) # [qlen, qlen]

    def positional_embedding(self, pos_seq, inv_freq):
        sinusoid_inp = torch.einsum('i,d->id', pos_seq, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]

        return pos_emb

    def _cache_mem(self, curr_out, prev_mem, mem_len, reuse_len=None):
        """cache hidden states into memory."""

        with torch.no_grad():
            if mem_len is None or mem_len == 0:
                return None
            else:
                if reuse_len is not None and reuse_len > 0:
                    curr_out = curr_out[:reuse_len]

                if prev_mem is None:
                    new_mem = curr_out[-mem_len:]
                else:
                    new_mem = torch.cat([prev_mem, curr_out], dim=0)[-mem_len:]

            return new_mem


    def relative_positional_encoding(self, qlen, klen, d_model, clamp_len, attn_type,
                                     bi_data, bsz=None, dtype=None):
        """create relative positional encoding."""

        freq_seq = torch.arange(0, d_model, 2.0)
        if dtype is not None and dtype != torch.float32:
            freq_seq = freq_seq.type(dtype)
        inv_freq = 1 / (10000 ** (freq_seq / d_model))

        if attn_type == 'bi':
            # beg, end = klen - 1, -qlen
            beg, end = klen, -qlen
        elif attn_type == 'uni':
            # beg, end = klen - 1, -1
            beg, end = klen, -1
        else:
            raise ValueError('Unknown `attn_type` {}.'.format(attn_type))

        if bi_data and bsz%2 is 0:
            fwd_pos_seq = torch.arange(beg, end, -1.0)
            bwd_pos_seq = torch.arange(-beg, -end, 1.0)

            if dtype is not None and dtype != torch.float32:
                fwd_pos_seq = fwd_pos_seq.type(dtype=dtype)
                bwd_pos_seq = bwd_pos_seq.type(dtype=dtype)

            if clamp_len > 0:
                fwd_pos_seq = torch.clamp(fwd_pos_seq, -clamp_len, clamp_len)
                bwd_pos_seq = torch.clamp(bwd_pos_seq, -clamp_len, clamp_len)

            fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
            bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1)
        else:
            fwd_pos_seq = torch.arange(beg, end, -1.0)
            if dtype is not None and dtype != torch.float32:
                fwd_pos_seq = fwd_pos_seq.type(dtype=dtype)
            if clamp_len > 0:
                fwd_pos_seq = torch.clamp(fwd_pos_seq, -clamp_len, clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)

        return pos_emb

    def forward(self, inp_k, seg_id, input_mask, mems, perm_mask, target_mapping, inp_q):
        new_mems = []
        bsz = inp_k.shape[1]
        qlen = inp_k.shape[0]
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        ##### Attention mask
        # causal attention mask
        if self.attn_type == 'uni':
            attn_mask = self._create_mask(qlen, mlen, torch.int64, self.same_length)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == 'bi':
            attn_mask = None
        else:
            raise ValueError('Unsupported attention type: {}'.format(self.attn_type))

        # data mask: input mask & perm mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            mems_mask = torch.zeros([data_mask.shape[0], mlen, bsz],
                                 dtype=torch.float32)
            mems_mask = mems_mask.to(torch.cuda.current_device())
            data_mask = torch.cat([mems_mask, data_mask], dim=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

#         print("attn_mask_BEFORE: ", attn_mask)
        
        if attn_mask is not None:
            attn_mask = attn_mask.gt(0).type(torch.float32)
#             print("attn_mask_AFTER: ", attn_mask)

        if attn_mask is not None:
            non_tgt_mask = -torch.eye(qlen, dtype=torch.float32) # [qlen, qlen]
            non_tgt_mask = torch.cat([torch.zeros([qlen, mlen], dtype=torch.float32), # [qlen, klen]
                                        non_tgt_mask],
                                        dim=-1)
            #changingggggg
            non_tgt_mask = non_tgt_mask.to(torch.cuda.current_device())
            non_tgt_mask = (attn_mask +
                            non_tgt_mask[:, :, None, None]).gt(0).type(dtype=torch.float32)
        else:
            non_tgt_mask = None

        ##### Word embedding
        lookup_table = self.embedding
        word_emb_k = lookup_table(inp_k)
        print(inp_k.shape,word_emb_k.shape,lookup_table.weight.shape,"a bC")


        if inp_q is not None:
            if target_mapping is not None:
                word_emb_q = self.mask_emb.repeat(target_mapping.shape[0], bsz, 1)
            else:
                inp_q_ext = inp_q[:, :, None]
                word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k

        #### Figure 2(a), Content Stream(Original Attention), h^(0)_t = e(x_i) = e(inp_k)
        output_h = self.Dropout(word_emb_k)
        if inp_q is not None:
            #### Query Stream, g^(0)_t = w
            #### the first layer query stream is initialized with a trainable vector
            output_g = self.Dropout(word_emb_q)

        ##### Segment embedding
        # paper
        # Given a pair of positions i and j in the sequence, if
        # i and j are from the same segment
        if seg_id is not None:
            # Convert `seg_id` to one-hot `seg_mat`
            mem_pad = torch.zeros([mlen, bsz], dtype=torch.int32)
            mem_pad = mem_pad.to(torch.cuda.current_device())

            cat_ids = torch.cat([mem_pad, seg_id], dim=0)

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = (~torch.eq(seg_id[:, None], cat_ids[None, :])).type(torch.long)
#             print("seg_mat_BEFORE: ", seg_mat, seg_mat.shape, seg_mat.type, seg_mat.device)
            seg_mat = torch.eye(2, dtype=torch.float32)[seg_mat]
#             print("seg_mat_AFTER: ", seg_mat, seg_mat.shape, seg_mat.type, seg_mat.device)
        else:
            seg_mat = None

        ##### Positional encoding
        pos_emb = self.relative_positional_encoding(
            qlen, klen, self.d_model, self.clamp_len, self.attn_type, self.bi_data,
            bsz=bsz, dtype=torch.float32)
        pos_emb = self.Dropout(pos_emb)
        pos_emb= pos_emb.to(torch.cuda.current_device())

        ##### Attention layers
        if mems is None:
            mems = [None] * self.n_layer

        for i in range(self.n_layer):
            # cache new mems
            new_mems.append(self._cache_mem(output_h, mems[i], self.mem_len, self.reuse_len))
            # segment bias
            if seg_id is None:
                r_s_bias_i = None
                seg_embed_i = None
            else:
                r_s_bias_i = self.r_s_bias[i]
                seg_embed_i = self.seg_embed[i]

            if inp_q is not None:
                ioutput_h, ioutput_g = self.two_stream_rel_attn(
                    h=output_h,
                    g=output_g,
                    r=pos_emb,
                    r_w_bias= self.r_w_bias[i],
                    r_r_bias= self.r_r_bias[i],
                    seg_mat=seg_mat,
                    r_s_bias=r_s_bias_i,
                    seg_embed=seg_embed_i,
                    attn_mask_h=non_tgt_mask,
                    attn_mask_g=attn_mask,
                    mems=mems[i],
                    target_mapping=target_mapping)
            else:
                output_h = self.rel_multihead_attn(
                    h=output_h,
                    r=pos_emb,
                    r_w_bias=self.r_w_bias[i],
                    r_r_bias=self.r_r_bias[i],
                    seg_mat=seg_mat,
                    r_s_bias=r_s_bias_i,
                    seg_embed=seg_embed_i,
                    attn_mask=non_tgt_mask,
                    mems=mems[i])

            if inp_q is not None:
                output_g = self.positionwise_ffn(ioutput_g)
                output_g=self.layer_norm(output_g + ioutput_g)


            output_h = self.positionwise_ffn(ioutput_h)
            output_h=self.layer_norm(output_h + ioutput_h)

        if inp_q is not None:
            output = self.Dropout(output_g)
        else:
            output = self.Dropout(output_h)
#         print(output.shape,output.device,"output",self.softmax_b.shape,self.softmax_b.device,"softmax")
#         print("lookup table",lookup_table.weight.shape,lookup_table.weight.device)
        #-----------------------srs----------------------------------------
        #lookup table weights are still parallel shhape is [15261,32]  softmax [30522]
#         # output is [85,1,32]
#         print("------------")
#         print(lookup_table.weight.t().shape,"-----------------------",sep="\n")
        ##option a call an allgather or all reduce on lookup weights or b parallelise output
        #Ssoftmax_b = copy_to_model_parallel_region( self.softmax_b)
#         print("####################################","output",output,"################################",sep="\n")
#         print("####################################","lookup",lookup_table.weight,"################################",sep="\n")
        logits =  torch.einsum('ibd,nd->ibn', output, lookup_table.weight) + self.softmax_b
        #torch.matmul(output,lookup_table.weight.t())
        
        return logits, new_mems