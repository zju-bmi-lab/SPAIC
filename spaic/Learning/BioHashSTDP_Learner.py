# -*- coding: utf-8 -*-
"""
Created on 2021/8/3
@project: SPAIC
@filename: BioHashSTDP_Learner
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description: 
"""
from .Learner import Learner
import torch



class BioHash(Learner):

    def __init__(self, trainable=None, **kwargs):
        super(BioHash, self).__init__(trainable=trainable)
        self.name = "biohash"
        self.prefered_backend = ['pytorch']
        self.trace_decay = kwargs.get('trace_decay', 0.999)
        self.lr_pos = kwargs.get('lr_pos', 1.0)
        self.lr_neg = kwargs.get('lr_neg', -0.05)
        self.neg_r = kwargs.get('neg_r', 3)

    def build(self, backend):
        super(BioHash, self).build(backend)
        for conn in self.trainable_connections.values():
            preg = conn.pre_assembly
            postg = conn.post_assembly
            pre_name = conn.get_input_name(preg, postg)
            post_name = conn.get_target_output_name(postg)
            weight_name = conn.get_link_name(preg, postg, 'weight')
            rank_name = post_name + '_{rank}'
            pre_trace_name = pre_name + '_{pre_trace}'
            post_trace_name = post_name + '_{post_trace}'
            rank_value = torch.zeros_like(backend._variables[post_name])
            # rank_value[:,0] = self.lr_pos
            for ii in range(self.neg_r):
                rank_value[:,ii] = self.lr_pos*(0.5)**ii
            rank_len = len(rank_value)-self.neg_r
            for ii in range(rank_len):
                rank_value[:,ii+self.neg_r] = self.lr_neg*(0.9)*ii

            backend.add_variable(pre_trace_name, backend._variables[pre_name].shape, value=0.0)
            backend.add_variable(post_trace_name, backend._variables[post_name].shape, value=0.0)
            backend.add_variable(rank_name, shape=rank_value.shape, value=rank_value)

            backend.add_operation([post_trace_name, self.update_STDP, [post_name, pre_trace_name, post_trace_name, weight_name, rank_name]])
            backend.add_operation([pre_trace_name, self.update_pre_trace, pre_name, pre_trace_name])





    def update_STDP(self, post:torch.Tensor, pre_trace:torch.Tensor, post_trace:torch.Tensor, weight:torch.Tensor, rank_weight:torch.Tensor):

        with torch.no_grad():
            rank = torch.argsort(post_trace, dim=-1)
            dw = torch.mean(torch.gather(rank_weight, -1, rank).unsqueeze(dim=2) * (
                        pre_trace.unsqueeze(dim=1) - post_trace.unsqueeze(dim=2) * (weight+0.01*torch.sign(weight)).unsqueeze(dim=0)),
                            dim=0) - self.lr_pos*1.0e-15*weight*torch.mean(post_trace,dim=0).unsqueeze(dim=1)
            # print(torch.mean(pre_trace), torch.mean(post_trace))
            weight.add_(dw)
            # print(torch.max(dw).item())
            post_trace = self.trace_decay*post_trace + (1-self.trace_decay)*post
        return post_trace

    def update_pre_trace(self, pre, pre_trace):
        pre_trace = self.trace_decay*pre_trace + (1-self.trace_decay)*pre
        return pre_trace






Learner.register('biohash', BioHash)
