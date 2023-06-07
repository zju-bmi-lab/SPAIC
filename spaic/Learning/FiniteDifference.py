# -*- coding: utf-8 -*-
"""
Created on 2022/5/30
@project: SPAIC
@filename: FiniteDifferenceGradientApproximation
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
A numerical analysis tool for approximating derivatives of networks using Finite Difference Method
"""
from ..Network import BaseModule
from ..Network.BaseModule import Op
from .Learner import Learner
from ..Backend.Backend import Backend
from ..Network.Topology import Connection
import torch

class FiniteDifference(Learner):
    
    def __init__(self, trainable=None, **kwargs):
        super(FiniteDifference, self).__init__(trainable)
        self.index = kwargs.get('index', None)
        self.epsilon = kwargs.get('epsilon', 0.001)
        self.target = kwargs.get('target', None)
        self.var_name = kwargs.get('var_name', None)
        assert isinstance(self.index, list)
        assert isinstance(self.index[0], list)
        assert len(trainable) == 1 and isinstance(trainable[0], Connection)
        assert isinstance(self.var_name, str)
        self.record_values = []



    def weight_change(self, weight):
        new_weights_pos = []
        new_weights_neg = []
        for ind in self.index:
            pos_weight = weight.clone()
            neg_weight = weight.clone()
            pos_weight[ind[0], ind[1]] += self.epsilon
            neg_weight[ind[0], ind[1]] -= self.epsilon
            new_weights_pos.append(pos_weight)
            new_weights_neg.append(neg_weight)

        new_weights_neg = torch.stack(new_weights_neg, dim=0).unsqueeze(1)
        new_weights_pos = torch.stack(new_weights_pos, dim=0).unsqueeze(1)
        new_weights = torch.stack([new_weights_pos, new_weights_neg], dim=0)
        return new_weights

    def record_target(self, value):
        self.record_values.append(value)

        pass

    def build(self, backend):
        assert isinstance(self.target, BaseModule)
        assert isinstance(backend, Backend)
        self._backend = backend
        self.var_name = self.target.get_full_name(self.var_name)
        self.op_to_backend(None, self.record_target, self.var_name)











