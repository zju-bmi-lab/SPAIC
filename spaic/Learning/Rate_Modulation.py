# -*- coding: utf-8 -*-
"""
Created on 2021/6/1
@project: SPAIC
@filename: Rate_Modulation
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description: 
"""

from .Learner import Learner
from ..Backend.Backend import Backend
import torch

class Rate_Modulate(Learner):
    '''
        neuron spiking rate modulation
    '''
    def __init__(self,trainable=None, **kwargs): #mean_rate, modulate_effect, ave_rate, trainable=None, post_var_name='O'):
        
        super(Rate_Modulate, self).__init__(trainable)
        self.m_r = kwargs.get("m_r")
        self.alpha = kwargs.get("alpha",0.001)
        # self.beta = kwargs.get("beta",0.9)
        self.trainable = trainable
        self.prefered_backend = ['pytorch']
        self.name = 'RateMode'
        self.post_var_name = kwargs.get("post_var_name", 'O')

        self.modulations = []


    def build(self, backend: Backend):
        super(Rate_Modulate, self).build(backend)
        ind = 0
        for key, con in self.trainable_connections.items():
            for unit_con in con.unit_connections:
                pre_group = unit_con[0]
                post_group = unit_con[1]
                weight = backend._parameters_dict[con.get_weight_name(pre_group, post_group)]
                if isinstance(self.m_r, list):
                    m_r = self.m_r[ind]
                else:
                    m_r = self.m_r
                if isinstance(self.alpha, list):
                    alpha = self.alpha[ind]
                else:
                    alpha = self.alpha
                if isinstance(self.post_var_name, list):
                    post_var_name = self.post_var_name[ind]
                else:
                    post_var_name = self.post_var_name

                output_name = post_group.id + ':' + '{' + post_var_name + '}'
                method = Modulation_method(weight, m_r, alpha, output_name)
                self.modulations.append(method)
                self.op_to_backend(None, method.record_rate, output_name)
                self.init_op_to_backend(None, method.modulate_step, [])
            ind += 1

    @property
    def rate_loss(self):
        loss = 0
        for mod in self.modulations:
            if mod.rate_loss is None:
                return None
            else:
                loss = loss + mod.rate_loss
        return loss



Learner.register("rate_modulate", Rate_Modulate)



class Modulation_method():

    def __init__(self, weight,  m_r, alpha, var_name):
        super(Modulation_method, self).__init__()
        self.var_name = var_name
        self.rec_rate = None
        self.weight = weight
        self.m_r = m_r
        self.alpha = alpha
        self.init_rate = torch.zeros([1, weight.shape[0]], device=weight.device)

    def record_rate(self, out):
        if '{[2]' in self.var_name:
            self.rec_rate = self.rec_rate + out[:,0, ...]
        else:
            self.rec_rate = self.rec_rate + out


    def modulate_step(self):
        self.rec_rate = self.init_rate

    @property
    def rate_loss(self):
        if self.rec_rate is not None:
            rate_loss = self.alpha*torch.norm(self.rec_rate - self.m_r)
            return rate_loss
        else:
            return None











