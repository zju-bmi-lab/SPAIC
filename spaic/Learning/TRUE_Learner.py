# -*- coding: utf-8 -*-
"""
Created on 2021/4/6
@project: SPAIC
@filename: TRUE_Leaner
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
Temporal-Rate Unified Efficient Spike Backpropgation Learning rule
"""
import spaic
from .Learner import Learner
import torch
import torch.nn.functional as F

class TRUE_SpikeProp(Learner):
    '''
    Temporal Rate Unified Efficient learning rule.
    '''

    def __init__(self, trainable=None, **kwargs):
        super(TRUE_SpikeProp, self).__init__(trainable=trainable)
        self.alpha = kwargs.get('alpha', 1.0)
        self.beta = kwargs.get('beta', 0.3)
        self.prefered_backend = ['pytorch']
        self.name = 'TRUE'
        self.firing_func = None
        self.decay = 0.999
        self.parameters = kwargs
        self.net = trainable



    def build(self, backend):
        super(TRUE_SpikeProp, self).build(backend)
        self.device = backend.device
        self.dt = backend.dt
        self.running_mean = None
        self.running_var = None
        if self.net is not None:
            neurons = self.net.get_groups()
        else:
            neurons = []
        for n in neurons:
            if isinstance(n, spaic.NeuronGroup) and (n.model_name=='slif' or n.model_name=='selif'):
                n.model.attach_learner(self)


        if backend.backend_name == 'pytorch':

            class ActFun(torch.autograd.Function):
                """
                Approximate firing func.
                """

                @staticmethod
                def forward(ctx,
                        x,
                        dx,
                        thresh,
                        alpha,
                        beta,
                        dt
                ):
                    # px = torch.clamp_min(dx, 0)
                    ctx.beta = beta
                    bx = torch.clamp_min(dx, 1.0e-6)
                    # x = x + 0.05*max(1-beta,0)*torch.randn_like(x)
                    out_i = (x+bx*dt).gt(thresh)
                    out_t = out_i*(dt - torch.clamp_max(torch.clamp_min(thresh-x, 0)/bx, dt))
                    # prob = torch.sigmoid(beta*px)
                    # out_i = tmp
                    sub_out = torch.clamp(x+alpha-thresh, 0, alpha)/alpha


                    ctx.save_for_backward(bx, out_i, sub_out)
                    out = torch.stack((out_i, out_t), dim=1)
                    return out



                @staticmethod
                def backward(
                        ctx,
                        grad_out):
                    grad_i = grad_out[:,0,...]
                    grad_t = grad_out[:,1,...]
                    dx, tmp, sub_out = ctx.saved_tensors

                    # grad_i = torch.clamp_min(grad_i,0)*torch.exp(-(0.09*r+0.1)) + torch.clamp_max(grad_i,0)*torch.exp(-0.1*r)
                    # grad_t = torch.clamp_min(grad_t,0)/(0.9*dx+0.15) + torch.clamp_max(grad_t,0)/(dx+0.1)

                    v_grad = ctx.beta*(tmp*grad_t/dx + sub_out*grad_i)


                    #result = prob_i*grad_i + prob_t*grad_t/ (10*torch.clamp_min(dx, 0) + 0.1) torch.exp(-(100.0*dx/ctx.thresh) **2)

                    return v_grad, None, None, None, None, None


            self.firing_func = ActFun()
            # self.backend.basic_operate['threshold'] = self.threshold

            # replace threshold operation in all trainable neuron_groups
            # for neuron in self.trainable_groups.values():
            #     for key in neuron._operations.keys():
            #         if 'threshold' in key:
            #             neuron._operations[key][1] = self.threshold


    def threshold(self, x, dx, v_th):

        output = self.firing_func.apply(x, dx, v_th, self.alpha, self.beta*self.dt, self.dt)
        return output

Learner.register("true_spikeprop", TRUE_SpikeProp)
