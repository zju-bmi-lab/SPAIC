# -*- coding: utf-8 -*-
"""
Created on 2021/3/30
@project: SPAIC
@filename: STBP_Learner
@author: Mengxiao Zhang
@contact: mxzhangice@gmail.com

@description:
"""
from .Learner import Learner


class STBP(Learner):
    '''
        STBP learning rule.

        Args:
            alpha(num) : The parameter alpha of STBP learning model.
            trainable : The parameter whether it can be trained.

        Methods:
            build(self, backend): Build the backend, realize the algorithm of STBP model.
            threshold(self, x, v_th): Get the threshold of the STBP model.

        Example:
            Net._learner = STBP(0.5, Net)

        Reference:
            Yujie Wu et al. "Spatio-Temporal Backpropagation for Training High-performance Spiking Neural Networks" In:
            Frontiers in Neuroscience, 2018. Volume 12. pp. 331.
            doi:10.3389/fnins.2018.00331
            url:ttps://www.frontiersin.org/article/10.3389/fnins.2018.00331

    '''

    def __init__(self,trainable=None, **kwargs):
        super(STBP, self).__init__(trainable=trainable, **kwargs)
        self.alpha = kwargs.get('alpha', 0.5)
        self.prefered_backend = ['pytorch']
        self.name = 'STBP'
        self.firing_func = None
        self.parameters = kwargs

    def build(self, backend):
        '''
            Build the backend, realize the algorithm of STBP model.
            Args：
                backend: The backend we used to compute.
        '''
        super(STBP, self).build(backend)
        self.device = backend.device0
        if backend.backend_name == 'pytorch':
            import torch
            import math
            class ActFun(torch.autograd.Function):
                """
                Approximate firing func.
                """
                @staticmethod
                def forward(
                        ctx,
                        input,
                        thresh,
                        alpha
                ):
                    import torch
                    import math
                    ctx.thresh = thresh
                    ctx.alpha = alpha
                    ctx.save_for_backward(input)
                    return input.gt(thresh).type_as(input)

                @staticmethod
                def backward(
                        ctx,
                        grad_output
                ):
                    input, = ctx.saved_tensors
                    grad_input = grad_output.clone()
                    ctx.alpha = ctx.alpha.to(input)
                    temp = torch.exp(-(input - ctx.thresh) ** 2 / (2 * ctx.alpha)) \
                           / (2 * math.pi * ctx.alpha)
                    result = grad_input * temp.type_as(input)
                    return result, None, None


            self.firing_func = ActFun()
            self.alpha = torch.tensor(self.alpha).to(self.device)
            # self.backend.basic_operate['threshold'] = self.threshold

            # replace threshold operation in all trainable neuron_groups
            for neuron in self.trainable_groups.values():
                for key in neuron._operations.keys():
                    if 'threshold' in key:
                        neuron._operations[key].func = self.threshold


    def threshold(self, x, v_th):
        '''
            Get the threshold of the STBP model.

            return:
                A method that use STBP model to compute the threshold.

        '''
        return self.firing_func.apply(x, v_th, self.alpha)


Learner.register("stbp", STBP)