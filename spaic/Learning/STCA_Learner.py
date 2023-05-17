# -*- coding: utf-8 -*-
"""
Created on 2020/11/9
@project: SPAIC
@filename: STCA_Learner
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description: 
"""
from .Learner import Learner

import torch
# from torch import fx


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
        ctx.thresh = thresh
        ctx.alpha = alpha
        ctx.save_for_backward(input)
        output = input.gt(thresh).type_as(input)
        return output

    @staticmethod
    def backward(
            ctx,
            grad_output
    ):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        ctx.alpha = ctx.alpha.to(input)
        temp = abs(input - ctx.thresh) < ctx.alpha  # 根据STCA，采用了sign函数
        result = grad_input * temp.type_as(input)
        return result, None, None

act_fun = ActFun()
def firing_func(x, v_th, alpha):
    return act_fun.apply(x, v_th, alpha)
# fx.wrap('firing_func')


class STCA(Learner):
    '''
        STCA learning rule.

        Args:
            alpha(num) : The parameter alpha of STCA learning model.
            preferred_backend(list) : The backend prefer to use, should be a list.
            name(str) : The name of this learning model. Should be 'STCA'.
            firing_func: The function of fire.

        Methods:
            build(self, backend): Build the backend, realize the algorithm of STCA model.
            threshold(self, x, v_th): Get the threshold of the STCA model.

        Example:
            Net._learner = STCA(0.5, Net)

        Reference:
            Pengjie Gu et al. “STCA: Spatio-Temporal Credit Assignment with Delayed Feedback in Deep SpikingNeural
            Networks.” In:Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence,
            IJCAI-19. International Joint Conferences on Artificial Intelligence Organization, July 2019,pp. 1366–1372.
            doi:10.24963/ijcai.2019/189.
            url:https://doi.org/10.24963/ijcai.2019/189.

    '''

    def __init__(self, trainable=None, **kwargs):
        super(STCA, self).__init__(trainable=trainable, **kwargs)

        self.alpha = kwargs.get('alpha', 0.5)
        self.prefered_backend = ['pytorch']
        self.name = 'STCA'
        self.firing_func = firing_func
        self.parameters = kwargs

    def build(self, backend):
        '''
            Build the backend, realize the algorithm of STCA model.

            Args：
                backend: The backend we used to compute.

        '''
        super(STCA, self).build(backend)
        self.device = backend.device0
        if backend.backend_name == 'pytorch':
            import torch
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
                    ctx.thresh = thresh
                    ctx.alpha = alpha
                    ctx.save_for_backward(input)
                    output = input.gt(thresh).type_as(input)
                    return output

                @staticmethod
                def backward(
                        ctx,
                        grad_output
                ):
                    input, = ctx.saved_tensors
                    grad_input = grad_output.clone()
                    temp = abs(input - ctx.thresh) < ctx.alpha  # 根据STCA，采用了sign函数
                    result = grad_input * temp.type_as(input)
                    return result, None, None

            self.firing_func = ActFun()
            self.alpha = torch.tensor(self.alpha).to(self.device)
            # self.backend.basic_operate['threshold'] = self.threshold

        backend_threshold = {'pytorch': self.torch_threshold}
        # replace threshold operation in all trainable neuron_groups
        for neuron in self.trainable_groups.values():
            for key in neuron._operations.keys():
                if 'threshold' in key:
                    # 这一步直接替换了神经元模型中的电压与阈值比较的计算
                    neuron._operations[key].func = backend_threshold[backend.backend_name]



    def torch_threshold(self, x, v_th):
        '''
            Get the threshold of the STCA model.

            return:
                A method that use STCA model to compute the threshold.

        '''
        return firing_func(x, v_th, self.alpha)

Learner.register('stca', STCA)

