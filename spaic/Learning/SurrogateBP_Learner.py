# encoding: utf-8
"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: SurrogateBP_Learner.py
@time:2022/8/12 17:34
@description:
"""
from .Learner import Learner
from .surrogate import *
import torch

class SurrogateBP(Learner):
    '''
        SurrogateBP learning rule.

        Args:
            alpha(num) : The parameter alpha of SurrogateBP learning model.
            preferred_backend(list) : The backend prefer to use, should be a list.
            name(str) : The name of this learning model. Should be 'sbp'.
            surrogate_func: The function of surrogate grad.

        Methods:
            build(self, backend): Build the backend, realize the algorithm of SurrogateBP model.
            threshold(self, x, v_th): Get the threshold of the SurrogateBP model.

        Example:
            Net._learner = Learner(trainable=Net, algorithm='sbp', surrogate_func=AtanGrad, alpha=2.0)


    '''

    def __init__(self, trainable=None, **kwargs):
        super(SurrogateBP, self).__init__(trainable=trainable)

        self.alpha = kwargs.get('alpha', 2)
        self.prefered_backend = ['pytorch']
        self.name = 'sbp'
        surrogate_func = kwargs.get('surrogate_func', AtanGrad)
        self.surrogate_func = surrogate_func(self.alpha, requires_grad=False)
        self.parameters = kwargs

    def build(self, backend):
        '''
            Build the backend, realize the algorithm of SurrogateBP model.

            Args：
                backend: The backend we used to compute.

        '''
        super(SurrogateBP, self).build(backend)
        self.device = backend.device0
        if backend.backend_name == 'pytorch':
            self.alpha = torch.tensor(self.alpha).to(self.device)

        backend_threshold = {'pytorch': self.torch_threshold}
        # replace threshold operation in all trainable neuron_groups
        for neuron in self.trainable_groups.values():
            for key in neuron._operations.keys():
                if 'threshold' in key:
                    # 这一步直接替换了神经元模型中的电压与阈值比较的计算
                    neuron._operations[key].func = backend_threshold[backend.backend_name]



    def torch_threshold(self, x, v_th):
        '''
            Get the threshold of the SurrogateBP model.

            return:
                A method that use SurrogateBP model to compute the threshold.

        '''
        return self.surrogate_func(x-v_th)

Learner.register('sbp', SurrogateBP)

