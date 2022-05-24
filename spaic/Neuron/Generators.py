# encoding: utf-8
"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: Generators.py
@time:2021/6/21 16:35
@description:
"""
from .Node import Node, Generator
import torch
import numpy as np

class Poisson_Generator(Generator):
    """
        泊松生成器，根据输入脉冲速率生成。
        Generate a poisson spike train according input rate.
        time: encoding window ms
        dt: time step
        HZ: cycles/s
    """
    def __init__(self, shape=None, num=None, dec_target=None,  dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), rate=None, **kwargs):

        super(Poisson_Generator, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.num = num
        # the unit of dt is 0.1ms, for each time step, the rate has to be multiplied by 1e-4
        self.unit_conversion = kwargs.get('unit_conversion', 0.1)
        self.weight = kwargs.get('weight', 1.0)
        if rate is not None:
            if hasattr(rate, '__iter__'):
                self.source = rate
            else:
                self.source = np.array([rate])
            self.new_input = True

        self.batch = kwargs.get('batch', 1)

    def torch_coding(self, source, device):
        # assert source.size == self.num, "The dimension of input data should be consistent with the number of input neurons."
        if source.size != self.num:
            import warnings
            warnings.warn("The dimension of input data should be consistent with the number of input neurons.")
        if source.__class__.__name__ == 'ndarray':
            source = torch.tensor(source, device=device)

        # if source.ndim == 0:
        #     batch = 1
        # else:
        #     batch = source.shape[0]

        # shape = [self.batch, self.num]
        spk_shape = [self.time_step] + list(self.shape)
        spikes = self.weight*torch.rand(spk_shape, device=device).le(source*self.unit_conversion).float()
        return spikes

Generator.register('poisson_generator', Poisson_Generator)

class Poisson_Generator2(Generator):
    """
        泊松生成器，根据输入脉冲速率生成。
        Generate a poisson spike train according input rate.
        time: encoding window ms
        dt: time step
    """
    def __init__(self, shape=None, num=None, dec_target=None,  dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):

        super(Poisson_Generator2, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.num = num




    def torch_coding(self, source, device):
        # assert (source >= 0).all(), "Input rate must be non-negative"
        if not (source >= 0).all():
            import warnings
            warnings.warn('Input rate shall be non-negative')
        if source.__class__.__name__ == 'ndarray':
            source = torch.tensor(source, device=device)

        if source.ndim == 0:
            batch = 1
        else:
            batch = source.shape[0]

        shape = list(self.shape)
        shape[0] = batch
        spk_shape = [self.time_step] + list(shape)
        spikes = torch.rand(spk_shape, device=device).le(source * self.dt).float()
        times = torch.zeros_like(spikes)
        spikes = torch.stack([spikes, times],dim=2)
        return spikes

Generator.register('poisson_generator2', Poisson_Generator2)


class CC_Generator(Generator):
    """
        恒定电流生成器。
        Generate a constant current input.
        time: encoding window ms
        dt: time step
    """
    def __init__(self, shape=None, num=None, dec_target=None,  dt=None,
                 coding_method=('poisson_generator', 'cc_generator', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(CC_Generator, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.num = num

    def torch_coding(self, source, device):
        # assert (source >= 0).all(), "Input rate must be non-negative"
        if not (source >= 0).all():
            import warnings
            warnings.warn('Input rate shall be non-negative')
        if source.__class__.__name__ == 'ndarray':
            source = torch.tensor(source, dtype=torch.float, device=device)

        # if source.ndim == 0:
        #     batch = 1
        # else:
        #     batch = source.shape[0]
        #
        # shape = [batch, self.num]
        spk_shape = [self.time_step] + list(self.shape)
        spikes = source * torch.ones(spk_shape, device=device)
        return spikes

Generator.register('cc_generator', CC_Generator)