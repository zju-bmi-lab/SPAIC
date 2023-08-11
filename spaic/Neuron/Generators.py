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
        self.start_time = kwargs.get('start_time', None)
        self.end_time = kwargs.get('end_time', None)
        if rate is not None:
            if hasattr(rate, '__iter__'):
                self.source = rate
            else:
                self.source = np.array([rate])
            self.new_input = True

        self.batch = kwargs.get('batch', 1)

    def torch_coding(self, source, device):

        if source.size != self.num:
            import warnings
            warnings.warn("The dimension of input data should be consistent with the number of input neurons.")
        if source.__class__.__name__ == 'ndarray':
            source = torch.tensor(source, device=device)
        self.inp_source = source
        #
        # # if source.ndim == 0:
        # #     batch = 1
        # # else:
        # #     batch = source.shape[SAS0]
        #
        # # shape = [self.batch, self.nu]
        # spk_shape = [self.time_step] + list(self.shape)
        # spikes = self.weight*torch.rand(spk_shape, device=device).le(source*self.unit_conversion).float()
        # if self.start_time is not None:
        #     start_time_step = int(self.start_time/self.dt)
        #     spikes[:start_time_step, ...] = 0.0
        # if self.end_time is not None:
        #     end_time_step = int(self.end_time/self.dt)
        #     spikes[end_time_step:, ...] = 0.0
        return None

    def next_stage(self):
        if self.new_input:
            self.get_input()
            self.shape[0] = self.inp_source.shape[0]
            self.new_input = False

        if (self.start_time is None or self.start_time< self.index*self.dt) \
            and (self.end_time is None or self.end_time> self.index*self.dt):
            spikes = self.weight*torch.rand(self.shape, device=self._backend.device[0]).le(
                self.inp_source*self.unit_conversion)
            return spikes.type(self._backend.data_type)
        else:
            return torch.zeros(self.shape, dtype=self._backend.data_type, device=self._backend.device)

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
        return spikes.type(self._backend.data_type)

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

    def torch_coding(self, source, device):
        # assert (source >= 0).all(), "Input rate must be non-negative"
        if not (source >= 0).all():
            import warnings
            warnings.warn('Input rate shall be non-negative')
        if source.__class__.__name__ == 'ndarray':
            source = torch.tensor(source, dtype=self._backend.data_type, device=device)

        spk_shape = [self.time_step] + list(self.shape)
        spikes = source * torch.ones(spk_shape, device=device)
        return spikes.type(self._backend.data_type)

Generator.register('cc_generator', CC_Generator)
Generator.register('constant_current', CC_Generator)
class Sin_Generator(Generator):
    """

        Generate a sin current input.
        time: encoding window ms
        dt: time step
    """
    def __init__(self, shape=None, num=None, dec_target=None,  dt=None,
                 coding_method=('poisson_generator', 'cc_generator', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Sin_Generator, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.num = num

    def torch_coding(self, source, device):
        # assert (source >= 0).all(), "Input rate must be non-negative"
        if source.__class__.__name__ == 'ndarray':
            source = torch.tensor(source, dtype=torch.float, device=device)
        amp = source[0]
        omg = 2*np.pi/source[1]
        # if source.ndim == 0:
        #     batch = 1
        # else:
        #     batch = source.shape[0]
        #
        # shape = [batch, self.num]
        spk_shape = [self.time_step] + [1 for _ in range(len(list(self.shape)))]

        t = torch.arange(0, self.time_step*self.dt, self.dt, device=device).view(spk_shape)
        spikes = amp*torch.sin(omg*t)
        return spikes

Generator.register('sin_generator', Sin_Generator)
Generator.register('sin', Sin_Generator)

class Ramp_Generator(Generator):
    def __init__(self, shape=None, num=None, dec_target=None,  dt=None,
                 coding_method=('poisson_generator', 'cc_generator', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Ramp_Generator, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.base = kwargs.get('base', 0.0)
        self.end_time = kwargs.get('end_time', None)
        self.amp = kwargs.get('amp', 0.001)

    def torch_coding(self, source, device):
        if source.__class__.__name__ == 'ndarray':
            source = torch.tensor(source, dtype=torch.float, device=device)
        if self.base.__class__.__name__ == 'ndarray':
            self.base = torch.tensor(self.base, dtype=torch.float, device=device)

        slope = source
        # spk_shape = [self.time_step] + [1 for _ in range(len(list(self.shape)))]
        t_shape = [self.time_step] + [1 for _ in range(len(list(self.shape)))]
        t = torch.arange(0, self.time_step*self.dt, self.dt, device=device).view(t_shape)
        spikes = self.amp*(slope*t + self.base)
        if self.end_time is not None:
            time_step = int(self.end_time/self.dt)
            spikes[time_step:,...] = 0.0
        return spikes
Generator.register('ramp_generator', Ramp_Generator)
Generator.register('ramp', Ramp_Generator)






