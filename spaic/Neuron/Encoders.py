
# encoding: utf-8
"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: Encoders.py
@time:2021/5/7 14:50
@description:
"""
from .Node import Node, Encoder
import torch
import numpy as np

class NullEncoder(Encoder):
    '''
        Pass the encoded data.
    '''


    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method='null', coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super().__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, source, device):
        # assert (source >= 0).all(), "Inputs must be non-negative"
        # Note: the shape of encoded date should be (batch, time_step, shape)
        if source.__class__.__name__ == 'ndarray':
            source = torch.tensor(source, device=device, dtype=torch.float32)
        return source.transpose(1, 0)

Encoder.register('null', NullEncoder)

class SigleSpikeToBinary(Encoder):
    '''
        Transform the spike train (each neuron firing one spike) into a binary matrix
        The source is the encoded time value in the range of [0,time]. The shape of encoded source should be [time_step, batch_size, neuron_shape].
    '''

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super().__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, source, device):
        assert (source >= 0).all(), "Inputs must be non-negative"
        # if source.__class__.__name__ == 'ndarray':
        #     source = torch.tensor(source, device=device, dtype=torch.float32)
        shape = list(source.shape)
        spk_shape = [self.time_step] + shape

        # mapping the spike times into [0,1] matrix.
        source_temp = source/self.dt
        spike_index = source_temp
        spike_index = spike_index.reshape([1] + shape).to(device=device, dtype=torch.long)
        spikes = torch.zeros(spk_shape, device=device)
        spike_src = torch.ones_like(spike_index, device=device, dtype=torch.float32)
        spikes.scatter_(dim=0, index=spike_index, src=spike_src)
        return spikes

Encoder.register('sstb', SigleSpikeToBinary)


class MultipleSpikeToBinary(Encoder):
    '''
        Transform the spike train (each neuron firing multiple spikes) into a binary matrix
        The source is the encoded time value in the range of [0,time]. The shape of encoded source should be [time_step, batch_size, neuron_shape].
    '''

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super().__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.deltaT = kwargs.get('deltaT', False)

    def torch_coding(self, source, device):
        # 直接使用for循环
        # if source.__class__.__name__ == 'ndarray':
        #     source = torch.tensor(source, device=device, dtype=torch.float32)
        all_spikes = []
        if '[2]' in self.coding_var_name:
            for i in range(source.shape[0]):
                spiking_times, neuron_ids = source[i]
                assert (spiking_times >= 0).all(), "Inputs must be non-negative"
                # spike_index = (spiking_times + np.random.rand(*(spiking_times.shape))) / self.dt
                spike_index = spiking_times / self.dt
                delta_times = torch.tensor((np.ceil(spike_index) - spike_index) * self.dt, device=device, dtype=torch.float)
                values = torch.ones(spike_index.shape, device=device)
                indexes = [spike_index, neuron_ids]
                indexes = torch.tensor(indexes, device=device, dtype=torch.long)
                indexes[0] = torch.clamp_max(indexes[0], self.time_step - 1)
                spk_shape = [self.time_step, self.num]
                spike_values = torch.sparse.FloatTensor(indexes, values, size=spk_shape).to_dense()
                spike_dts = torch.sparse.FloatTensor(indexes, delta_times, size=spk_shape).to_dense()
                all_spikes.append(torch.stack([spike_values, spike_dts], dim=1))
            spikes = torch.stack(all_spikes, dim=1)

        else:
            for i in range(source.shape[0]):
                spiking_times, neuron_ids = source[i]

                assert (spiking_times >= 0).all(), "Inputs must be non-negative"
                spike_index = spiking_times / self.dt
                spike_index = spike_index #.squeeze(-1)
                # neuron_ids = np.float32(neuron_ids) #.squeeze(-1)
                indexes = [spike_index, neuron_ids]
                if source.__class__.__name__ == 'ndarray':
                    indexes = torch.tensor(indexes, device=device, dtype=torch.long)
                    indexes[0] = torch.clamp_max(indexes[0], self.time_step-1)


                spk_shape = [self.time_step, self.num]

                # mapping the spike times into [0,1] matrix.
                indexes = indexes.to(dtype=torch.long)
                values = torch.ones(spike_index.shape, device=device)
                spike = torch.sparse.FloatTensor(indexes, values, size=spk_shape)
                spike = spike.to_dense()
                all_spikes.append(spike)
            spikes = torch.stack(all_spikes, dim=1)

        # # 尝试使用多线程
        # def get_spike(source_data):
        #     spiking_times, neuron_ids = source_data
        #     assert (spiking_times >= 0).all(), "Inputs must be non-negative"
        #     spike_index = spiking_times / self.dt
        #     spike_index = spike_index  # .squeeze(-1)
        #     neuron_ids = np.float32(neuron_ids)  # .squeeze(-1)
        #     indexes = [spike_index, neuron_ids]
        #     if source.__class__.__name__ == 'ndarray':
        #         indexes = torch.tensor(indexes, device=device)
        #
        #     spk_shape = [self.time_step, self.num]
        #
        #     # mapping the spike times into [0,1] matrix.
        #     indexes = indexes.to(dtype=torch.long)
        #     values = torch.ones(spike_index.shape, device=device)
        #     spike = torch.sparse.FloatTensor(indexes, values, size=spk_shape)
        #     spike = spike.to_dense()
        #     return spike
        # all_spikes = []
        # with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 64)) as executor:
        #     for spike in executor.map(get_spike, source):
        #         all_spikes.append(spike)
        # spikes = torch.stack(all_spikes)
        # spikes = spikes.permute(1, 0, 2)
        return spikes

Encoder.register('mstb', MultipleSpikeToBinary)

class PoissonEncoding(Encoder):
    """
        泊松频率编码，发放脉冲的概率即为刺激强度，刺激强度需被归一化到[0, 1]。
        Generate a poisson spike train.
        time: encoding window ms
        dt: time step
    """
    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(PoissonEncoding, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.unit_conversion = kwargs.get('unit_conversion', 1.0)

    def numpy_coding(self, source, device):
        # assert (source >= 0).all(), "Inputs must be non-negative"
        shape = list(source.shape)
        spk_shape = [self.time_step] + shape
        spikes = np.random.rand(*spk_shape).__le__(source * self.dt).astype(float)
        return spikes

    def torch_coding(self, source, device):
        # assert (source >= 0).all(), "Inputs must be non-negative"
        if source.__class__.__name__ == 'ndarray':
            source = torch.tensor(source, device=device, dtype=torch.float32)
        shape = source.shape
        # source_temp = source.view(shape[0], -1)
        spk_shape = [self.time_step] + list(shape)
        spikes = torch.rand(spk_shape, device=device).le(source * self.unit_conversion*self.dt).float()
        return spikes

Encoder.register('poisson', PoissonEncoding)

class Latency(Encoder):
    """
        延迟编码，刺激强度越大，脉冲发放越早。刺激强度被归一化到[0, 1]。
        Generate a latency encoding spike train.
        time: encoding window ms
        dt: time step
    """

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Latency, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, source, device):
        assert (source >= 0).all(), "Inputs must be non-negative"

        if source.__class__.__name__ == 'ndarray':
            source = torch.tensor(source, device=device, dtype=torch.float32)
        shape = list(source.shape)
        spk_shape = [self.time_step] + shape
        max_scale = self.time_step - 1.0

        # Create spike times in order of decreasing intensity.
        min_value = 1.0e-10
        source_temp = (source - torch.min(source)) / (torch.max(source) - torch.min(source) + min_value)
        spike_index = max_scale*(1-source_temp)
        spike_index = spike_index.reshape([1] + shape).to(device=device, dtype=torch.long)
        spikes = torch.zeros(spk_shape, device=device)
        spike_src = torch.ones_like(spike_index, device=device, dtype=torch.float)
        spikes.scatter_(dim=0, index=spike_index, src=spike_src)
        return spikes
Encoder.register('latency', Latency)

class Relative_Latency(Encoder):
    '''
        相对延迟编码，在一个样本中，其相对强度越大，放电越靠前
    '''

    def __init__(self,shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Relative_Latency, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        # self.time_step = int(self.time/self.dt)

        self.amp = kwargs.get('amp', 1.0)  # nn.Parameter(amp)
        self.bias = kwargs.get('bias', 0)
        scale = kwargs.get('scale', 0.9999999)
        if scale < 1.0 and scale > 0.0:
            self.scale = scale
        else:
            raise ValueError("scale out of defined scale ")

    # def build(self, backend):
    #     super(Relative_Latency, self).build(backend)


    def torch_coding(self, source, device):
        import torch.nn.functional as F
        if source.__class__.__name__ == 'ndarray':
            source = torch.tensor(source, device=device, dtype=torch.float32)

        self.max_scale = self.time_step - 1.0
        shape = list(source.shape)
        tmp_source = source.view(shape[0], -1)
        spk_shape = [self.time_step] + shape
        tmp_source = torch.exp(-self.amp*tmp_source)
        tmp_source = tmp_source - torch.min(tmp_source, dim=1, keepdim=True)[0]
        spike_index = self.max_scale*self.scale*tmp_source #torch.sigmoid(tmp_source + self.bias) #-torch.mean(tmp_source, dim=-1, keepdim=True)
        spike_index = spike_index.reshape([1]+shape).to(device=device, dtype=torch.long)
        max_index = torch.max(spike_index)
        min_index = torch.min(spike_index)
        cut_index = (min_index + 0.8*(max_index-min_index)).to(torch.long)
        spikes = torch.zeros(spk_shape, device=device)
        spike_src = torch.ones_like(spike_index, device=device, dtype=torch.float)
        spikes.scatter_(dim=0, index=spike_index, src=spike_src)
        spikes[cut_index:, ...] = 0
        return spikes
Encoder.register('relative_latency', Relative_Latency)




