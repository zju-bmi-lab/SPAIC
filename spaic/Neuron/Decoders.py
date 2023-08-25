
# encoding: utf-8
"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: Decoders.py
@time:2021/5/7 14:50
@description:
"""

from .Node import Node, Decoder
import torch
import numpy as np

class Spike_Rate(Decoder):
    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Spike_Rate, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.pop_size = kwargs.get('pop_size', 1)
        self.bias = kwargs.get('bias', 0.0)
        self.scale = kwargs.get('scale', 1.0)

    def numpy_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        spike_rate = record.mean(0)
        pop_num = int(self.num / self.pop_size)
        pop_spikes_temp = (
            [
                np.sum(spike_rate[:, (i * self.pop_size): (i * self.pop_size) + self.pop_size], axis=1)
                for i in range(pop_num)
            ]
        )
        pop_spikes = np.stack(pop_spikes_temp, axis=1)
        return pop_spikes

    def torch_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        if '[2]' in self.coding_var_name:
            pop_spikes = record[:,:,0,:].mean(0).to(device=device)
        else:
            spike_rate = record.mean(0).to(device=device)
            pop_num = int(self.num / self.pop_size)
            pop_spikes_temp = (
                [
                    spike_rate[:, (i * self.pop_size): (i * self.pop_size) + self.pop_size].sum(dim=1)
                    for i in range(pop_num)
                ]
            )
            pop_spikes = torch.stack(pop_spikes_temp, dim=-1)
        return (pop_spikes + self.bias)*self.scale

Decoder.register('spike_rate', Spike_Rate)


class Spike_Counts(Decoder):

    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Spike_Counts, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.pop_size = kwargs.get('pop_size', 1)
        self.bias = kwargs.get('bias', 0.0)
        self.scale = kwargs.get('scale', 1.0)

    def numpy_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        spike_rate = record.mean(0)
        pop_num = int(self.num / self.pop_size)
        pop_spikes_temp = (
            [
                np.sum(spike_rate[:, (i * self.pop_size): (i * self.pop_size) + self.pop_size], axis=1)
                for i in range(pop_num)
            ]
        )
        pop_spikes = np.stack(pop_spikes_temp, axis=1)
        # spikes_list = pop_spikes.tolist()
        # max_value = np.max(pop_spikes, 1)
        # batch_size = pop_spikes.shape[0]
        # predict_labels = []
        # for i in range(batch_size):
        #     index = spikes_list[i].index(max_value[i])
        #     predict_labels.append(index)
        return pop_spikes

    def torch_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        if '[2]' in self.coding_var_name:
            pop_spikes = record[:,:,0,:].sum(0).to(device=device)
        else:
            spike_rate = record.sum(0).to(device=device)
            pop_num = int(self.num / self.pop_size)
            pop_spikes_temp = (
                [
                    spike_rate[:, (i * self.pop_size): (i * self.pop_size) + self.pop_size].sum(dim=1)
                    for i in range(pop_num)
                ]
            )
            pop_spikes = torch.stack(pop_spikes_temp, dim=-1)
        return (pop_spikes + self.bias)*self.scale

Decoder.register('spike_counts', Spike_Counts)


class Spike_Rates(Decoder):

    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Spike_Rates, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def numpy_coding(self, record, target, device):
        pass

    def torch_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        spike_rates = record.sum(0)/self.time_step
        return spike_rates

Decoder.register('spike_rates', Spike_Rates)

class Final_Step_Voltage(Decoder):

    """
    Get label that has the highest voltage.
    """

    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='V', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Final_Step_Voltage, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        final_step_voltage = record[-1, :, :]

        return final_step_voltage

Decoder.register('final_step_voltage', Final_Step_Voltage)

class First_Spike(Decoder):


    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(First_Spike, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    # def numpy_decoding(self, record):
    #     # the shape of record is (time_step, batch_size, n_neurons)
    #     # get predict label
    #     spikes_list = record.tolist()
    #
    #     return predict_matrix

    def torch_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        record_temp = record.permute(1, 0, 2)
        [batch_size, time_step, n_neurons] = record_temp.shape
        batch_index = []
        for i in range(batch_size):
            index = torch.nonzero(record_temp[i, ::])
            if len(index) == 0:
                first_spike_row = 0 + i * time_step
            else:
                first_spike_row = index[0, 0].item() + i * time_step

            batch_index.append(first_spike_row)
        batch_index = torch.LongTensor(batch_index).to(device=device)
        record_temp = record_temp.reshape(batch_size*time_step, n_neurons)
        predict_matrix = torch.index_select(record_temp, 0, batch_index).to(device=device)
        return predict_matrix

Decoder.register('first_spike', First_Spike)

class TimeSpike_Counts(Decoder):


    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(TimeSpike_Counts, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def numpy_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        # get predict label
        record = np.sum(record, axis=-1)
        spikes_list = record.tolist()
        max_value = np.max(record, 1)
        batch_size = record.shape[0]
        predict_labels = []
        for i in range(batch_size):
            index = spikes_list[i].index(max_value[i])
            predict_labels.append(index)
        return predict_labels

    def torch_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        tlen = record.shape[0]
        tt = torch.arange(0, tlen, device=device, dtype=torch.float)
        tw = 0.1*torch.exp(-tt/(0.5*tlen))
        predict_labels = torch.sum(record.permute(1, 2, 0)*tw, dim=-1)
        return predict_labels

Decoder.register('time_spike_counts',TimeSpike_Counts)

class NullDeocder(Decoder):
    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(NullDeocder, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, record, target, device):
        return record
Decoder.register('null', NullDeocder)


class V_Trajectory(Decoder):

    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='V', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(V_Trajectory, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)

        return record

Decoder.register('v_t', V_Trajectory)


class Time_Softmax(Decoder):


    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Time_Softmax, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def numpy_coding(self, record, target, device):
        pass
      
    def torch_coding1(self, record: torch.Tensor, target, device):
        def grad_regulate_hook(grad):
            return grad - torch.mean(grad)

        spike_i = record[:,:,0,...]
        spike_t = record[:,:,1,...]
        old_shape = spike_t.shape
        tlen = record.shape[0]
        time_array = self.dt*torch.arange(0, tlen, device=device, dtype=torch.float)
        spike_t = spike_i.detach()*time_array.view(-1, 1, 1) - spike_t# + (1-spike_i.detach())*1000.0
        max_t, ind = torch.max(spike_t.permute(0, 2, 1).reshape(-1, old_shape[1]).detach(), dim=0)
        mshape = [1, -1] + [1]*(spike_t.dim()-2)
        spike_t = 0.1*(max_t.view(mshape) + (spike_i - spike_i.detach()) - spike_t)*spike_i.detach()
        # spike_t.register_hook(grad_regulate_hook)
        # spike_t[:,0] = 7
        out = torch.softmax(torch.norm(spike_t, dim=0), dim=-1)
        # torch.sum(spike_t, dim=-1)     *spike_t.view(old_shape[0],-1)
        # from matplotlib import pyplot as plt
        # plt.plot(out.detach().cpu().numpy()[0,:])
        # plt.show()
        # out = out.view(old_shape)
        # out = torch.sum(out, dim=-1)
        # if torch.mean(out) <1:
        return out

    def torch_coding(self, record: torch.Tensor, target, device):
        # def grad_regulate_hook(grad):
        #     return grad - torch.mean(grad)

        shape = list(record.shape)
        shape[-1] = self.num
        shape.append(-1)
        record = record.view(*shape)


        spike_i = record[:,:,0,...]
        spike_t = record[:,:,1,...]
        spike = spike_t + spike_i
        # spk_ind = spike_t.clamp_max(1.0).detach()
        # old_shape = spike_t.shape
        tlen = record.shape[0]
        time_array = self.dt*torch.arange(0, tlen, device=device, dtype=torch.float)
        spike_t = time_array.view(-1, 1, 1, 1) - spike + spike.detach() + spike_i.le(0.0)*self.dt*tlen
        # spike_t = spike_i.gt(0.0)*time_array.view(-1, 1, 1) - spike_t
        frist_times = torch.amin(spike_t, dim=(0,2), keepdim=True).detach()
        # spike_ft = torch.amin(spike_t, dim=0, keepdim=True)
        # target_frist = torch.gather(frist_times, 2, target.view(1,-1,1,1)).detach()
        out_count = torch.mean(torch.sum(spike_i, dim=0), dim=-1).detach() + 1.0
        out_ti = 5.0*torch.exp((frist_times-spike_t)/50.0)
        out = torch.mean(torch.sum(out_ti,dim=0),dim=-1)
        out = out/out_count + out.detach()*(out_count-1)/out_count
        rateloss = torch.norm(torch.mean(torch.sum(spike_i,dim=0), dim=-2)-3.0)
        return out, rateloss

    def torch_coding3(self, record: torch.Tensor, target, device):
        # def grad_regulate_hook(grad):
        #     return grad - 10000#torch.mean(grad)
        spike_i = record[:,:,0,...]
        spike_t = record[:,:,1,...]
        self.out = torch.sum(record, dim=2)
        self.target = target.repeat(spike_t.shape[0], 1).unsqueeze(-1)


        with torch.no_grad():
            tlen = record.shape[0]
            time_array = self.dt*torch.arange(0, tlen, device=device, dtype=torch.float)
            spike_t = time_array.view(-1, 1, 1) - spike_t
            exp_n = spike_i*torch.exp(-spike_t/20.0)
            exp_sum = torch.sum(exp_n, dim=(0, 2), keepdim=True) + 1.0e-20
            self.exp_n = exp_n/exp_sum

        return torch.sum(self.exp_n, dim=0)

    @property
    def loss(self):
        with torch.no_grad():
            pos_n = torch.gather(self.exp_n, dim=2, index=self.target)
            sum_pos = torch.sum(pos_n, dim=0, keepdim=True) + 1.0e-20
            pos_n = (sum_pos - 1) * (pos_n + 1.0e-18) / sum_pos
            out_grad = torch.scatter(self.exp_n * torch.gt(sum_pos, 1.0e-10), 2, self.target, pos_n)
        if self.out.requires_grad == True:
            self.out.backward(out_grad)
        return  -torch.mean(torch.log(sum_pos))

Decoder.register('time_softmax', Time_Softmax)

class Voltage_Sum(Decoder):

    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='V', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Voltage_Sum, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def numpy_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        record_temp = record.sum(0)
        spikes_list = record_temp.tolist()
        max_value = np.max(record_temp, 1)
        batch_size = record_temp.shape[0]
        predict_labels = []
        for i in range(batch_size):
            index = spikes_list[:, i].index(max_value[i])
            predict_labels.append(index)
        return predict_labels

    def torch_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        predict_matrix = record[-1,...].to(device=device)
        mp = torch.mean(predict_matrix).detach()
        out = (predict_matrix - mp)
        # predict_matrix = predict_temp.permute(1, 0)
        return out

Decoder.register('voltage_sum', Voltage_Sum)

class Complex_Count(Decoder):
    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Complex_Count, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.tlen = None
    def torch_coding(self, record: torch.Tensor, target, device):
        assert record.dtype.is_complex
        if self.tlen is None:
            self.tlen = record.shape[0]*1.0
        time_array = torch.arange(0, self.tlen, device=device, dtype=torch.float).view(-1,1,1)
        spk = record.real.gt(0.0)
        # for gradient test
        sum_spk = torch.cumsum(spk, dim=0)
        spk = spk*sum_spk.lt(5)

        # count = record.imag.gt(0)
        out = torch.sum(spk*(-(time_array-record.imag)/self.tlen), dim=0) + 1.0e-6
        # out = torch.sum(count, dim=0) + 1.0e-6
        # rate = torch.sum(count, dim=0)/self._backend.time
        return out

Decoder.register('complex_count', Complex_Count)


class Complex_Phase(Decoder):
    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Complex_Phase, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.trange = kwargs.get('trange',2.0)
        self.period = kwargs.get('period', 20.0)
        # class TransClamp(torch.autograd.Function):
        #     @staticmethod
        #     def forward(ctx, x, min=None, max=None):
        #         return torch.clamp(x, min, max)
        #
        #     @staticmethod
        #     def backward(ctx, grad_outputs):
        #         return grad_outputs, None, None
        # self.clamp = TransClamp.apply




    def torch_coding(self,  record: torch.Tensor, target, device):
        assert record.dtype.is_complex
        # record.shape = (time, batch, num)
        tlen = record.shape[0]
        reference = torch.mean(record.abs(), dim=-1, keepdim=True)
        reference = torch.softmax(reference, dim=0)*self.dt
        phase = record.real/(record.abs()+1.0)
        phase = torch.sum(phase*reference, dim=0)
        return phase

    def torch_coding2(self, record, target, device):
        import torch.nn.functional as F
        tlen = record.shape[0]
        batch_size = record.shape[1]
        out_num = record.shape[2]
        kernel_range = int(self.period / self.dt)
        w = 2 * np.pi / self.period
        tt = self.dt * torch.arange(0, kernel_range, device=device, dtype=torch.float64)
        kernel_i = torch.sin(-w * tt).view(1, 1, -1)
        kernel_t = -self.dt*w*torch.cos(-w * tt).view(1, 1, -1)

        x = record.view(tlen, -1).t().view(-1, 1, tlen)
        conv_i = torch.mean(F.conv1d(x.real, kernel_i, padding=kernel_range)[...,kernel_range//2:kernel_range//2+tlen]
                            .view(batch_size, out_num, tlen), dim=1, keepdim=True)
        conv_t = torch.mean(F.conv1d(x.real, kernel_t, padding=kernel_range)[...,kernel_range//2:kernel_range//2+tlen]
                            .view(batch_size, out_num, tlen), dim=1, keepdim=True)
        x = x.view(batch_size, out_num, tlen)
        out_phase = torch.sum(conv_i*x.real.detach() + conv_t*x.imag*self.dt, dim=-1)

        return out_phase, conv_i


Decoder.register('complex_phase', Complex_Phase)

class Complex_Latency(Decoder):
    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Complex_Latency, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.tlen = None

    def torch_coding(self,  record: torch.Tensor, target, device):
        assert record.dtype.is_complex
        # if self.tlen is None:
        self.tlen = record.shape[0]
        time_array = torch.arange(0, self.tlen, device=device, dtype=torch.float).view(-1,1,1)
        spk = record.real.gt(0)
        spk_time = (self.tlen-(time_array - record.imag))*spk
        spk_time = torch.exp(1*(spk_time-torch.amax(spk_time, dim=(0,2), keepdim=True).detach())/self.tlen)
        spk_rate = record.real
        spk_weight = torch.exp(-torch.cumsum(record.real.detach(),dim=0)/5.0)
        weighted_spk_time = torch.sum(spk_weight*spk_rate*spk_time*spk,dim=0)
        return weighted_spk_time

Decoder.register('complex_latency', Complex_Latency)

class Complex_TimingDistance(Decoder):
    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Complex_TimingDistance, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        from matplotlib import pyplot as plt
        self.tlen = None
        self.filter_time = kwargs.get("filter_time", 10.0)


    def build(self, backend):
        super(Complex_TimingDistance, self).build(backend)
        tdt = 6.0*self.dt/(1.0*self.filter_time)
        tt = torch.arange(0, 6.0+tdt, tdt)
        self.rate_filter = (torch.exp(-(tt - 2) ** 2)).view(1, 1, -1)
        self.d_rate_filter = (0.5*(tt-2)*torch.exp(-(tt - 1) ** 2)).view(1, 1, -1)*tdt*0.01


    def torch_coding(self, record: torch.Tensor, target: torch.Tensor , device: str):
        from torch.nn.functional import mse_loss
        # record shape (time, batch, neuron)
        # target shape (batch, neuron, time)
        assert record.dtype.is_complex
        n_time, n_batch, n_neuron = record.shape
        self.rate_filter = self.rate_filter.to(device)
        self.d_rate_filter = self.d_rate_filter.to(device)
        target = target.view(-1, 1, n_time)
        record = record.permute(1,2,0).view(-1, 1, n_time)

        target_rate = torch.conv1d(target, self.rate_filter, padding='same')
        record_rate = torch.conv1d(record.real, self.rate_filter, padding='same') - torch.conv1d(record.imag, self.d_rate_filter, padding='same')
        rate_loss = mse_loss(record_rate, target_rate)
        return rate_loss, record_rate, target_rate

Decoder.register('complex_timing_distance', Complex_TimingDistance)




class Complex_Trajectory(Decoder):

    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Complex_Trajectory, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.tau_d = kwargs.get('tau_d', 20.0+20*torch.rand(num).view(1,-1))
        self.tau_r = kwargs.get('tau_r', 10.0+50*torch.rand(num).view(1,-1))
        self.group_num = kwargs.get('group_num', 1)
        self.num = num
        assert num%self.group_num == 0
        if not isinstance(self.tau_d, torch.Tensor):
            self.tau_d = torch.tensor(self.tau_d)
        if not isinstance(self.tau_r, torch.Tensor):
            self.tau_r = torch.tensor(self.tau_r)
        self.weight = kwargs.get('weight', None)
        if self.weight is None:
            self.weight = torch.randn(num).view(1,-1)

    def build(self, backend):
        super(Complex_Trajectory, self).build(backend)
        tau_d = self.variable_to_backend(self.id+'_Complex_Trajectory_tau_d', self.tau_d.shape, self.tau_d, True)
        tau_r = self.variable_to_backend(self.id+'_Complex_Trajectory_tau_r', self.tau_r.shape, self.tau_r, True)
        weight = self.variable_to_backend(self.id+'_Complex_Trajectory_weight', self.tau_r.shape, self.weight, True)
        self.tau_d = tau_d.value
        self.tau_r = tau_r.value
        self.weight = weight.value




    def torch_coding(self,  record: torch.Tensor, target=None, device='cpu'):
        decay = torch.exp(-self.dt/self.tau_d).to(device)
        rota = (2.0*torch.pi*self.dt/self.tau_r).to(device)
        complex_beta = torch.view_as_complex(torch.stack([decay * torch.cos(-rota),
                                                          decay * torch.sin(-rota)], dim=-1))
        weight = self.weight.unsqueeze(-1)
        tlen = record.shape[0]
        Xs = []
        x = torch.zeros_like(record[0])
        for ii in range(tlen):
            x = complex_beta*x + record[ii].real*(0.0+1.0j)*complex_beta**record[ii].imag
            Xs.append(x)
        Xs = weight*torch.stack(Xs, dim=-1)
        Xs = Xs.view(-1, self.group_num, self.num//self.group_num, tlen)

        # trace = Xs.real
        # dtrace = rota*Xs.imag - Xs.real/self.tau_d

        return torch.sum(Xs.real, dim=-2)

Decoder.register('complex_trajectory', Complex_Trajectory)


class Complex_Spike_Conv(Decoder):
    def __init__(self, num=None, dec_target=None, dt=None, coding_method='spike_counts', coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), decay=0.9, ocillate=-0.01, **kwargs):
        super(Complex_Spike_Conv, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.kernel = kwargs.get('kernel', None)
        self.time_window = kwargs.get('time_window', 80)/2.0

        if self.kernel is None:
            tt = torch.arange(-self.time_window, self.time_window, self.dt)
            self.kernel = torch.exp(-(3*tt/self.time_window)**2).view(1, 1, -1)
            self.d_kernel = (6*tt/self.time_window)*torch.exp(-(3*tt/self.time_window)**2).view(1, 1, -1)
        self.klen = self.kernel.shape[-1]

    def torch_coding(self, source, target, device='cpu'):
        weight = self.kernel.to(device).expand(source.shape[1],source.shape[1],self.klen)
        d_weight = self.d_kernel.to(device).expand(source.shape[1],source.shape[1],self.klen)
        source = source.transpose(0, 1).transpose(1, 2)
        target = target.transpose(0, 1).transpose(1, 2)
        filt_source = torch.conv1d(source.real.to(weight.dtype), weight, padding='same') + torch.conv1d((source.real*source.imag).to(weight.dtype)*self.dt, d_weight, padding='same')
        filt_target = torch.conv1d(target.real.to(weight.dtype), weight, padding='same') + torch.conv1d((target.real*target.imag).to(weight.dtype)*self.dt, d_weight, padding='same')
        return filt_source, filt_target

Decoder.register('complex_conv', Complex_Spike_Conv)

class Spike_Conv(Decoder):
    def __init__(self, num=None, dec_target=None, dt=None, coding_method='spike_counts', coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), decay=0.9, ocillate=-0.01, **kwargs):
        super(Spike_Conv, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.kernel = kwargs.get('kernel', None)
        self.time_window = kwargs.get('time_window', 80)/2.0

        if self.kernel is None:
            tt = torch.arange(-self.time_window, self.time_window, self.dt)
            self.kernel = torch.exp(-(3*tt/self.time_window)**2).view(1, 1, -1)
        self.klen = self.kernel.shape[-1]

    def torch_coding(self, source, target, device='cpu'):
        weight = self.kernel.to(device).expand(source.shape[1],source.shape[1],self.klen)
        source = source.transpose(0, 1).transpose(1, 2)
        target = target.transpose(0, 1).transpose(1, 2)
        filt_source = torch.conv1d(source.to(weight.dtype), weight, padding='same')
        filt_target = torch.conv1d(target.to(weight.dtype), weight, padding='same')
        return filt_source, filt_target

Decoder.register('spike_conv', Spike_Conv)