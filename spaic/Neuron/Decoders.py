
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


class Spike_Counts(Decoder):

    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Spike_Counts, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.pop_size = kwargs.get('pop_size', 1)

    def numpy_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        spike_rate = record.sum(0)
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
        return pop_spikes

Decoder.register('spike_counts', Spike_Counts)

class Final_Step_Voltage(Decoder):

    """
    Get label that has the highest voltage.
    """

    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
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
        # the shape of record is (batch_size, n_neurons)
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
        # the shape of record is (batch_size, n_neurons)
        tlen = record.shape[-1]
        tt = torch.arange(0, tlen, device=device, dtype=torch.float)
        tw = 0.1*torch.exp(-tt/(0.5*tlen))
        predict_labels = torch.sum(record*tw, dim=-1)
        return predict_labels

Decoder.register('time_spike_counts',TimeSpike_Counts)


class V_Trajectory(Decoder):

    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='V', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(V_Trajectory, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, record, target, device):
        # the shape of record is (batch_size, n_neurons)

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

class Spike_Conv(Decoder):
    def __init__(self, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), decay=0.9, ocillate=-0.01, **kwargs):
        super(Spike_Conv, self).__init__(num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
