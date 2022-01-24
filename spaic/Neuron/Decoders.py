
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

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Spike_Counts, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def numpy_coding(self, source, target, device):
        # the shape of source is (time_step, batch_size, n_neurons)
        source_temp = source.sum(0)
        spikes_list = source_temp.tolist()
        max_value = np.max(source_temp, 1)
        batch_size = source_temp.shape[0]
        predict_labels = []
        for i in range(batch_size):
            index = spikes_list[:, i].index(max_value[i])
            predict_labels.append(index)
        return predict_labels

    def torch_coding(self, source, target, device):
        # the shape of source is (time_step, batch_size, n_neurons)
        predict_matrix = source.sum(0).to(device=device)
        # predict_matrix = predict_temp.permute(1, 0)
        return predict_matrix

Decoder.register('spike_counts', Spike_Counts)

class First_Spike(Decoder):


    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(First_Spike, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    # def numpy_decoding(self, source):
    #     # the shape of source is (time_step, batch_size, n_neurons)
    #     # get predict label
    #     spikes_list = source.tolist()
    #
    #     return predict_matrix

    def torch_coding(self, source, target, device):
        # the shape of source is (time_step, batch_size, n_neurons)
        source_temp = source.permute(1, 0, 2)
        [batch_size, time_step, n_neurons] = source_temp.shape
        batch_index = []
        for i in range(batch_size):
            index = torch.nonzero(source_temp[i, ::])
            if len(index) == 0:
                first_spike_row = 0 + i * time_step
            else:
                first_spike_row = index[0, 0].item() + i * time_step

            batch_index.append(first_spike_row)
        batch_index = torch.LongTensor(batch_index).to(device=device)
        source_temp = source_temp.reshape(batch_size*time_step, n_neurons)
        predict_matrix = torch.index_select(source_temp, 0, batch_index).to(device=device)
        return predict_matrix

Decoder.register('first_spike', First_Spike)

class TimeSpike_Counts(Decoder):


    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(TimeSpike_Counts, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def numpy_coding(self, source, target, device):
        # the shape of source is (batch_size, n_neurons)
        # get predict label
        source = np.sum(source, axis=-1)
        spikes_list = source.tolist()
        max_value = np.max(source, 1)
        batch_size = source.shape[0]
        predict_labels = []
        for i in range(batch_size):
            index = spikes_list[i].index(max_value[i])
            predict_labels.append(index)
        return predict_labels

    def torch_coding(self, source, target, device):
        # the shape of source is (batch_size, n_neurons)
        tlen = source.shape[-1]
        tt = torch.arange(0, tlen, device=device, dtype=torch.float)
        tw = 0.1*torch.exp(-tt/(0.5*tlen))
        predict_labels = torch.sum(source*tw, dim=-1)
        return predict_labels

Decoder.register('time_spike_counts',TimeSpike_Counts)


class Time_Softmax(Decoder):


    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Time_Softmax, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

        self.tau_m = kwargs.get('tau_m', 30.0)

    def numpy_coding(self, source, target, device):
        pass
      
    def torch_coding1(self, source: torch.Tensor, target, device):
        def grad_regulate_hook(grad):
            return grad - torch.mean(grad)

        spike_i = source[:,:,0,...]
        spike_t = source[:,:,1,...]
        old_shape = spike_t.shape
        tlen = source.shape[0]
        time_array = self.dt*torch.arange(0, tlen, device=device, dtype=torch.float)
        spike_t = spike_i.detach()*time_array.view(-1, 1, 1) - spike_t# + (1-spike_i.detach())*1000.0
        max_t, ind = torch.max(spike_t.permute(0, 2, 1).reshape(-1, old_shape[1]).detach(), dim=0)
        mshape = [1,-1] + [1]*(spike_t.dim()-2)
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

    def torch_coding(self, source: torch.Tensor, target, device):
        def grad_regulate_hook(grad):
            return grad - torch.mean(grad)

        spike_i = source[:,:,0,...]
        spike_t = source[:,:,1,...]
        spk_ind = spike_t.clamp_max(1.0).detach()
        old_shape = spike_t.shape
        tlen = source.shape[0]
        time_array = self.dt*torch.arange(0, tlen, device=device, dtype=torch.float)
        spike_t = spike_i.detach()*time_array.view(-1, 1, 1) - spike_t# + (1-spike_i.detach())*1000.0
        max_t, ind = torch.max(spike_t.permute(0, 2, 1).reshape(-1, old_shape[1]).detach(), dim=0)
        mshape = [1,-1] + [1]*(spike_t.dim()-2)
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



Decoder.register('time_softmax',Time_Softmax)

class EachStep_Reward(Decoder):


    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(EachStep_Reward, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, source, target, device):
        # the shape of source is (batch_size, n_neurons)
        if source.sum().item() > 0:
            predict = torch.argmax(source, dim=1)
            reward = -0.001 * torch.ones(target.shape, device=device)
            label = torch.tensor(target).long().to(device=device)
            flag = predict.eq(label)
            reward[flag] = 0.001

        else:
            reward = torch.zeros(target.shape, device=device)
        # label = target.long()
        # label = label.unsqueeze(0)
        # target = torch.zeros(source.shape).to(device=self.device)
        # one_hot_target = target.scatter_(0, label, 1)
        # reward = -1.0 * torch.ones(source.shape, device=self.device)
        # flag = (source == one_hot_target)
        # reward[flag] = 1.0
        if len(reward) > 1:
            reward = reward.mean()
        return reward

Decoder.register('step_reward', EachStep_Reward)

class Global_Reward(Decoder):


    def __init__(self,shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Global_Reward, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, source, target, device):
        # the shape of source is (time_step, batch_size, n_neurons)
        spike_rate = source.sum(0)
        predict = torch.argmax(spike_rate, dim=1)  # return the indices of the maximum values of a tensor across columns.
        reward = -0.1 * torch.ones(predict.shape, device=device)
        flag = (predict == target)
        reward[flag] = 0.1
        if len(reward) > 1:
            reward = reward.mean()
        return reward

Decoder.register('global_reward', Global_Reward)


class PopulationRate_Action(Decoder):
    """
    Selects an action probabilistically based on output spiking activity of population.
    """
    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(PopulationRate_Action, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, source, target, device):
        """
        Args:
            source: spiking activity of output layer. The shape of source is (time_step, batch_size, n_neurons)
        Returns:
            Action sampled from population output activity.
        """

        assert source.shape[2] % self.num == 0, (
            f"Output layer size of {source.shape[2]} is not divisible by action space size of"
            f" {self.num}."
        )

        pop_size = int(source.shape[2] / self.num)
        spike_num = source.sum().float()

        # Choose action based on population's spiking.
        if spike_num == 0:
            action = torch.randint(low=0, high=self.num, size=(1,))[0]
        else:
            pop_spikes = torch.tensor(
                [
                    source[:, :, (i * pop_size): (i * pop_size) + pop_size].sum()
                    for i in range(self.num)
                ],
                device=source.device,
            )
            # multinomial(input, num_samples,replacement=False), 采样的时候是根据输入张量的数值当做权重来进行抽样的, 返回index, 数值越大, 抽到的可能性越大, 如果是0 则不会抽到
            action = torch.multinomial((pop_spikes.float() / spike_num).view(-1), 1)[0].item()
        return action

Decoder.register('pop_rate_action', PopulationRate_Action)

class Softmax_Action(Decoder):

    """
    Selects an action using softmax function based on spiking of output layer.
    """

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Softmax_Action, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, source, target, device):
        """
        Args:
            source: spiking activity of output layer.
        Returns:
            Action sampled from softmax over spiking activity of output layer.
        """
        assert (
            source.shape[2] == self.num
        ), "Output layer size is not equal to the size of the action space."
        spikes = torch.sum(source, dim=0)
        probabilities = torch.softmax(spikes, dim=0)
        return torch.multinomial(probabilities, num_samples=1).item()

Decoder.register('softmax_action', Softmax_Action)


class Highest_Spikes_Action(Decoder):

    """
    Selects an action that has the highest firing rate. In case of equal spiking select randomly
    """

    def __init__(self,shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Highest_Spikes_Action, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, source, target, device):
        """
        Args:
            source: spiking activity of output layer.
        Returns:
            Action sampled from highest activities of output layer.
        """

        assert (
           source.shape[2] == self.num
        ), "Output layer size is not equal to the size of the action space."


        spikes = torch.sum(source, dim=0).squeeze()
        action = torch.where(spikes == spikes.max())[0]
        if torch.sum(spikes) == 0:
            action[0] = torch.randint(low=0, high=1, size=(1,))[0]

        return action[0]

Decoder.register('highest_spikes_action', Highest_Spikes_Action)

class Highest_Voltage_Action(Decoder):

    """
    Selects an action that has the highest voltage. In case of equal spiking select randomly
    """
    steps_done = 0
    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Highest_Voltage_Action, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, source, target, device):
        """
        Args:
            source: voltage of output layer.
        Returns:
            Action sampled from highest voltage of output layer.
        """
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 200

        assert (
           source.shape[2] == self.num
        ), "Output layer size is not equal to the size of the action space."

        import random
        import math
        sample = random.random()

        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * Highest_Voltage_Action.steps_done / EPS_DECAY)
        if source.shape[1] == 1:
            Highest_Voltage_Action.steps_done += 1
        if sample > eps_threshold:
            final_step_voltage = source[-1, :, :]
            action = final_step_voltage.max(1)[1]  # max(dim)返回的是value, index
            return action[0]
        else:
            return torch.randint(low=0, high=self.num, size=(1,))[0]

Decoder.register('highest_voltage_action', Highest_Voltage_Action)


class First_Spike_Action(Decoder):

    """
    Selects an action with have the highst spikes. In case of equal spiking select randomly
    """

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(First_Spike_Action, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, source, target, device):
        """
        Args:
            source: spiking activity of output layer.
        Returns:
            Action sampled from first spike of output layer.
        """

        assert (
            source.shape[2] == self.num
        ), "Output layer size is not equal to the size of the action space."


        spikes = source.squeeze().nonzero()
        if spikes.shape[0] == 0:
            action = torch.randint(low=0, high=1, size=(1,))[0]
        else:
            action = spikes[0, 1]

        return action

Decoder.register('first_spike_action', First_Spike_Action)


class Random_Action(Decoder):

    """
    Selects an action randomly from the action space.
    """

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Random_Action, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, source, target, device):
        """
        Used for the PyTorch backend

        Args:
           source: spiking activity of output layer.
        Returns:
           Action sampled from action space randomly.
       """
        return torch.randint(low=0, high=self.num, size=(1,))[0]

    def numpy_action(self, source):
        """
        Args:
           source: spiking activity of output layer.
        Returns:
           Action sampled from action space randomly.
       """
        return np.random.choice(self.num)

Decoder.register('random_action', Random_Action)
