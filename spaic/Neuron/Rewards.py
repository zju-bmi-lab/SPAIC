# encoding: utf-8
"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: Rewards.py
@time:2021/12/2 9:43
@description:
"""
from .Node import Node, Reward
import torch
import numpy as np


class Global_Reward(Reward):

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Global_Reward, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type,
                                            **kwargs)
        self.pop_size = kwargs.get('pop_size', 1)
        self.reward_signal = kwargs.get('reward_signal', 1)
        self.punish_signal = kwargs.get('punish_signal', -1)

    def torch_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        spike_rate = record.sum(0)
        pop_num = int(self.num / self.pop_size)
        pop_spikes_temp = (
            [
                spike_rate[:, (i * self.pop_size): (i * self.pop_size) + self.pop_size].sum(dim=1)
                for i in range(pop_num)
            ]
        )
        pop_spikes = torch.stack(pop_spikes_temp, dim=-1)
        predict = torch.argmax(pop_spikes,
                               dim=1)  # return the indices of the maximum values of a tensor across columns.
        reward = self.punish_signal * torch.ones(predict.shape, device=device)
        flag = torch.tensor([predict[i] == target[i] for i in range(predict.size(0))])
        reward[flag] = self.reward_signal
        if len(reward) > 1:
            reward = reward.mean()
        return reward


Reward.register('global_reward', Global_Reward)


class DA_Reward(Reward):

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(DA_Reward, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.pop_size = kwargs.get('pop_size', 1)
        self.reward_signal = kwargs.get('reward_signal', 1)
        self.punish_signal = kwargs.get('punish_signal', 0)

    def torch_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        spike_rate = record.sum(0)
        pop_num = int(self.num / self.pop_size)
        pop_spikes_temp = (
            [
                spike_rate[:, (i * self.pop_size): (i * self.pop_size) + self.pop_size].sum(dim=1)
                for i in range(pop_num)
            ]
        )
        pop_spikes = torch.stack(pop_spikes_temp, dim=-1)
        predict = torch.argmax(pop_spikes,
                               dim=1)  # return the indices of the maximum values of a tensor across columns.
        reward = self.punish_signal * torch.ones(spike_rate.shape, device=device)
        for i in range(len(target)):
            if predict[i] == target[i]:
                reward[i, predict[i] * self.pop_size:(predict[i] + 1) * self.pop_size] = self.reward_signal
        if reward.size(0) > 1:
            reward = reward.sum(dim=0).unsqueeze(dim=0)
        return reward


Reward.register('da_reward', DA_Reward)


class XOR_Reward(Reward):

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(XOR_Reward, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type,
                                         **kwargs)
        self.pop_size = kwargs.get('pop_size', 1)
        self.reward_signal = kwargs.get('reward_signal', 1)
        self.punish_signal = kwargs.get('punish_signal', -1)

    def torch_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        spike_rate = record.sum(0)
        pop_num = int(self.num / self.pop_size)
        pop_spikes_temp = (
            [
                spike_rate[:, (i * self.pop_size): (i * self.pop_size) + self.pop_size].sum(dim=1)
                for i in range(pop_num)
            ]
        )
        pop_spikes = torch.stack(pop_spikes_temp, dim=-1)
        reward = self.punish_signal * torch.ones(pop_spikes.shape, device=device)
        if target == 1:
            if pop_spikes > 0:
                reward = torch.tensor(self.reward_signal, device=device)

        elif target == 0:
            if pop_spikes > 0:
                reward = torch.tensor(self.punish_signal, device=device)
        return reward


Reward.register('xor_reward', XOR_Reward)


class Environment_Reward(Reward):

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Environment_Reward, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type,
                                                 **kwargs)

    def torch_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        reward = torch.tensor(target, device=device, dtype=self.data_type)
        return reward


Reward.register('environment_reward', Environment_Reward)


class Classifiy_Reward(Reward):

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Classifiy_Reward, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type,
                                               **kwargs)
        self.beta = 0.99
        self.out = 0
        # self.pos_reward = 0
        # self.neg_reward = 0
        self.rewards = []

    def init_state(self):
        super(Classifiy_Reward, self).init_state()
        self.out = 0
        # self.pos_reward = 0
        # self.neg_reward = 0
        self.rewards = []

    def torch_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        target_index = torch.tensor(target, device=device).view(-1, 1)
        output = torch.mean(record, 0)
        mask = torch.zeros_like(output)
        mask.scatter_(1, target_index, 1)
        self.out = self.beta * self.out + output
        output = (self.out - torch.mean(self.out).detach()) / (torch.std(self.out).detach() + 0.1)
        softmax = torch.softmax(output + 1.0e-8, 1)

        # self.pos_reward = self.beta*self.pos_reward + mask*output
        # self.neg_reward = self.beta*self.neg_reward + (1-mask)*output

        reward = (1 - softmax) * mask - softmax * (1 - mask)
        # print(torch.mean(reward))
        # self.rewards.append(reward)
        return reward


Reward.register('classify_reward', Classifiy_Reward)
