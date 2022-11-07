# encoding: utf-8
"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: Actions.py
@time:2021/12/3 9:22
@description:
"""
from .Node import Action
import torch
import numpy as np
class PopulationRate_Action(Action):
    """
    Selects an action probabilistically based on output spiking activity of population.
    """
    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(PopulationRate_Action, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, record, target, device):
        """
        Args:
            record: spiking activity of output layer. The shape of record is (time_step, batch_size, n_neurons)
        Returns:
            Action sampled from population output activity.
        """

        assert record.shape[2] % self.num == 0, (
            f"Output layer size of {record.shape[2]} is not divisible by action space size of"
            f" {self.num}."
        )

        pop_size = int(record.shape[2] / self.num)
        spike_num = record.sum().type(self._backend.data_type)

        # Choose action based on population's spiking.
        if spike_num == 0:
            action = torch.randint(low=0, high=self.num, size=(1,))[0]
        else:
            pop_spikes = torch.tensor(
                [
                    record[:, :, (i * pop_size): (i * pop_size) + pop_size].sum()
                    for i in range(self.num)
                ],
                device=record.device,
            ).type(self._backend.data_type)
            # multinomial(input, num_samples,replacement=False), 采样的时候是根据输入张量的数值当做权重来进行抽样的, 返回index, 数值越大, 抽到的可能性越大, 如果是0 则不会抽到
            action = torch.multinomial((pop_spikes / spike_num).view(-1), 1)[0].item()
        return action

Action.register('pop_rate_action', PopulationRate_Action)

class Softmax_Action(Action):

    """
    Selects an action using softmax function based on spiking of output layer.
    """

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Softmax_Action, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, record, target, device):
        """
        Args:
            record: spiking activity of output layer.
        Returns:
            Action sampled from softmax over spiking activity of output layer.
        """
        assert (
            record.shape[2] == self.num
        ), "Output layer size is not equal to the size of the action space."
        spikes = torch.sum(record, dim=0)
        probabilities = torch.softmax(spikes, dim=0)
        return torch.multinomial(probabilities, num_samples=1).item()

Action.register('softmax_action', Softmax_Action)


class Highest_Spikes_Action(Action):

    """
    Selects an action that has the highest firing rate. In case of equal spiking select randomly
    """

    def __init__(self,shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Highest_Spikes_Action, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, record, target, device):
        """
        Args:
            record: spiking activity of output layer.
        Returns:
            Action sampled from highest activities of output layer.
        """

        assert (
           record.shape[2] == self.num
        ), "Output layer size is not equal to the size of the action space."


        spikes = torch.sum(record, dim=0).squeeze()
        action = torch.where(spikes == spikes.max())[0]
        if torch.sum(spikes) == 0:
            action[0] = torch.randint(low=0, high=1, size=(1,))[0]

        return action[0]

Action.register('highest_spikes_action', Highest_Spikes_Action)

class Highest_Voltage_Action(Action):

    """
    Selects an action that has the highest voltage. In case of equal spiking select randomly
    """
    steps_done = 0
    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Highest_Voltage_Action, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.seed = kwargs.get('seed', 1)

    def torch_coding(self, record, target, device):
        """
        Args:
            record: voltage of output layer.
        Returns:
            Action sampled from highest voltage of output layer.
        """
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 200

        assert (
           record.shape[2] == self.num
        ), "Output layer size is not equal to the size of the action space."

        import random
        import math
        random.seed(self.seed)
        sample = random.random()

        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * Highest_Voltage_Action.steps_done / EPS_DECAY)
        if record.shape[1] == 1:
            Highest_Voltage_Action.steps_done += 1
        if sample > eps_threshold:
            final_step_voltage = record[-1, :, :]
            action = final_step_voltage.max(1)[1]  # max(dim)返回的是value, index
            return action[0]
        else:
            return torch.randint(low=0, high=self.num, size=(1,))[0]

Action.register('highest_voltage_action', Highest_Voltage_Action)


class First_Spike_Action(Action):

    """
    Selects an action with the highst spikes. In case of equal spiking select randomly
    """

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(First_Spike_Action, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, record, target, device):
        """
        Args:
            record: spiking activity of output layer.
        Returns:
            Action sampled from first spike of output layer.
        """

        assert (
            record.shape[2] == self.num
        ), "Output layer size is not equal to the size of the action space."

        spikes = record.squeeze().nonzero()
        if spikes.shape[0] == 0:
            action = torch.randint(low=0, high=1, size=(1,))[0]
        else:
            action = spikes[0, 1]

        return action

Action.register('first_spike_action', First_Spike_Action)


class Random_Action(Action):

    """
    Selects an action randomly from the action space.
    """

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'), coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Random_Action, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def torch_coding(self, record, target, device):
        """
        Used for the PyTorch backend

        Args:
           record: spiking activity of output layer.
        Returns:
           Action sampled from action space randomly.
       """
        return torch.randint(low=0, high=self.num, size=(1,))[0]

    def numpy_action(self, record):
        """
        Args:
           record: spiking activity of output layer.
        Returns:
           Action sampled from action space randomly.
       """
        return np.random.choice(self.num)

Action.register('random_action', Random_Action)
