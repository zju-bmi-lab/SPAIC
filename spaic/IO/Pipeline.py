# -*- coding: utf-8 -*-
"""
Created on 2020/8/17
@project: SPAIC
@filename: IO
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
"""
from abc import abstractmethod
from ..Neuron.Node import Action, Encoder, Reward
import numpy as np
from tqdm import tqdm
import torch
from collections import namedtuple, deque
import random

class Pipline:
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def update_step(self, batch, **kwargs):
        """
        Perform a pass of the network given the input batch.

        :param batch: The current batch. This could be anything as long as the subclass
            agrees upon the format in some way.
        :return: Any output that is need for recording purposes.
        """
        raise NotImplementedError("You need to provide a step_ method.")

class RLPipeline(Pipline):
    def __init__(self, network, environment, time=None, **kwargs):
        self.network = network
        self._backend = self.network._backend
        self.sim_name = self._backend.backend_name
        self.device = self._backend.device
        self.time = time
        self.time_step = int(self.time / self._backend.dt)
        self.actuator = None
        self.encoder = None
        self.rewarder = None
        # Get actuator and encoder
        for group in self.network.get_groups():
            if isinstance(group, Action):
                self.actuator = group
            if isinstance(group, Encoder):
                self.encoder = group
            if isinstance(group, Reward):
                self.rewarder = group

        if self.actuator is None:
            raise ValueError('Lack of Action object')
        if self.encoder is None:
            raise ValueError('Lack of Encoder object')

        self.network.build(self._backend)

        self.state = np.zeros(self.encoder.num)
        self.environment = environment
        if self.environment.shape is not None and len(self.environment.shape) >= 2:
            self.conv_state = True
        else:
            self.conv_state = False
        self.step_count = 0
        self.episode = 0
        self.num_episodes = kwargs.get('num_episodes', 100)

        self.accumulated_reward = 0.0
        self.reward_list = []
        self.action = -1
        self.last_action = -1

        self.action_repeat_count = 0
        self.action_repeat = kwargs.get('action_repeat', 2)

        self.probability_random_action = kwargs.get('probability_random_action', 0.0)
        self.render_interval = kwargs.get('render_interval', None)
        self.reward_delay = kwargs.get('reward_delay', None)
        self.replay_memory = kwargs.get('replay_memory', False)
        if self.replay_memory:
            self.memory_capacity = kwargs.get('memory_capacity', 10000)
            self.memory_pool = ReplayMemory(self.memory_capacity)

        if self.reward_delay is not None:
            assert self.reward_delay > 0
            self.rewards = np.zeros(self.reward_delay)

    def env_step(self):
        """
        Single step of the environment which includes rendering, getting and performing
        the action, and accumulating/delaying rewards.

        Returns:
            An OpenAI ``gym`` compatible tuple (next_state, reward, done).
        """
        # Render the environment.
        if (self.render_interval is not None and self.step_count % self.render_interval == 0):
            self.environment.render()

        # Get action
        self.last_action = self.action
        # action sampled from the action space with a certain probability
        if np.random.rand(1) < self.probability_random_action:
            self.action = np.random.randint(
                low=0, high=self.environment.action_num, size=(1,)
            )[0]
        elif self.action_repeat_count > self.action_repeat:
            if self.last_action == 0:
                self.action = 1
                tqdm.write(f"Act -> too many times {self.last_action} ")
            else:
                self.action = np.random.randint(
                    low=0, high=self.environment.action_num, size=(1,)
                )[0]
                tqdm.write(f"Act -> too many times {self.last_action} ")
        else:
            if self.sim_name == 'pytorch':
                self.action = int(self.actuator.action)  # Get action from the predict result of action

        if self.last_action == self.action:
            self.action_repeat_count += 1
        else:
            self.action_repeat_count = 0

        # Run a step of the environment.
        next_state, reward, done, _ = self.environment.step(self.action)
        if self.conv_state:
            next_state = next_state[np.newaxis, :]

        if done:
            next_state = None

        # Set reward in case of delay.
        if self.reward_delay is not None:
            if self.sim_name == 'pytorch':
                self.rewards = torch.tensor([reward, *self.rewards[1:]], device=self.device)
            reward = self.rewards[-1]

        # Accumulate reward.
        self.accumulated_reward += reward
        if self.replay_memory:
            self.memory_pool.push(self.state, self.action, next_state, reward)
        self.state = next_state

        return next_state, reward, done

    def update_step(self, gym_batch, **kwargs):
        """
        Run a single iteration of the network and update it and the reward list when
        done.

        Args:
            gym_batch (tuple): An OpenAI ``gym`` compatible tuple (next_state, reward, done).
        """
        self.step_count += 1
        next_state, reward, done = gym_batch

        if done:
            self.reward_list.append(self.accumulated_reward)
            print("Episode finished after {} steps".format(self.step_count))
        else:
            # Add a placeholder for batch_size
            next_state = next_state[np.newaxis, :]

            # Place the observations into the network.
            # self.network.input(next_state)
            self.encoder(next_state)
            if self.rewarder is not None:
                self.rewarder(reward)

            self.network.run(self.time)

    def reset_pipeline(self):
        """
        Reset the pipeline.
        """
        self.environment.reset()
        self.accumulated_reward = 0.0
        self.step_count = 0
        self.action = -1
        self.last_action = -1
        self.action_repeat_count = 0
        if self.conv_state:
            self.state = np.zeros(self.environment.shape)
            self.state = self.state[np.newaxis, :]
        else:
            self.state = np.zeros(self.encoder.num)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)




