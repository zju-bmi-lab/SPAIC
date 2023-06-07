# -*- coding: utf-8 -*-
"""
Created on 2020/8/12
@project: SPAIC
@filename: Environment
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
定义强化学习的环境交互模块
"""
from abc import ABC, abstractmethod
from .utils import RGBtoGray, GraytoBinary, reshape
# import gym
import numpy as np

'''
# Examples: initialize the environment of CartPole-v1
import gym
environment = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = environment.reset()
    for t in range(100):
        environment.render()
        print(observation)
        action = environment.action_space.sample()
        observation, reward, done, info = environment.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
environment.close()
'''

class BaseEnvironment(ABC):
    """
    Abstract environment class.
    """
    def __init__(self):
        pass

    @abstractmethod
    def step(self, action: int):
        """
        Abstract method for ``step()``.

        Args:
            action (int): action to take in environment.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Abstract method for ``reset()``.
        """
        pass

    @abstractmethod
    def render(self):
        """
        Abstract method for ``render()``.
        """
        pass

    @abstractmethod
    def seed(self, seed):
        """
        Abstract method for ``seed()``.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Abstract method for ``close()``.
        """
        pass

class GymEnvironment(BaseEnvironment):
    """
    Wrapper the OpenAI ``gym`` environments.
    """

    def __init__(self, name: str,  **kwargs):
        """
        Initializes the environment wrapper. This class makes the
        assumption that the OpenAI ``gym`` environment will provide an image
        of format HxW as an observation.

        Args:
            name (str): The name of an OpenAI ``gym`` environment.
            encoding (str): The key of encoding class which is used to encode observations into spike trains.

        Attributes:
            max_prob (float): Maximum spiking probability.
            clip_rewards (bool): Whether or not to use ``np.sign`` of rewards.
            binary (bool): Whether to convert the image to binary
        """
        import gym
        self.name = name
        self.environmet = gym.make(name)
        self.action_space = self.environmet.action_space
        self.action_num = self.action_space.n

        self.shape = kwargs.get('shape', None)
        self.binary = kwargs.get('binary', False)
        self.gray = kwargs.get('binary', True)
        self.flatten = kwargs.get('flatten', True)

        # Keyword arguments.
        self.max_prob = kwargs.get('max_prob', 1.0)
        self.clip_rewards = kwargs.get('clip_rewards', True)

        self.episode_step_count = 0

        self.obs = None
        self.reward = None

        assert (
            0.0 < self.max_prob <= 1.0
        ), "Maximum spiking probability must be in (0, 1]."

    def step(self, action):
        """
        Wrapper around the OpenAI ``gym`` environment ``step()`` function.

        Args:
            action (int): Action to take in the environment.
        Returns:
             Observation, reward, done flag, and information dictionary.
        """
        # Call gym's environment step function.
        self.obs, self.reward, self.done, info = self.environmet.step(action)

        if self.clip_rewards:
            self.reward = np.sign(self.reward)
        """
        After encoding the shape of 1D observations will become [Time_step, batch_size, length].
        2D observations are mono images. They will be flatten into 1D.
        3D observations are color images that will be converted to grayscale images and then will be flatten into 1D.
        """
        if len(self.obs.shape) >= 3 and self.gray:
            self.obs = RGBtoGray(self.obs)

        if self.binary:
            self.obs = GraytoBinary(self.obs)

        if self.shape is not None:
            if self.shape != self.obs.shape:
                self.obs = reshape(self.obs, self.shape)

        # Flatten
        if len(self.obs.shape) >= 2 and self.flatten:
            self.obs = self.obs.flatten()

        # Add the raw observation from the gym environment into the info for display.
        info['gym_obs'] = self.obs

        self.episode_step_count += 1

        # Return converted observations and other information.
        return self.obs, self.reward, self.done, info

    def reset(self):

        """
        Wrapper around the OpenAI ``gym`` environment ``reset()`` function.

        :return: Observation from the environment.
        """
        # Call gym's environment reset function.
        self.obs = self.environmet.reset()

        if len(self.obs.shape) >= 3 and self.gray:
            self.obs = RGBtoGray(self.obs)

        if self.binary:
            self.obs = GraytoBinary(self.obs)

        if self.shape is not None:
            self.shape = tuple(self.shape)
            if self.shape != self.obs.shape:
                self.obs = reshape(self.obs, self.shape)

        # Flatten
        if len(self.obs.shape) >= 2 and self.flatten:
            self.obs = self.obs.flatten()

        self.episode_step_count = 0

        return self.obs

    def render(self, mode):

        """
        Wrapper around the OpenAI ``gym`` environment ``render()`` function.
        """
        return self.environmet.render(mode)

    def seed(self, seed):
        """
        Wrapper around the OpenAI ``gym`` environment ``render()`` function.
        """
        self.environmet.seed(seed)

    def close(self):
        """
        Wrapper around the OpenAI ``gym`` environment ``close()`` function.
        """
        self.environmet.close()