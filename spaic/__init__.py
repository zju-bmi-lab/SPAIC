# -*- coding: utf-8 -*-
"""
Created on 2020/8/11
@project: SNNFlow
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description: 
"""
# print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
from .Network import Network, Connection, Assembly
from .Neuron import NeuronGroup

from .Neuron import Node, Encoders, Decoders, Generators
from .Neuron.Node import Encoder, Decoder, Generator
from .Network.BaseModule import BaseModule
from .Simulation.Backend import Backend
from .Simulation.Torch_Backend import Torch_Backend
# from .Simulation.Tensorflow_Backend import Tensorflow_Backend
from .Monitor.Monitor import StateMonitor, SpikeMonitor
from .Learning.STCA_Learner import STCA
from .Learning.Learner import Learner
from .IO.Dataset import Dataset, CustomDataset, MNIST, FashionMNIST, OctMNIST, PathMNIST, AudioMNIST, cifar10, SHD, SSC
from .IO.Dataloader import Dataloader
from .IO.Pipeline import RLPipeline, ReplayMemory
from .IO.Environment import GymEnvironment



# ============== Global variable block for network building ==================
global_assembly_context_list = list()
global_assembly_context_omit_start = 10000000000000000000000
global_assembly_context_omit_end = -1
global_assembly_init_count = 0
global_module_name_count = 0
debug_grad = dict()

