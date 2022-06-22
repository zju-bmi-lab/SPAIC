# -*- coding: utf-8 -*-
"""
Created on 2020/8/11
@project: SPAIC
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description: 
"""
# print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
from .Network import Network, Connection, Assembly, Projection, Synapse
from .Neuron import NeuronGroup

from .Neuron import Node, Encoders, Decoders, Generators, Rewards, Actions
from .Neuron.Node import Encoder, Decoder, Generator, Reward, Action
from .Neuron.Module import Module
from .Network.BaseModule import BaseModule
from .Network.ConnectPolicy import ExcludedTypePolicy, IndexConnectPolicy, IncludedTypePolicy
from .Backend.Backend import Backend
from .Backend.Torch_Backend import Torch_Backend
# from .Backend.Tensorflow_Backend import Tensorflow_Backend
from .Monitor.Monitor import StateMonitor, SpikeMonitor
#from .Learning.Rate_Modulation import Rate_Modulate
from .Learning.STCA_Learner import STCA
#from .Learning.TRUE_Learner import TRUE_SpikeProp
#from .Learning.BioHashSTDP_Learner import BioHash
from .Learning.Learner import Learner
from .IO.Dataset import Dataset, CustomDataset, MNIST, FashionMNIST, OctMNIST, PathMNIST, MNISTVoices, cifar10, SHD, SSC
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

