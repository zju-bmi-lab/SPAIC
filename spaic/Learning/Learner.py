# -*- coding: utf-8 -*-
"""
Created on 2020/8/12
@project: SPAIC
@filename: Learner
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
定义学习模块，包括各种学习算法对仿真计算过程中插入的各种计算模块，以及记录需要学习连接的接口
"""

from ..Network.Assembly import BaseModule
from abc import ABC, abstractmethod
from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np

from ..Backend.Backend import Backend
from ..Network.Assembly import Assembly
from ..Network.Topology import Connection, Projection
from ..Neuron.Neuron import NeuronGroup
from ..Neuron.Module import Module
from ..Neuron.Node import Node


class Learner(BaseModule, ABC):

    '''
        Base learner model for all the learner model.

        Args:
            parameters(dict) : The parameters for learner.
            super_parameters(dict) : Super parameters for future use.
            backend_functions(dict) : Contains all the learner model we can choose.

            name(str) : The typical name for the learner.
            preferred_backend(str) : Choose which kind of backend to use. Like "Pytorch", "Tensorflow" or "Jax".
            trainable_groups(dict) : Trainable container, includes nodes and layers to train.
            trainable_connections(dict) : Trainable container, includes connections to train.

            init_trainable: The initial state of this learner of whether it is trainable.

        Methods:
            add_trainable(self, trainable) : Add target object (Network, Assembly, Connection,
                            or list of them) to the trainable container
            build(self, backend) : Build Learner, choose the backend as user wish, if we have already finished the api.

    '''
    _class_label = '<learner>'
    learning_algorithms = dict()
    learning_optims = dict()
    optim_dict = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW,
                  'SparseAdam': torch.optim.SparseAdam, 'Adamax': torch.optim.Adamax,
                  'ASGD': torch.optim.ASGD, 'LBFGS': torch.optim.LBFGS,
                  'RMSprop': torch.optim.RMSprop, 'Rpop': torch.optim.Rprop,
                  'SGD': torch.optim.SGD, 'Adadelta': torch.optim.Adadelta,
                  'Adagrad': torch.optim.Adagrad}

    lr_schedule_dict = {'LambdaLR': torch.optim.lr_scheduler.LambdaLR,
                        'StepLR': torch.optim.lr_scheduler.StepLR,
                        'MultiStepLR': torch.optim.lr_scheduler.MultiStepLR,
                        'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
                        'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
                        'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
                        'CyclicLR': torch.optim.lr_scheduler.CyclicLR,
                        'CosineAnnealingWarmRestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts}

    def __init__(self, trainable=None, pathway=None, algorithm=('STCA', 'STBP', 'RSTDP', '...'), name=None, **kwargs):

        super(Learner, self).__init__()

        self.parameters = kwargs
        self.super_parameters = OrderedDict()
        self.backend_functions = OrderedDict()

        self.name = self.set_name(name)
        self.optim_name = None
        self.optim = None
        self.lr_schedule_name = None
        self.gradient_based = True
        self.prefered_backend = ['pytorch']
        self.trainable_groups = OrderedDict()
        self.trainable_connections = OrderedDict()
        self.trainable_nodes = OrderedDict()
        self.trainable_modules = OrderedDict()
        self.trainable_others = OrderedDict()
        self.init_trainable = trainable
        self.training_param_name = []
        self.param_run_update = kwargs.get("param_run_update", False)
        self._variables = dict()
        self._constant_variables = dict()
        self._tau_constant_variables = dict()
        self._tau_membrane_variables = dict()


        self.pathway_groups = OrderedDict()
        self.pathway_connections = OrderedDict()
        self.pathway_nodes = OrderedDict()
        self.pathway_modules = OrderedDict()
        self.pathway_others = OrderedDict()
        self.init_pathway = pathway
        self._operations = []
        self._active = True


    def add_trainable(self, trainable: list):
        '''
            Add target object (Assembly, Connection, or list of them) to the trainable container
            Args:
                trainable(list) : The trainable target waiting for added.
        '''
        if not isinstance(trainable, list):
            trainable = [trainable]

        for target in trainable:
            if isinstance(target, NeuronGroup):
                self.trainable_groups[target.id] = target
            elif isinstance(target, Connection):
                self.trainable_connections[target.id] = target
            elif isinstance(target, Node):
                self.trainable_nodes[target.id] = target
            elif isinstance(target, Module):
                self.trainable_modules[target.id] = target
            elif isinstance(target, Assembly):
                for sub_t in target.get_groups():
                    trainable.append(sub_t)
                for sub_t in target.get_connections():
                    trainable.append(sub_t)
            elif isinstance(target, Projection):
                for sub_t in target.get_connections():
                    trainable.append(sub_t)
            elif isinstance(target, BaseModule):
                self.trainable_others[target.id] = target


    def add_pathway(self, pathway: list):
        '''
            Add target object (Assembly, Connection, or list of them) to the pathway container
            Args:
                pathway(list) : The pathway target waiting for added.
        '''
        if not isinstance(pathway, list):
            pathway = [pathway]

        for target in pathway:
            if isinstance(target, NeuronGroup):
                self.pathway_groups[target.id] = target
            elif isinstance(target, Connection):
                self.pathway_connections[target.id] = target
            elif isinstance(target, Node):
                self.pathway_nodes[target.id] = target
            elif isinstance(target, Module):
                self.pathway_modules[target.id] = target
            elif isinstance(target, Assembly):
                for sub_t in target.get_groups():
                    pathway.append(sub_t)
                for sub_t in target.get_connections():
                    pathway.append(sub_t)
            elif isinstance(target, Projection):
                for sub_t in target.get_connections():
                    pathway.append(sub_t)
            elif isinstance(target, BaseModule):
                self.pathway_others[target.id] = target

    def build(self, backend: Backend):
        '''
            Build Learner, choose the backend as user wish, if we have already finished the api.
            Args:
                backend(backend) : Backend we have.
        '''
        if self.init_trainable is not None:  # If user has given the 'trainable' parameter.
            self.add_trainable(self.init_trainable)
        if self.init_pathway is not None:
            self.add_pathway(self.init_pathway)

        if backend.backend_name in self.prefered_backend:
            self._backend = backend
        else:
            raise ValueError(
                "the backend %s is not supported by the learning rule %s" % (backend.backend_name, self.name))
        if self.optim_name is not None:
            self.build_optimizer()
        else:
            self.get_param()

        if self.lr_schedule_name is not None:
            self.build_lr_schedule()

        # set all trainable or pathway compontents to requires_grad=Ture
        if self.gradient_based:
            for node in self.trainable_nodes.values():
                for op in node._ops:
                    op.requires_grad = True
            for group in self.trainable_groups.values():
                for op in group._ops:
                    op.requires_grad = True
            for con in self.trainable_connections.values():
                for op in con._ops:
                    op.requires_grad = True
            for other in self.trainable_others.values():
                for op in other._ops:
                    op.requires_grad = True
            for mod in self.trainable_modules.values():
                mod.module.requires_grad_()
                for op in mod._ops:
                    op.requires_grad = True

            for node in self.pathway_nodes.values():
                for op in node._ops:
                    op.requires_grad = True
            for group in self.pathway_groups.values():
                for op in group._ops:
                    op.requires_grad = True
            for con in self.pathway_connections.values():
                for op in con._ops:
                    op.requires_grad = True
            for other in self.pathway_others.values():
                for op in other._ops:
                    op.requires_grad = True
            for mod in self.pathway_modules.values():
                mod.module.requires_grad_()
                for op in mod._ops:
                    op.requires_grad = True


        # add learner variables to the backend
        self.dt = backend.dt
        for (key, tau_var) in self._tau_constant_variables.items():
            tau_var = np.exp(-self.dt / tau_var)
            shape = ()
            self.variable_to_backend(self._add_label(key), shape, value=tau_var)
        for (key, tau_membrane_var) in self._tau_membrane_variables.items():
            tau_membrane_var = self.dt/tau_membrane_var
            shape = ()  # (1, neuron_num)
            self.variable_to_backend(self._add_label(key), shape, value=tau_membrane_var)
        for (key, var) in self._constant_variables.items():
            if isinstance(var, np.ndarray):
                if var.size > 1:
                    var_shape = var.shape
                    shape = (1, *var_shape)  # (1, shape)
                else:
                    shape = ()
            elif isinstance(var, list):
                if len(var) > 1:
                    var_len = len(var)
                    shape = (1, var_len)  # (1, shape)
                else:
                    shape = ()
            else:
                shape = ()
            self.variable_to_backend(self._add_label(key), shape, value=var)
        for (key, var) in self._variables.items():
            if isinstance(var, np.ndarray):
                if var.size > 1:
                    var_shape = var.shape
                    shape = (1, *var_shape)  # (1, shape)
                else:
                    shape = ()
            elif isinstance(var, list):
                if len(var) > 1:
                    var_len = len(var)
                    shape = (1, var_len)  # (1, shape)
                else:
                    shape = ()
            else:
                shape = ()
            self.variable_to_backend(self._add_label(key), shape, value=var)

        # build custom learning rules
        if self.is_overridden(self.custom_rule):
            self.custom_rule(backend)
        if self.is_overridden(self.connection_rule):
            for conn in self.trainable_connections.values():
                self.connection_rule(conn, backend, 'trainable')
            for conn in self.pathway_connections.values():
                self.connection_rule(conn, backend, 'pathway')
        if self.is_overridden(self.neuron_rule):
            for neuron in self.trainable_groups.values():
                self.neuron_rule(neuron, backend, 'trainable')
            for neuron in self.pathway_groups.values():
                self.neuron_rule(neuron, backend, 'pathway')



    def __new__(cls, trainable=None, pathway=None, algorithm=('STCA', 'STBP', 'RSTDP', '...'), **kwargs):
        if cls is not Learner:
            return super().__new__(cls)

        if algorithm == ('STCA', 'STBP', 'RSTDP', '...'):
            return super().__new__(cls)
        elif algorithm.lower() in cls.learning_algorithms:
            return cls.learning_algorithms[algorithm.lower()](trainable=trainable, pathway=None, **kwargs)
        else:
            raise ValueError("No algorithm %s in algorithm list" % algorithm)

    def active(self):
        self._active = True
        #TODO: setting all learnerable and pathway into gradient = True
    def deactive(self):
        self._active = False
        #TODO: setting all learnerable and pathway into gradient = Fasle

    def set_optimizer(self, optim_name, optim_lr, **kwargs):
        self.optim_lr = optim_lr
        self.optim_para = kwargs
        self.optim_name = optim_name

        if self.optim_name not in Learner.optim_dict.keys():
            raise ValueError("No optim %s in optim list" % Learner.optim_dict)

    def set_schedule(self, lr_schedule_name, **kwargs):

        self.lr_schedule_para = kwargs

        self.lr_schedule_name = lr_schedule_name

        if self.lr_schedule_name not in Learner.lr_schedule_dict.keys():
            raise ValueError("No lr_schedule %s in lr_schedule list")

    def get_param(self):
        param = list()
        var_name = list()
        for key, conn in self.trainable_connections.items():
            for name in conn._var_names:
                var_name.append(name)
        for key, node in self.trainable_nodes.items():
            for name in node._var_names:
                var_name.append(name)
        for key, group in self.trainable_groups.items():
            for name in group._var_names:
                var_name.append(name)
        self.training_param_name = []
        for key, value in self._backend._parameters_dict.items():
            if key in var_name:
                value.requires_grad = True
                param.append(value)
                self.training_param_name.append(key)
            else:
                value.requires_grad =False


        for mod in self.trainable_modules.values():
            param.extend(mod.parameters)

        return param

    def get_varname(self, key):
        name = self.name + ':{' + key + '}'
        return name
    def get_var_names(self):
        return self._var_names

    def build_optimizer(self):

        param = self.get_param()
        self.optim = Learner.optim_dict[self.optim_name](param, self.optim_lr, **self.optim_para)

    def build_lr_schedule(self):

        self.schedule = Learner.lr_schedule_dict[self.lr_schedule_name](self.optim, **self.lr_schedule_para)

    def optim_step(self):
        if self.param_run_update:
            with torch.no_grad():
                for key, value in self._backend._parameters_dict.items():
                    if key in self.training_param_name:
                        varialbe_value = self._backend._variables[key]
                        if varialbe_value is not value:
                            value.data = varialbe_value.data
                            self._backend._variables[key] = value #Is this step needed?

        if self.optim is not None:
            self.optim.step()

    def optim_zero_grad(self):

        self.optim.zero_grad()

    def optim_schedule(self):

        self.schedule.step()

    @staticmethod
    def register(name, algorithm):
        name = name.lower()
        if name in Learner.learning_algorithms:
            raise ValueError(('A learning algorithm with the name "%s" has already been registered') % name)

        if not issubclass(algorithm, Learner):
            raise ValueError(
                ('Given algorithm of type %s does not seem to be a valid algorithm.' % str(type(algorithm))))

        Learner.learning_algorithms[name] = algorithm

    def custom_rule(self, backend: Backend):
        return None

    def connection_rule(self, con : Connection, backend: Backend, obj_type='trainable'):
        return None


    def neuron_rule(self, neuron: NeuronGroup, backend: Backend, obj_type='trainable'):
        return None


    def is_overridden(self, func):
        if isinstance(self, Learner):
            return False
        super_func = getattr(super(type(self), self), func.__name__)
        return super_func != func



class ReSuMe(Learner):

    def __init__(self):
        super(ReSuMe, self).__init__()
        pass


# Learner.register("ReSuMe", ReSuMe)


class FORCE(Learner):

    def __init__(self):
        super(FORCE, self).__init__()
        pass


# Learner.register("force", FORCE)
