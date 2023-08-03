# -*- coding: utf-8 -*-
"""
Created on 2020/9/9
@project: SPAIC
@filename: BaseModule
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description: 
"""
from abc import abstractmethod
from collections import OrderedDict
from typing import Optional, Any, List
from dataclasses import dataclass, field
from copy import copy
import uuid
from ..Backend.Backend import Backend
from .Operator import Op

from .. import global_module_name_count

global global_module_name_count


class BaseModule():
    '''
    Base class for all snn modules (assemblies, connection, learner, monitor, piplines).

    '''
    _Module_Count = 0
    _class_label = '<bm>'

    def __init__(self):
        self.id = None
        self.name = None
        self.enabled = True
        self.training = True
        self._backend: Backend = None
        self._supers = []
        self._var_names = []
        self._var_dict = dict()
        self._ops = list()
        self.prefer_device = None

    @abstractmethod
    def build(self, backend):
        NotImplementedError()

    @abstractmethod
    def get_str(self, level):
        NotImplementedError()

    def set_name(self, given_name):
        global global_module_name_count
        if isinstance(given_name, str):
            if self.name is None:
                self.name = given_name
            elif 'autoname' in self.name:
                self.name = given_name

        # elif isinstance(given_name, list):
        #     context = given_name[-1]
        #     spaic.global_module_name_count += 1
        #     self.name = context.name +'subgroup' + str(spaic.global_module_name_count)

        else:
            global_module_name_count += 1
            self.name = 'autoname' + str(global_module_name_count)

        return self.name

    def set_id(self):
        if len(self._supers) == 0:
            self.id = self.name + self.__class__._class_label
        else:
            super_ids = []
            for super in self._supers:
                if super.id is not None:
                    super_ids.append(super.id)
                else:
                    super_ids.append(super.set_id())

            self.id = self.name + self.__class__._class_label
            if len(super_ids) == 1:
                self.id = super_ids[0] + '_' + self.id
            else:
                pre_id = '/'
                for prefix in super_ids:
                    pre_id += prefix + ','
                pre_id += '/'
                self.id = pre_id + '_' + self.id
        return self.id

    def set_build_level(self, level):

        if self.build_level < 0:
            self.build_level = level
        elif self.build_level > level:
            self.build_level = level

    def variable_to_backend(self, name, shape, value=None, is_parameter=False, is_sparse=False, init=None,
                            init_param=None,
                            min=None, max=None, is_constant=False, prefer_device=None):
        self._var_names.append(name)
        self._var_dict[name] = self._backend.add_variable(self, name, shape, value, is_parameter, is_sparse, init,
                                                          init_param, min, max, is_constant, prefer_device)
        return self._var_dict[name]

    def op_to_backend(self, outputs: list, func: callable, inputs: list):
        # check if the inputs and outputs variables belongs to this module object, if backend don't have this variable it will be added the module label
        if isinstance(inputs, list):
            for ind, input_name in enumerate(inputs):
                if '[updated]' in input_name:
                    input_name = input_name.replace('[updated]', '')
                if not self._backend.has_variable(input_name):
                    input_name = self._add_label(input_name)
                    inputs[ind] = input_name
                    # assert self._backend.has_variable(input_name)
        elif isinstance(inputs, str):
            if not self._backend.has_variable(inputs):
                inputs = self._add_label(inputs)
                inputs = [inputs]
                # assert self._backend.has_variable(inputs[-1])
        else:
            raise ValueError("the preprocessing of op_to_backend do not support this input type")
        if isinstance(outputs, list):
            for ind, output_name in enumerate(outputs):
                if not self._backend.has_variable(output_name):
                    output_name = self._add_label(output_name)
                    outputs[ind] = output_name
                    # assert self._backend.has_variable(output_name)
        elif isinstance(outputs, str):
            if not self._backend.has_variable(outputs):
                outputs = self._add_label(outputs)
                outputs = [outputs]
                # assert self._backend.has_variable(outputs[-1])
        else:
            raise ValueError("the preprocessing of op_to_backend do not support this input type")

        addcode_op = Op(outputs, func, inputs, owner=self, operation_type='_operations')
        self._backend.add_operation(addcode_op)

    def init_op_to_backend(self, outputs, func, inputs, prefer_device=0):
        addcode_op = Op(outputs, func, inputs, place=prefer_device, owner=self, operation_type='_operations')
        self._backend.register_initial(addcode_op)

    # adding label of the module object, cut from neurongroup and generalized to all Modules
    def _add_label(self, key):
        if isinstance(key, str):
            if key == '[dt]':
                return key
            elif '[updated]' in key:
                return self.id + ':' + '{' + key.replace('[updated]', "") + '}' + '[updated]'
            else:
                return self.id + ':' + '{' + key + '}'
        elif isinstance(key, VariableAgent):
            return key.var_name
        else:
            raise ValueError(" the key data type is not supported for add_label")

    def get_full_name(self, name):
        name = '{' + name + '}'
        full_name = None
        for key in self._var_names:
            if name in key:
                if full_name is not None:
                    raise ValueError("multiple variable with same name in this module")
                else:
                    full_name = key
        return full_name

    def get_value(self, name):
        full_name = self.get_full_name(name)
        if full_name is None:
            raise ValueError("No such variable name in this module")
        else:
            return self._var_dict[full_name].value

    def set_value(self, name, value):
        name = '{' + name + '}'
        full_name = None
        for key in self._var_names:
            if name in key:
                if full_name is not None:
                    raise ValueError("multiple variable with same name in this module")
                else:
                    full_name = key
        if full_name is None:
            raise ValueError("No such variable name in this module")
        else:
            self._var_dict[full_name].value = value

    def _direct_set_variable(self, name, variable):
        # only for debug at the beginning of the network run
        name = '{' + name + '}'
        full_name = None
        for key in self._var_names:
            if name in key:
                if full_name is not None:
                    raise ValueError("multiple variable with same name in this module")
                else:
                    full_name = key
        if full_name is None:
            raise ValueError("No such variable name in this module")
        else:
            is_parameter = self._var_dict[full_name]._is_parameter
            if is_parameter:
                self._backend._parameters_dict[full_name] = variable
            else:
                self._backend._InitVariables_dict[full_name] = variable


class VariableAgent(object):
    def __init__(self, backend, var_name, is_parameter=False, dict_label=None):
        super(VariableAgent, self).__init__()
        assert isinstance(backend, Backend)
        self._backend: Backend = backend
        self._var_name = var_name
        self._is_parameter = is_parameter
        self.data_type = None
        self.device = None
        self.dict_label = dict_label

        self.set_funcs = []
        self.get_funcs = []

    @property
    def var_name(self):
        return self._var_name

    def new_labeled_agent(self, dict_label):
        assert (dict_label == 'variables_dict' or dict_label == 'update_dict'
                or dict_label == 'reduce_dict' or dict_label == 'temp_dict')
        agent = copy(self)
        agent.dict_label = dict_label
        return agent

    @property
    def value(self):
        if self.dict_label is None:
            return self._backend.get_varialble(self._var_name)
        elif self.dict_label == 'variables_dict':
            return self._backend._variables[self._var_name]
        elif self.dict_label == 'update_dict':
            return self._backend._update_dict[self._var_name]
        elif self.dict_label == 'reduce_dict':
            return self._backend._reduce_dict[self._var_name]
        elif self.dict_label == 'temp_dict':
            return self._backend._temp_dict[self._var_name]
        else:
            raise ValueError("can't find variable %s" % self._var_name)

    @value.setter
    def value(self, value):
        if self.dict_label is None:
            self._backend.set_variable_value(self._var_name, value, self._is_parameter)
        elif self.dict_label == 'update_dict':
            self._backend._update_dict[self._var_name] = value
        elif self.dict_label == 'reduce_dict':
            if self._var_name in self._backend._reduce_dict:
                self._backend._reduce_dict[self._var_name].append(value)
            else:
                self._backend._reduce_dict[self._var_name] = [value]
        elif self.dict_label == 'temp_dict':
            self._backend._temp_dict[self._var_name] = value
        elif self.dict_label == 'variables_dict':
            self._backend._variables[self._var_name] = value
        else:
            raise ValueError("can't set value of variable %s" % self._var_name)


class OperationCommand(object):
    def __init__(self, front_module, output, function, input):
        super(OperationCommand, self).__init__()
        assert isinstance(front_module, BaseModule)
        assert isinstance(output, list)
        assert isinstance(function, str) or callable(function)
        assert isinstance(input, list)

        self.front_module = front_module
        self.output = output
        self.function = function
        self.input = input
        self.training_only = False

    @property
    def enabled(self):
        if self.training_only:
            return self.front_module.enabled and self.front_module.training
        else:
            return self.front_module.enabled



# class NetModule(BaseModule):
#     '''
#     Base class for snn network modules: assemblies, connection
#     '''
#
#     def __init__(self):
#         super(NetModule, self).__init__()
#
#         self.trainable_parameter_names = OrderedDict()
#
#     def add_trainable_names(self, name):
#         pass
