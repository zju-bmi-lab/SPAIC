# -*- coding: utf-8 -*-
"""
Created on 2021/4/12
@project: SPAIC
@filename: Module
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
wrap around deep learning module such as a cnn network lstm cell
"""
import torch


from ..Network.Assembly import Assembly
from ..Network.Operator import Op


class Module(Assembly):
    _class_label = '<mod>'

    def __init__(self, module=None, name=None, input_targets=[], input_var_names=['O[updated]'], output_targets=None, output_var_names=['Isyn'], module_backend='pytorch'):
        super(Module, self).__init__(name)
        self.module: torch.nn.Module = module
        if isinstance(input_targets, list):
            self.input_targets = input_targets
        else:
            self.input_targets = [input_targets]
        if isinstance(input_var_names, list):
            self.input_var_names = input_var_names
        else:
            self.input_var_names = [input_var_names]

        if isinstance(output_targets, list):
            self.output_targets = output_targets
        else:
            self.output_targets = [output_targets]
        if isinstance(output_var_names, list):
            self.output_var_names = output_var_names
        else:
            self.output_var_names = [output_var_names]

        for in_targ in self.input_targets:
            in_targ.register_module(self, True)
        for out_targ in self.output_targets:
            out_targ.register_module(self, False)

        self.module_backend = module_backend


    def standalone_run(self, *args):
        return self.module(*args)

    def init_variable(self, var_names=None, var_shapes=None, var_value_dict=None):
        if var_names is None:
            self._var_names = []
        elif hasattr(var_names, '__iter__'):
            self._var_names = var_names
        else:
            self._var_names = [var_names]

        if var_shapes is None:
            self.var_shapes = []
        elif isinstance(var_shapes, list):
            self._var_shapes = var_shapes
        else:
            self._var_shapes = [var_shapes for _ in range(len(self._var_names))]
        self._var_values = []

        if var_value_dict is None:
            var_value_dict = []
        elif hasattr(var_value_dict, '__iter__'):
            var_value_dict = var_value_dict
        else:
            var_value_dict = [var_value_dict]
        for var_name in self._var_names:
            if var_name in var_value_dict:
                self._var_values.append(var_value_dict[var_name])
            else:
                self._var_values.append(0.0)

    def build(self, backend):

        # Add module owned variables to backend
        self._backend = backend
        self.init_variable()
        var_len = len(self._var_names)
        for ii in range(var_len):
            key = self.id + ":" + "{" + self._var_names[ii] + "}"
            shape = (1, *self._var_shapes[ii])
            self.variable_to_backend(key, shape, self._var_values[ii])

        # add standalone operation
        output_var_name = self.output_targets[0].id + ":" + "{" + self.output_var_names[0] + "}"
        self._var_names.append(output_var_name)
        input_var_names = []
        for input_target, input_name in zip(self.input_targets, self.input_var_names):
            input_var_name = input_target.id + ":" + "{" + input_name + "}"
            self._var_names.append(input_var_name)
            input_var_names.append(input_var_name)

        backend.register_standalone(Op(output_var_name, self.standalone_run, input_var_names, owner=self))
        self.module.to(backend.device0)



    @property
    def parameters(self):

        return self.module.parameters()  # .state_dict()

    @property
    def state_dict(self):

        return self.module.state_dict()

    def load_state_dict(self, state):

        return self.module.load_state_dict(state)

    def train(self, mode=True):
        self.module.train(mode)
