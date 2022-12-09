# -*- coding: utf-8 -*-
"""
Created on 2020/8/5
@project: SPAIC
@filename: Neuron
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
定义神经集群和神经元模型。
神经元集群保存神经元数目、神经元编号、类型、模型、模型参数、神经元位置等信息，参与网络构建
"""
# print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
import numpy as np
from ..Network import Assembly
from abc import ABC, abstractmethod
from collections import OrderedDict
from ..Network.BaseModule import VariableAgent, Op
import re
import torch

# from brian2 import *


class NeuronGroup(Assembly):
    '''Class for a group of neurons.
    '''

    _class_label = '<neg>'
    _is_terminal = True
    def __init__(self, num=None,
                 model=None,
                 shape=None,
                 neuron_type=('excitatory', 'inhibitory', 'pyramidal', '...'),
                 neuron_position=('x, y, z' or 'x, y'),
                 name=None,
                 **kwargs
                 ):
        super(NeuronGroup, self).__init__(name=name)
        self.set_num_shape(num=num, shape=shape)
        self.outlayer = kwargs.get("outlayer", False)

        # self.model = model
        if neuron_type == ('excitatory', 'inhibitory', 'pyramidal', '...'):
            self.type = ['nontype']
        elif isinstance(neuron_type, list):
            self.type = neuron_type
        else:
            self.type = [neuron_type]

        if neuron_position == ('x, y, z' or 'x, y'):
            self.position = []
        else:
            neuron_position = np.array(neuron_position)
            assert neuron_position.shape[0] == num, " Neuron_position not equal to neuron number"
            self.position = neuron_position

        self.parameters = kwargs
        if isinstance(model, str):
            self.model_class = NeuronModel.apply_model(model)
            self.model_name = model  # self.model -> self.model_name
            self.model = None
        elif isinstance(model, NeuronModel):
            self.model = model
            self.model_class = None
            self.model_name = 'custom_model'
        else:
            raise ValueError("only support set neuron model with string or NeuronModel class constructed by @custom_model()")
        self._var_names = list()
        self._var_dict = dict()
        self._operations = OrderedDict()
        self._init_operations = OrderedDict()


    def set_num_shape(self, num, shape):
        self.num = num
        self.shape = shape
        if self.shape is not None:
            num = np.prod(self.shape)
            if self.num is None:
                self.num = num
            else:
                assert self.num == num, "the neuron number is not accord with neuron shape"
        elif self.num is not None:
            self.shape = [self.num]
        else:
            raise ValueError("neither neuron number nor neuron shape is defined")

    def set_parameter(self):
        pass

    def get_model(self):
        return self.model

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

    def add_neuron_label(self, key: str):
        if isinstance(key, str) or isinstance(key, VariableAgent):
            return self._add_label(key)
        elif isinstance(key, list) or isinstance(key, tuple):
            keys = []
            for k in key:
                keys.append(self._add_label(k))
            return keys

    def build(self, backend):
        '''
        Parameters
        ----------
        backend : Backend.Backend
        Returns
        -------
        '''
        self._backend = backend
        batch_size = self._backend.get_batch_size()
        # if(self.parameters is not None):
        #     self.model = self.model_class(**self.model_parameters)
        # else:
        #     self.model = self.model_class()
        if self.model_class is not None:
            if 'dt' not in self.parameters:
                self.parameters['dt'] = self._backend.dt
            self.model = self.model_class(**self.parameters)


        dt = backend.dt
        for (key, tau_var) in self.model._tau_variables.items():
            key = self.add_neuron_label(key)
            tau_var = np.exp(-dt / tau_var)
            # shape = ()
            if tau_var.size > 1:
                shape = tau_var.shape
                assert tau_var.size == self.num, "The number of tau should equal the number of neurons."
            else:
                shape = ()

            self.variable_to_backend(key, shape, value=tau_var)

        for (key, membrane_tau_var) in self.model._membrane_variables.items():
            key = self.add_neuron_label(key)
            # TODO: 计划把membrane tau反过来了变成membrane_tau/dt，需要把用的模型也改一下
            membrane_tau_var = dt/membrane_tau_var
            shape = (1, *self.shape)  # (1, neuron_num)
            self.variable_to_backend(key, shape, value=membrane_tau_var)


        for (key, var) in self.model._variables.items():
            # add the rule to extend new dimension before shape (for slif model)
            extend_tag = re.search("\[\d*\]", key)
            if extend_tag is not None:
                extend_tag = int(key[extend_tag.start() + 1:extend_tag.end() - 1])
            key = self.add_neuron_label(key)

            if extend_tag is not None:
                shape = (1, extend_tag, *self.shape)
            else:
                shape = (1, *self.shape)  # (batch_size, shape)
            self.variable_to_backend(key, shape, value=var)

        for (key, var) in self.model._parameter_variables.items():
            key = self.add_neuron_label(key)

            if isinstance(var, np.ndarray):
                if var.size > 1:
                    shape = var.shape
                    # assert var.size == self.num, "The number of tau should equal the number of neurons."
                else:
                    shape = ()
            elif isinstance(var, torch.Tensor):
                shape = var.shape
            elif hasattr(var, '__iter__'):
                var = np.array(var)
                if var.size > 1:
                    shape = var.shape
                    assert var.size == self.num, "The number of tau should equal the number of neurons."
                else:
                    shape = ()
            else:
                shape = ()

            self.variable_to_backend(key, shape, value=var)

        for (key, var) in self.model._constant_variables.items():
            key = self.add_neuron_label(key)

            # if isinstance(var, np.ndarray):
            #     if var.size > 1:
            #         shape = var.shape
            #     else:
            #         shape = ()
            # elif isinstance(var, torch.Tensor):
            #     shape = var.shape
            # elif hasattr(var, '__iter__'):
            #     var = np.array(var)
            #     if var.size > 1:
            #         shape = var.shape
            #     else:
            #         shape = ()
            # else:
            shape = None

            self.variable_to_backend(key, shape, value=var, is_constant=True)

        op_count = 0
        for op in self.model._operations:
            addcode_op = Op(owner=self)
            if isinstance(op[1], str):
                # op_name = str(op_count) + ':' + op[1]
                op_name = f'{op_count}:{op[1]}'
                addcode_op.func = op[1]
                addcode_op.output = self.add_neuron_label(op[0])
                if len(op) > 3:  # 为了解决历史的单一list格式的问题
                    addcode_op.input = self.add_neuron_label(op[2:])
                else:
                    addcode_op.input = self.add_neuron_label(op[2])
                backend.add_operation(addcode_op)
            else:
                op_name = f'{op_count}:custom_function'
                addcode_op.func = op[1]
                addcode_op.output = self.add_neuron_label(op[0])
                if len(op) > 3:  # 为了解决历史的单一list格式的问题
                    addcode_op.input = self.add_neuron_label(op[2:])
                else:
                    addcode_op.input = self.add_neuron_label(op[2])
                backend.register_standalone(addcode_op)
            op_count += 1

            self._operations[op_name] = addcode_op

        for op in self.model._init_operations:
            addcode_op = []
            for ind, value in enumerate(op):
                if ind != 1:
                    addcode_op.append(self.add_neuron_label(value))
                else:
                    addcode_op.append(value)
            if len(op) > 3:
                addcode_op[2] = addcode_op[2:]
                addcode_op = addcode_op[:3]
            op_count += 1
            self.init_op_to_backend(addcode_op[0], addcode_op[1], addcode_op[2])
            op_name = str(op_count) + ':' + 'custom_initial_function'
            self._init_operations[op_name] = addcode_op



        if self.model_name == "slif" or self.model_name == 'selif':
            self.model.build((1, *self.shape), backend)
            self.model.outlayer = self.outlayer
            update_code = self.model.update_op_code
            intital_code = self.model.initial_op_code
            self.init_op_to_backend(intital_code[0], intital_code[1], intital_code[2])
            backend.register_standalone(Op(self.add_neuron_label(update_code[0]), update_code[1],
                                          [self.add_neuron_label(update_code[2])], owner=self))
            backend.register_standalone(Op(self.add_neuron_label('V'), self.model.return_V, [], owner=self))
            backend.register_standalone(Op(self.add_neuron_label('S'), self.model.return_S, [], owner=self))


    @staticmethod
    def custom_model(input_vars, output_vars, new_vars_dict, equation_type=('iterative','euler_iterative','exp_euler_iterative','ode'), backend='torch', custom_function_name='custom', base_model=None, add_threshold=True):
        '''
        Examples:
            @NeuronGroup.custom_model(input_vars=['M', 'S', 'WgtSum'], output_vars=['V', 'M', 'S'],
            new_vars_dict={'V':0, 'M':0, 'S':0, 'WgtSum':0}, equation_type='exp_euler_iterative')
            def func(M, S, WgtSum):
                M = (WgtSum-M)/tau
                S = (WgtSum-S)/tau
                V = M - S
                return V, M, S

            NeuronGroup(...., model=func)
        '''
        assert backend == 'torch'
        if base_model is None:
            model = NeuronModel()
        elif isinstance(base_model, NeuronModel):
            model = base_model
        else:
            raise ValueError("base model is given wrong type")
        model.name = custom_function_name
        if equation_type == 'iterative' or equation_type == 'ode':
            for key, value in new_vars_dict.items():
                if '[constant]' in key:
                    model._constant_variables[key.replace('[constant]','')] = value
                else:
                    model._variables[key] = value
        elif equation_type == 'euler_iterative':
            for key, value in new_vars_dict.items():
                if '[constant]' in key:
                    model._constant_variables[key.replace('[constant]','')] = value
                elif 'tau' in key.lower():
                    model._membrane_variables[key] = value
                else:
                    model._variables[key] = value
        elif equation_type == 'exp_euler_iterative':
            for key, value in new_vars_dict.items():
                if '[constant]' in key:
                    model._constant_variables[key.replace('[constant]','')] = value
                elif 'tau' in key.lower():
                    model._tau_variables[key] = value
                else:
                    model._variables[key] = value
        new_vars_dict = dict()
        new_vars_dict.update(model._variables)
        new_vars_dict.update(model._tau_variables)
        new_vars_dict.update(model._membrane_variables)
        new_vars_dict.update(model._constant_variables)
        for var in input_vars:
            if isinstance(var, VariableAgent):
                continue
            elif var not in new_vars_dict:
                if '[updated]' in var:
                    if var.replace('[updated]', '') in new_vars_dict:
                        continue
                else:
                    raise ValueError("The variable %s is not in model variable dict and not a Variable of other modules"%var)
        for var in output_vars:
            if isinstance(var, VariableAgent):
                continue
            elif var not in new_vars_dict:
                if '[updated]' in var:
                    if var.replace('[updated]', '') in new_vars_dict:
                        continue
                else:
                    raise ValueError("The variable %s is not in model variable dict and not a Variable of other modules"%var)

        def model_function(func):
            op_code = [output_vars, func,input_vars ]
            model._operations.append(op_code)
            if add_threshold == True:
                model._operations.append(('O', 'threshold', 'V[updated]', 'Vth'))
            return model
        return model_function


class NeuronModel(ABC):
    '''
    op -> (return_name, operation_name, input_name1, input_name2...)
    '''

    #: A dictionary mapping neuron model names to `Model` objects
    neuron_models = dict()

    def __init__(self, **kwargs):
        super(NeuronModel, self).__init__()
        self.name = 'none'
        self._operations = []
        self._init_operations = []
        self._variables = dict()
        # self._tau_constant_variables = dict()
        self._tau_variables = dict()
        self._parameter_variables = dict()
        self._membrane_variables = dict()
        self._constant_variables = dict()
        # self.neuron_parameters = dict()

    @staticmethod
    def register(name, model):
        '''
        Register a neuron model. Registered neuron models can be referred to
        # via their name.
        Parameters
        ----------
        name : str
            A short name for the state updater (e.g. `'lif'`)
        model : `NeuronModel`
            The neuron model object, e.g. an `CLIFModel`, 'SLIFModel'.
        '''

        # only deal with lower case names -- we don't want to have 'LIF' and
        # 'lif', for example
        name = name.lower()
        if name in NeuronModel.neuron_models:
            raise ValueError(('A model with the name "%s" has already been registered') % name)

        if not issubclass(model, NeuronModel):
            raise ValueError(('Given model of type %s does not seem to be a valid NeuronModel.' % str(type(model))))

        NeuronModel.neuron_models[name] = model
        model.name = name

    @staticmethod
    def apply_model(model_name):
        '''
        Parameters
        ----------
        model_name : str
        Returns
        -------
        '''
        model_name = model_name.lower()
        if model_name not in NeuronModel.neuron_models:
            raise ValueError(('Given model name is not in the model list'))
        else:
            return NeuronModel.neuron_models[model_name]

    # @abstractmethod
    # def get_var(self):
    #     NotImplementedError()

    # @abstractmethod
    # def get_op(self):
    #     NotImplementedError()

    # @abstractmethod
    # def get_tau(self):
    #     NotImplementedError()


class CLIFModel(NeuronModel):
    """
    Current LIF 3-kernel model:
    V(t) = M(t) − S(t) − E(t)
    I^n[t] = V0 * Isyn^n[t-1] #sum(w * O^(n-1)[t])
    M^n[t] = betaM * M^n[t-1] + I^n[t-1]
    S^n[t] = betaS * S^n[t-1] + I^n[t-1]
    E^n[t] = betaM * E^n[t-1] + Vth * O^n[t-1]
    O^n[t] = spike_func(V^n[t-1])
    """

    def __init__(self, **kwargs):
        super(CLIFModel, self).__init__()
        # self.neuron_parameters['tau_p'] = kwargs.get('tau_p', 12.0)
        # self.neuron_parameters['tau_q'] = kwargs.get('tau_q', 8.0)
        # self.neuron_parameters['tau_m'] = kwargs.get('tau_m', 20.0)
        # self.neuron_parameters['v_th'] = kwargs.get('v_th', 1.0)

        self._variables['M'] = 0.0
        self._variables['S'] = 0.0
        self._variables['E'] = 0.0
        # self._variables['I_che'] = 0.0
        # self._variables['I_ele'] = 0.0
        self._variables['I'] = 0.0
        self._variables['O'] = 0.0
        self._variables['V'] = 0.0
        self._variables['Isyn'] = 0.0

        self._tau_variables['tauM'] = np.asarray(kwargs.get('tau_m', 20.0))  # self.neuron_parameters['tau_m']
        self._tau_variables['tauP'] = np.asarray(kwargs.get('tau_p', 12.0))  # self.neuron_parameters['tau_p']
        self._tau_variables['tauQ'] = np.asarray(kwargs.get('tau_q', 8.0))  # self.neuron_parameters['tau_q']

        beta = self._tau_variables['tauP'] / self._tau_variables['tauQ']
        V0 = (1 / (beta - 1)) * (beta ** (beta / (beta - 1)))
        self._parameter_variables['V0'] = V0

        self._constant_variables['Vth'] = kwargs.get('v_th', 1.0)

        # self._operations.append(('I_che', 'var_mult', 'V0', 'WgtSum[updated]'))
        # self._operations.append(('I', 'add', 'I_che[updated]', 'I_ele'))
        self._operations.append(('I', 'var_mult', 'V0', 'Isyn[updated]'))
        self._operations.append(('M', 'var_linear', 'tauP', 'M', 'I[updated]'))
        self._operations.append(('S', 'var_linear', 'tauQ', 'S', 'I[updated]'))
        self._operations.append(('PSP', 'minus', 'M[updated]', 'S[updated]'))
        self._operations.append(('V', 'minus', 'PSP', 'E'))
        self._operations.append(('O', 'threshold', 'V[updated]', 'Vth'))
        self._operations.append(('Resetting', 'var_mult', 'Vth', 'O[updated]'))
        self._operations.append(('E', 'var_linear', 'tauM', 'E', 'Resetting'))


NeuronModel.register("clif", CLIFModel)


class AdaptiveCLIFModel(NeuronModel):
    """
    Current LIF 3-kernel model:
    V(t) = M(t) − S(t) − E(t)
    I^n[t] = V0 * Isyn^n[t-1] #sum(w * O^(n-1)[t])
    M^n[t] = betaM * M^n[t-1] + I^n[t-1]
    S^n[t] = betaS * S^n[t-1] + I^n[t-1]
    E^n[t] = betaM * E^n[t-1] + Vth * O^n[t-1]
    O^n[t] = spike_func(V^n[t-1])
    """

    def __init__(self, **kwargs):
        super(AdaptiveCLIFModel, self).__init__()
        # self.neuron_parameters['tau_p'] = kwargs.get('tau_p', 12.0)
        # self.neuron_parameters['tau_q'] = kwargs.get('tau_q', 8.0)
        # self.neuron_parameters['tau_m'] = kwargs.get('tau_m', 20.0)
        # self.neuron_parameters['v_th'] = kwargs.get('v_th', 1.0)

        self._variables['M'] = 0.0
        self._variables['S'] = 0.0
        self._variables['E'] = 0.0
        # self._variables['I_che'] = 0.0
        # self._variables['I_ele'] = 0.0
        self._variables['I'] = 0.0
        self._variables['O'] = 0.0
        self._variables['V'] = 0.0
        self._variables['Isyn'] = 0.0

        self._tau_variables['tauM'] = np.asarray(kwargs.get('tau_m', 20.0))  # self.neuron_parameters['tau_m']
        self._tau_variables['tauP'] = np.asarray(kwargs.get('tau_p', 12.0))  # self.neuron_parameters['tau_p']
        self._tau_variables['tauQ'] = np.asarray(kwargs.get('tau_q', 8.0))  # self.neuron_parameters['tau_q']

        beta = self._tau_variables['tauP'] / self._tau_variables['tauQ']
        V0 = (1 / (beta - 1)) * (beta ** (beta / (beta - 1)))
        self._parameter_variables['V0'] = V0

        self._constant_variables['Vth0'] = kwargs.get('v_th', 1.0)
        self._variables['Vth1'] = 0.0

        # self._operations.append(('I_che', 'var_mult', 'V0', 'WgtSum[updated]'))
        # self._operations.append(('I', 'add', 'I_che[updated]', 'I_ele'))

        self._operations.append(('Vth1', 'var_linear', 'tauP', 'Vth1', 'O'))
        self._operations.append(('Vth', 'add', 'Vth1[updated]', 'Vth0'))
        self._operations.append(('I', 'var_mult', 'V0', 'Isyn[updated]'))
        self._operations.append(('M', 'var_linear', 'tauP', 'M', 'I[updated]'))
        self._operations.append(('S', 'var_linear', 'tauQ', 'S', 'I[updated]'))
        self._operations.append(('PSP', 'minus', 'M[updated]', 'S[updated]'))
        self._operations.append(('V', 'minus', 'PSP', 'E'))
        self._operations.append(('O', 'threshold', 'V[updated]', 'Vth'))
        self._operations.append(('Resetting', 'var_mult', 'Vth', 'O[updated]'))
        self._operations.append(('E', 'var_linear', 'tauM', 'E', 'Resetting'))


NeuronModel.register("aclif", AdaptiveCLIFModel)


class IFModel(NeuronModel):
    """
    IF model:
    V(t) = V(t-1) * (1 - O(t-1)) + Isyn[t] - ConstantDecay

    O^n[t] = spike_func(V^n[t-1])
    """

    def __init__(self, **kwargs):
        super(IFModel, self).__init__()
        # self.neuron_parameters['ConstantDecay'] = kwargs.get('ConstantDecay', 0.0)
        # self.neuron_parameters['v_th'] = kwargs.get('v_th', 1.0)

        self._variables['O'] = 0.0
        self._variables['V'] = 0.0
        self._variables['Isyn'] = 0.0

        self._parameter_variables['ConstantDecay'] = kwargs.get('ConstantDecay', 0.0)
        self._parameter_variables['Vth'] = kwargs.get('v_th', 1.0)

        self._operations.append(('Vtemp', 'add', 'V', 'Isyn[updated]'))
        self._operations.append(('Vtemp1', 'minus', 'Vtemp', 'ConstantDecay'))
        self._operations.append(('O', 'threshold', 'Vtemp1', 'Vth'))
        self._operations.append(('Resetting', 'var_mult', 'Vtemp1', 'O[updated]'))
        self._operations.append(('V', 'minus', 'Vtemp1', 'Resetting'))


NeuronModel.register("if", IFModel)


class NullModel(NeuronModel):
    """
    return 'O'
    """

    def __init__(self, **kwargs):
        super(NullModel, self).__init__()
        # self.neuron_parameters['ConstantDecay'] = kwargs.get('ConstantDecay', 0.0)
        # self.neuron_parameters['v_th'] = kwargs.get('v_th', 1.0)
        self._variables['Isyn'] = 0.0
        self._variables['O'] = 0.0

        self._operations.append(('O', 'assign', 'Isyn[updated]'))


NeuronModel.register("null", NullModel)


class LIFModel(NeuronModel):
    """
    LIF model:
    # V(t) = tuaM * V^n[t-1] + Isyn[t]   # tauM: constant membrane time (tauM=RmCm)
    O^n[t] = spike_func(V^n[t-1])
    """

    def __init__(self, **kwargs):
        super(LIFModel, self).__init__()
        # initial value for state variables
        self._variables['V'] = 0.0
        self._variables['O'] = 0.0
        self._variables['Isyn'] = 0.0


        self._parameter_variables['Vth'] = kwargs.get('v_th', 1)
        self._constant_variables['Vreset'] = kwargs.get('v_reset', 0.0)

        self._tau_variables['tauM'] = kwargs.get('tau_m', 8.0)

        self._operations.append(('Vtemp', 'var_linear', 'tauM', 'V', 'Isyn[updated]'))
        self._operations.append(('O', 'threshold', 'Vtemp', 'Vth'))
        self._operations.append(('V', 'reset', 'Vtemp',  'O[updated]'))


NeuronModel.register("lif", LIFModel)


class ConstantCurrentLIFModel(NeuronModel):
    """
    ConstantCurrentLIF model:
    V(t) = V^n[t-1] + (dt/taum) * (Ureset-V^n[t-1]+I)  # tauM: constant membrane time (tauM=RmCm)  Isyn = I*Weight
    O^n[t] = spike_func(V^n[t-1])
    """

    def __init__(self, **kwargs):
        super(ConstantCurrentLIFModel, self).__init__()
        # initial value for state variables
        # self.neuron_parameters['tau_m'] = kwargs.get('tau_m', 20.0)
        # self.neuron_parameters['v_th'] = kwargs.get('v_th', 1.0)
        # self.neuron_parameters['v_reset'] = kwargs.get('v_reset', 0.0)

        self._variables['V'] = 0.0
        self._variables['O'] = 0.0
        self._variables['I'] = 0.0
        self._variables['Isyn'] = 0.0
        # beta = self.neuron_parameters['tau_p'] / self.neuron_parameters['tau_q']
        # V0 = 1.0  # (1 / (beta - 1)) * (beta ** (beta / (beta - 1)))
        # self._constant_variables['V0'] = V0

        self._parameter_variables['Vth'] = kwargs.get('v_th', 1.0)
        self._constant_variables['Vreset'] = kwargs.get('v_reset', 0.0)

        self._membrane_variables['tauM'] = kwargs.get('tau_m', 20.0)

        # self._operations.append(('I', 'var_mult', 'V0', 'I_synapse'))
        self._operations.append(('decayV', 'minus', 'Isyn', 'V'))
        self._operations.append(('Vtemp', 'var_linear', 'tauM', 'decayV', 'V'))
        self._operations.append(('O', 'threshold', 'Vtemp', 'Vth'))
        self._operations.append(('Resetting', 'var_mult', 'Vtemp', 'O'))
        self._operations.append(('V', 'minus', 'Vtemp', 'Resetting'))


NeuronModel.register("constantcurrentlif", ConstantCurrentLIFModel)


class NonSpikingLIFModel(NeuronModel):
    """
    NonSpikingLIF model:
    # V(t) = -tuaM * V^n[t-1] + M^n[t] - S^n[t]   # tauM: constant membrane time (tauM=RmCm)
    V(t) = V^n[t-1] + (dt/taum) * (PSP-V^n[t-1])  # tauM: constant membrane time (tauM=RmCm)
    I^n[t] = V0 * Isyn^n[t-1]                # sum(w * O^(n-1)[t])
    M^n[t] = tauP * M^n[t-1] + I^n[t-1]        # tauP: decaying time constants of membrane integration
    S^n[t] = tauQ * S^n[t-1] + I^n[t-1]        # tauQ: decaying time constants of synaptic currents
    PSP = M - S
    """

    def __init__(self, **kwargs):
        super(NonSpikingLIFModel, self).__init__()
        # initial value for state variables
        # self.neuron_parameters['tau_p'] = kwargs.get('tau_p', 4.0)
        # self.neuron_parameters['tau_q'] = kwargs.get('tau_q', 1.0)
        # self.neuron_parameters['tau_m'] = kwargs.get('tau_m', 1.0)

        self._variables['I'] = 0.0
        self._variables['M'] = 0.0
        self._variables['S'] = 0.0
        self._variables['V'] = 0.0
        self._variables['O'] = 0.0
        self._variables['Isyn'] = 0.0

        self._tau_variables['tauP'] = np.asarray(kwargs.get('tau_p', 4.0))
        self._tau_variables['tauQ'] = np.asarray(kwargs.get('tau_q', 1.0))
        beta = self._tau_variables['tauP'] / self._tau_variables['tauQ']
        V0 = (1 / (beta - 1)) * (beta ** (beta / (beta - 1)))
        self._parameter_variables['V0'] = V0

        self._membrane_variables['tauM'] = kwargs.get('tau_m', 1.0)
        # self._tau_constant_variables['tauM'] = kwargs.get('tau_m', 1.0)



        self._operations.append(('I', 'var_mult', 'V0', 'Isyn'))
        self._operations.append(('M', 'var_linear', 'tauP', 'M', 'I'))
        self._operations.append(('S', 'var_linear', 'tauQ', 'S', 'I'))
        self._operations.append(('PSP', 'minus', 'M', 'S'))
        self._operations.append(('decayV', 'minus', 'PSP', 'V'))
        self._operations.append(('V', 'var_linear', 'tauM', 'decayV', 'V'))


NeuronModel.register("nonspikinglif", NonSpikingLIFModel)


class LIFMModel(NeuronModel):
    """
    LIF model:
    # I_che = tauP*I + Isyn^n[t-1] + b^n                         # sum(w * O^(n-1)[t])
    # I = I_che + I_ele
    # F = tauM * exp(-O^n[t-1] / tauM)
    # V(t) = V^n[t-1] * F + I
    # O^(n)[t] = spike_func(V^n(t))
    """

    def __init__(self, **kwargs):
        super(LIFMModel, self).__init__()
        # initial value for state variables
        # self.neuron_parameters['tau_p'] = kwargs.get('tau_p', 1.0)
        # self.neuron_parameters['tau_m'] = kwargs.get('tau_m', 10.0)
        # self.neuron_parameters['v_th']  = kwargs.get('v_th', 1.0)

        self._variables['V'] = 0.0
        self._variables['O'] = 0.0
        self._variables['Isyn'] = 0.0
        # self._variables['b'] = 0.0
        # self._variables['I_che'] = 0.0
        # self._variables['I_ele'] = 0.0
        self._variables['I'] = 0.0

        self._parameter_variables['Vth'] = kwargs.get('v_th', 1.0)
        # self._constant_variables['Vreset'] = v_reset

        self._tau_variables['tauM'] = kwargs.get('tau_m', 10.0)
        self._tau_variables['tauP'] = kwargs.get('tau_p', 1.0)

        self._operations.append(('I', 'var_linear', 'tauP', 'I', 'Isyn[updated]'))

        # self._operations.append(('I_che', 'var_linear', 'tauP', 'I_che', 'PSP'))
        # self._operations.append(('I', 'add', 'I_che[updated]', 'I_ele'))
        self._operations.append(('Vtemp', 'var_linear', 'V', 'tauM', 'I[updated]'))
        self._operations.append(('O', 'threshold', 'Vtemp', 'Vth'))
        self._operations.append(('Vreset', 'var_mult', 'Vtemp', 'O[updated]'))
        self._operations.append(('V', 'minus', 'Vtemp', 'Vreset'))


NeuronModel.register("lifm", LIFMModel)


class IZHModel(NeuronModel):
    """
    IZH model:

    .. math::
        V &= V + dt / tauM * (C1 * V * V + C2 * V + C3 - U + I) \\

        V &= V + dt / tauM * (V * (C1 * V + C2) + C3 - U + I)  \\

        U &= U + a. * (b. * V - U) \\

        O^n[t] &= spike\_func(V^n[t-1]) \\

        if V &> Vth, \\

        then V &= C,

        U &= U + d

    References:
        Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on neural networks, 14(6), 1569-1572.
    """

    def __init__(self, **kwargs):
        super(IZHModel, self).__init__()

        self._variables['I'] = 0.0
        self._variables['O'] = 0.0
        self._variables['V'] = -65.0
        self._variables['U'] = 1.0
        self._variables['Isyn'] = 0.0  # 1.8

        self._constant_variables['a'] = kwargs.get('a', 0.02)
        self._constant_variables['b'] = kwargs.get('b', 0.2)
        self._parameter_variables['Vth'] = np.asarray(kwargs.get('v_th', 30))# 30.0
        self._parameter_variables['Vreset'] = np.asarray(kwargs.get('Vreset', -65.0))
        self._parameter_variables['d'] = kwargs.get('d', 8.0)  # 8.0
        self._constant_variables['C1'] = 0.04
        self._constant_variables['C2'] = 5
        self._constant_variables['C3'] = 140

        self._membrane_variables['tauM'] = 1.0

        # V = V + dt / tauM * (C1 * V * V + C2 * V + C3 - U + I)
        # V = V + dt / tauM * (V* (C1 * V + C2) + C3 - U + I)
        # U = U + dt /tauM * a. * (b. * V - U)

        self._operations.append(('temp_V1', 'var_linear', 'C1', 'V', 'C2'))
        self._operations.append(('temp_V2', 'var_linear', 'temp_V1', 'V', 'C3'))
        self._operations.append(('temp_V3', 'minus', 'temp_V2', 'U'))
        self._operations.append(('temp_V4', 'add', 'temp_V3', 'Isyn[updated]'))
        self._operations.append(('temp_V', 'var_linear', 'tauM', 'temp_V4', 'V'))

        self._operations.append(('temp_U1', 'var_mult', 'b', 'V'))
        self._operations.append(('temp_U2', 'minus', 'temp_U1', 'U'))
        self._operations.append(('temp_U', 'var_linear', 'a', 'temp_U2', 'U'))

        self._operations.append(('O', 'threshold', 'temp_V', 'Vth'))

        # if V > Vth,
        # then V <- C, U <- U + d
        self._operations.append(('VResetting', 'minus', 'Vreset', 'temp_V'))
        self._operations.append(('V', 'var_linear', 'O[updated]', 'VResetting', 'temp_V'))
        self._operations.append(('U', 'var_linear', 'd', 'O[updated]', 'temp_U'))


NeuronModel.register("izh", IZHModel)


class aEIFModel(NeuronModel):
    """
    aEIF model:

    .. math::
        V &= V + dt / tauM * (EL - V + EXP - U + I^n[t]) \\

        U &= U + dt / tauW * (a * (V - EL) - U) \\

        EXP &= delta\_t * delta\_t2 * exp(dv\_th/delta\_t2) \\

        dv &= V - EL \\

        dv\_th &= V - Vth \\

        O^n[t] &= spike\_func(V^n[t-1]) \\

        If V &> 20: \\

        then V &= EL,

        U &= U + b

    References:
        Brette, R., & Gerstner, W. (2005). Adaptive exponential integrate-and-fire model as an
        effective description of neuronal activity. Journal of neurophysiology, 94(5), 3637-3642.
    """

    def __init__(self, **kwargs):
        super(aEIFModel, self).__init__()

        self._variables['I'] = 0.0

        self._variables['O'] = 0.0
        self._variables['V'] = -70.6
        # self._variables['Vt'] = -70.6
        self._variables['U'] = 0.0
        self._variables['Isyn'] = 0.0
        self._variables['EXP'] = 0.0

        self._constant_variables['EL'] = kwargs.get('EL', -70.6)
        self._constant_variables['a'] = kwargs.get('a', 4) #0.05
        self._constant_variables['b'] = kwargs.get('b', 0.0805)
        self._parameter_variables['Vth'] = kwargs.get('v_th', -50.4)
        self._parameter_variables['Vspk'] = kwargs.get('Vspk', 20)

        self._constant_variables['delta_t'] = kwargs.get('delta_t', 2.0)

        self._membrane_variables['C'] = kwargs.get('C', 281)
        self._membrane_variables['tauW'] = kwargs.get('tau_w', 144) #144

        self._constant_variables['gL'] = kwargs.get('gL', 30)

        # V = V + dt / tauM * (gL * (EL - V + EXP) - U + I ^ n[t])
        # U = U + dt / tauW * (a * (V - EL) - U)
        # EXP = delta_t * exp(du_th / delta_t2)
        # du = V - EL
        # du_th = V - Vth
        # I ^ n[t] = V0 * WgtSum ^ n[t - 1]
        # O ^ n[t] = spike_func(V ^ n[t - 1])

        self._operations.append(('dv', 'minus', 'V', 'EL'))
        self._operations.append(('dv_th', 'minus', 'V', 'Vth'))

        self._operations.append(('EXP_T1', 'div', 'dv_th', 'delta_t'))
        self._operations.append(('EXP_T2', 'exp', 'EXP_T1'))
        self._operations.append(('EXP', 'var_mult', 'delta_t', 'EXP_T2'))

        self._operations.append(('temp_V0', 'minus', 'EXP[updated]', 'dv'))
        self._operations.append(('temp_V1', 'var_mult', 'gL', 'temp_V0'))
        self._operations.append(('temp_V2', 'minus', 'temp_V1', 'U'))
        self._operations.append(('temp_V3', 'add', 'temp_V2', 'Isyn[updated]'))
        self._operations.append(('Vt', 'var_linear', 'C', 'temp_V3', 'V'))

        self._operations.append(('temp_U1', 'var_mult', 'a', 'dv'))
        self._operations.append(('temp_U2', 'minus', 'temp_U1', 'U'))
        self._operations.append(('Ut', 'var_linear', 'tauW', 'temp_U2', 'U'))

        self._operations.append(('O', 'threshold', 'Vt', 'Vspk'))

        self._operations.append(('Vtemp2', 'var_mult', 'Vt', 'O[updated]'))
        self._operations.append(('Vtemp3', 'minus', 'Vt', 'Vtemp2'))
        self._operations.append(('V', 'var_linear', 'EL', 'O[updated]', 'Vtemp3'))
        self._operations.append(('U', 'var_linear', 'b', 'O[updated]', 'Ut'))


NeuronModel.register("aeif", aEIFModel)
NeuronModel.register("adex", aEIFModel)


class GLIFModel(NeuronModel):
    """
    Current GLIF5 model:

    .. math::

        V &= V + dt / C * (I + I1 + I2 - (V - E\_L) / R)  \\

        Theta\_s &= Theta\_s - dt * b\_s * Theta\_s \\

        I\_j &= I\_j - dt * k\_j * I\_j (j = 1,2) \\

        Theta\_v &= Theta\_v + dt * (a\_v * (V - E\_L) - b\_v * Theta\_v) \\

        v\_th &= Theta\_v + Theta\_s + Theta\_inf \\

        O &= spike\_func(V) \\

        Reset function:
        V &= E\_L + f\_v * (V - E\_L) - delta\_v \\

        Theta\_s &= Theta\_s + delta\_Theta\_s \\

        I\_j &= f\_j * I\_j + delta\_I\_j (j = 1, 2) \\

        Theta\_v &= Theta\_v

    References:
        Teeter, C., Iyer, R., Menon, V., Gouwens, N., Feng, D., Berg, J., ... & Mihalas, S. (2018). Generalized leaky
        integrate-and-fire models classify multiple neuron types. Nature communications, 9(1), 1-15.
    """

    def __init__(self, **kwargs):
        super(GLIFModel, self).__init__()

        self._variables['V']       = 0.0
        self._variables['Theta_s'] = 0.1
        self._variables['Theta_v'] = 0.2
        self._variables['I1']      = 0.08
        self._variables['I2']      = 0.06
        self._variables['I']       = 0.0
        self._variables['S']       = 0.0
        self._variables['O']       = 0.0
        self._variables['Vth']    = 1.0
        self._variables['Isyn'] = 0.0

        # self._constant_variables['deltaV'] = kwargs.get('E_L', 0.0) - kwargs.get('delta_v', 0.05)
        self._membrane_variables['C'] = kwargs.get('C', 60.0)
        self._membrane_variables['b_s'] = -1 / kwargs.get('b_s', 0.02)
        self._membrane_variables['k1'] = -1 / kwargs.get('k_1', 0.8)
        self._membrane_variables['k2'] = -1 / kwargs.get('k_2', 0.6)

        self._membrane_variables['a_v'] = 1 / kwargs.get('a_v', 0.12)
        self._membrane_variables['b_v'] = -1 / kwargs.get('b_v', 0.3)

        self._constant_variables['R'] = -1 / kwargs.get('R', 2)
        self._constant_variables['E_L'] = kwargs.get('E_L', 0.0)

        self._constant_variables['Theta_inf'] = kwargs.get('Theta_inf', 1.0)

        self._constant_variables['f_v'] = kwargs.get('f_v', 0.1)
        self._constant_variables['delta_v'] = kwargs.get('delta_v', 0.05)

        self._constant_variables['delta_Theta_s'] = kwargs.get('delta_Theta_s', 0.2)

        self._constant_variables['f1'] = kwargs.get('f1', 1.0)
        self._constant_variables['f2'] = kwargs.get('f2', 1.0)
        self._constant_variables['delta_I1'] = kwargs.get('delta_I1', 0.1)
        self._constant_variables['delta_I2'] = kwargs.get('delta_I2', 0.1)

        # I_j = I_j - dt * k_j * I_j
        self._operations.append(('I1temp', 'var_linear', 'k1', 'I1', 'I1'))
        self._operations.append(('I2temp', 'var_linear', 'k2', 'I2', 'I2'))

        # I_sum = I + I1 + I2
        self._operations.append(('I_sum', 'add', 'I1temp', 'I2temp'))
        self._operations.append(('I_sum', 'add', 'I_sum', 'Isyn[updated]'))

        # dv = V - E_L
        self._operations.append(('dv', 'minus', 'V', 'E_L'))

        # V_up = I + I1 + I2 - (V - E_L) / R
        self._operations.append(('V_up', 'var_linear', 'dv', 'R', 'I_sum'))

        # V = V + dt / C * (I + I1 + I2 - (V - E_L) / R)
        self._operations.append(('Vtemp', 'var_linear', 'C', 'V_up', 'V'))

        # Theta_s = Theta_s - dt * b_s * Theta_s
        self._operations.append(('Theta_s_temp', 'var_linear', 'b_s', 'Theta_s', 'Theta_s'))

        # Theta_v = Theta_v + dt * (a_v * (V - E_L) - b_v * Theta_v)
        # Theta_temp =Theta_v + dt * a_v * (V - E_L)
        # Theta_v = Theta_temp - dt * b_v * Theta_v

        self._operations.append(('Theta_temp', 'var_linear', 'a_v', 'dv', 'Theta_v'))
        self._operations.append(('Theta_v', 'var_linear', 'b_v', 'Theta_v', 'Theta_temp'))

        # v_th = Theta_v + Theta_s + Theta_inf
        self._operations.append(('Theta', 'add', 'Theta_v[updated]', 'Theta_s_temp'))
        self._operations.append(('Vth', 'add', 'Theta', 'Theta_inf'))

        self._operations.append(('O', 'threshold', 'Vtemp', 'Vth[updated]'))

        # V = E_L + f_v * (V - E_L) - delta_v
        self._operations.append(('dv_reset', 'minus', 'Vtemp', 'E_L'))
        self._operations.append(('Vreset', 'var_linear', 'f_v', 'dv_reset', 'E_L'))
        self._operations.append(('Vreset', 'minus', 'Vreset', 'delta_v'))
        self._operations.append(('Vreset', 'minus', 'Vreset', 'Vtemp'))

        self._operations.append(('V', 'var_linear', 'Vreset', 'O[updated]', 'Vtemp'))

        # Theta_s = Theta_s + delta_Theta_s
        self._operations.append(('Theta_s', 'var_linear', 'delta_Theta_s', 'O[updated]', 'Theta_s_temp'))

        # I_j = f_j * I_j + delta_I_j (j = 1, 2)
        self._operations.append(('I1_temp1', 'var_linear', 'f1', 'I1temp', 'delta_I1'))
        self._operations.append(('I1_temp2', 'minus', 'I1_temp1', 'I1temp'))
        self._operations.append(('I1', 'var_linear', 'I1_temp1', 'O[updated]', 'I1temp'))
        self._operations.append(('I2_temp1', 'var_linear', 'f2', 'I2temp', 'delta_I2'))
        self._operations.append(('I2_temp2', 'minus', 'I2_temp1', 'I2temp'))
        self._operations.append(('I2', 'var_linear', 'I2_temp1', 'O[updated]', 'I2temp'))


NeuronModel.register("glif", GLIFModel)


class HodgkinHuxleyModel(NeuronModel):
    """
    Hodgkin-Huxley model:

    V = V + dt/tau_v * (I - Ik)
    Ik = NA + K + L
    NA = g_NA * m^3 * h * (V - V_NA)
    K = g_K * n^4 * (V - V_K)
    L = g_L * (V - V_L)

    Na activation:
    m = m + dt/tau_m * (alpha_m * (1-m) - beta_m * m)

    K activation:
    n = n + dt/tau_n * (alpha_n * (1-n) - beta_n * n)

    Na inactivation:
    h = h + dt/tau_h * (alpha_h * (1-h) - beta_h * h)

    original function1:
    alpha_m = 0.1 * (-V + 25) / (exp((-V+25)/10) - 1)
    beta_m = 4 * exp(-V/18)
    alpha_n = 0.01 * (-V + 10) / (exp((-V+10)/10) - 1)
    beta_n = 0.125 * exp(-V/80)
    alpha_h = 0.07 * exp(-V/20)
    beta_h = 1/(exp((-V+30)/10) + 1)

    O^n[t] = spike_func(V^n[t-1])
    """

    def __init__(self, **kwargs):
        super(HodgkinHuxleyModel, self).__init__()

        self._variables['I'] = 0.0
        self._variables['O'] = 0.0
        self._variables['V'] = kwargs.get('V', 0.0)

        self._variables['m'] = kwargs.get('m', 0.5)
        self._variables['n'] = kwargs.get('n', 0.5)
        self._variables['h'] = kwargs.get('h', 0.06)

        self._variables['Isyn'] = 0.0

        self._constant_variables['Vth'] = kwargs.get('v_th', 1.0)
        # self._constant_variables['Vreset'] = 0.0

        self._membrane_variables['tauM'] = 1.0
        self._membrane_variables['tauN'] = 1.0
        self._membrane_variables['tauH'] = 1.0
        self._membrane_variables['tauV'] = 1.0

        self._constant_variables['1'] = 1.0
        # self._constant_variables['65'] = kwargs.get('V65', 0.0)
        self._constant_variables['Vreset'] = kwargs.get('Vreset', 0.0)
        self._constant_variables['dt'] = kwargs.get('dt', 0.1)

        self._constant_variables['g_NA'] = kwargs.get('g_NA', 120.0)
        self._constant_variables['g_K'] = kwargs.get('g_K', 36.0)
        self._constant_variables['g_L'] = kwargs.get('g_L', 0.3)

        self._constant_variables['E_NA'] = kwargs.get('E_NA', 120.0)
        self._constant_variables['E_K'] = kwargs.get('E_K', -12.0)
        self._constant_variables['E_L'] = kwargs.get('E_L', 10.6)

        self._constant_variables['a_m1'] = kwargs.get('alpha_m1', 0.1)
        self._constant_variables['a_m2'] = kwargs.get('alpha_m2', 25.0)
        self._constant_variables['a_m3'] = kwargs.get('alpha_m3', 10.0)
        self._constant_variables['b_m1'] = kwargs.get('beta_m1', 4.0)
        self._constant_variables['b_m2'] = kwargs.get('beta_m2', 0.0)
        self._constant_variables['b_m3'] = kwargs.get('beta_m3', 18.0)

        self._constant_variables['a_n1'] = kwargs.get('alpha_n1', 0.01)
        self._constant_variables['a_n2'] = kwargs.get('alpha_n2', 10.0)
        self._constant_variables['a_n3'] = kwargs.get('alpha_n3', 10.0)
        self._constant_variables['b_n1'] = kwargs.get('beta_n1', 0.125)
        self._constant_variables['b_n2'] = kwargs.get('beta_n2', 0.0)
        self._constant_variables['b_n3'] = kwargs.get('beta_n3', 80.0)

        self._constant_variables['a_h1'] = kwargs.get('alpha_h1', 0.07)
        self._constant_variables['a_h2'] = kwargs.get('alpha_h2', 0.0)
        self._constant_variables['a_h3'] = kwargs.get('alpha_h3', 20.0)
        self._constant_variables['b_h1'] = kwargs.get('beta_h1', 1.0)
        self._constant_variables['b_h2'] = kwargs.get('beta_h2', 30.0)
        self._constant_variables['b_h3'] = kwargs.get('beta_h3', 10.0)

        self._operations.append(('V65', 'add', 'V', 'Vreset'))

        # a_m = a_m1 * (-V + a_m2) / (exp((-V + a_m2) / a_m3) - 1)
        # b_m = b_m1 * exp((-V + b_m2) / b_m3)
        # alpha_m
        self._operations.append(('Vam', 'minus', 'a_m2', 'V65'))
        self._operations.append(('Vamd', 'div', 'Vam', 'a_m3'))
        self._operations.append(('expVamd1', 'exp', 'Vamd'))
        self._operations.append(('expVamd', 'minus', 'expVamd1', '1'))
        self._operations.append(('amtemp', 'div', 'Vam', 'expVamd'))
        self._operations.append(('a_m', 'var_mult', 'a_m1', 'amtemp'))

        # beta_m
        self._operations.append(('Vbm', 'minus', 'b_m2', 'V65'))
        self._operations.append(('Vbmd', 'div', 'Vbm', 'b_m3'))
        self._operations.append(('expVbmd', 'exp', 'Vbmd'))
        self._operations.append(('b_m', 'var_mult', 'b_m1', 'expVbmd'))

        # a_n = a_n1 * (-V + a_n2) / (exp((-V + a_n2) / a_n3) - 1)
        # b_n = b_n1 * exp((-V + b_n2) / b_n3)
        # alpha_n
        self._operations.append(('Van', 'minus', 'a_n2', 'V65'))
        self._operations.append(('Vand', 'div', 'Van', 'a_n3'))
        self._operations.append(('expVand1', 'exp', 'Vand'))
        self._operations.append(('expVand', 'minus', 'expVand1', '1'))
        self._operations.append(('antemp', 'div', 'Van', 'expVand'))
        self._operations.append(('a_n', 'var_mult', 'a_n1', 'antemp'))

        # beta_n
        self._operations.append(('Vbn', 'minus', 'b_n2', 'V65'))
        self._operations.append(('Vbnd', 'div', 'Vbn', 'b_n3'))
        self._operations.append(('expVbnd', 'exp', 'Vbnd'))
        self._operations.append(('b_n', 'var_mult', 'b_n1', 'expVbnd'))

        # a_h = a_h1 * exp((-V + a_h2) / a_h3)
        # b_h = b_h1 / (exp((-V + b_h2) / b_h3 + 1)
        # alpha_h
        self._operations.append(('Vah', 'minus', 'a_h2', 'V65'))
        self._operations.append(('Vahd', 'div', 'Vah', 'a_h3'))
        self._operations.append(('expVahd', 'exp', 'Vahd'))
        self._operations.append(('a_h', 'var_mult', 'a_h1', 'expVahd'))

        # beta_h
        self._operations.append(('Vbh', 'minus', 'b_h2', 'V65'))
        self._operations.append(('Vbhd', 'div', 'Vbh', 'b_h3'))
        self._operations.append(('expVbhd1', 'exp', 'Vbhd'))
        self._operations.append(('expVbhd', 'add', 'expVbhd1', '1'))
        self._operations.append(('b_h', 'div', 'b_h1', 'expVbhd'))

        # m = m + alpha_m * (1 - m) - beta_m * m
        # n = n + alpha_n * (1 - n) - beta_n * n
        # h = h + alpha_h * (1 - h) - beta_h * h

        # m
        self._operations.append(('mtemp1', 'minus', '1', 'm'))
        self._operations.append(('mtemp2', 'var_mult', 'a_m', 'mtemp1'))
        self._operations.append(('betam', 'var_mult', 'b_m', 'm'))
        self._operations.append(('mtemp3', 'minus', 'mtemp2', 'betam'))
        self._operations.append(('mtemp4', 'var_mult', 'tauM', 'mtemp3'))
        self._operations.append(('m', 'add', 'mtemp4', 'm'))
        # n
        self._operations.append(('ntemp1', 'minus', '1', 'n'))
        self._operations.append(('ntemp2', 'var_mult', 'a_n', 'ntemp1'))
        self._operations.append(('betan', 'var_mult', 'b_n', 'n'))
        self._operations.append(('ntemp3', 'minus', 'ntemp2', 'betan'))
        self._operations.append(('ntemp4', 'var_mult', 'tauN', 'ntemp3'))
        self._operations.append(('n', 'add', 'ntemp4', 'n'))
        # h
        self._operations.append(('htemp1', 'minus', '1', 'h'))
        self._operations.append(('htemp2', 'var_mult', 'a_h', 'htemp1'))
        self._operations.append(('betah', 'var_mult', 'b_h', 'h'))
        self._operations.append(('htemp3', 'minus', 'htemp2', 'betah'))
        self._operations.append(('htemp4', 'var_mult', 'tauH', 'htemp3'))
        self._operations.append(('h', 'add', 'htemp4', 'h'))

        # g_NAm3h
        self._operations.append(('m2', 'var_mult', 'm[updated]', 'm[updated]'))
        self._operations.append(('m3', 'var_mult', 'm2', 'm[updated]'))
        self._operations.append(('m3h', 'var_mult', 'm3', 'h[updated]'))
        self._operations.append(('g_NAm3h', 'var_mult', 'g_NA', 'm3h'))

        # g_Kn4
        self._operations.append(('n2', 'var_mult', 'n[updated]', 'n[updated]'))
        self._operations.append(('n4', 'var_mult', 'n2', 'n2'))
        self._operations.append(('g_Kn4', 'var_mult', 'g_K', 'n4'))

        self._operations.append(('d_NA', 'minus', 'V', 'E_NA'))
        self._operations.append(('d_K', 'minus', 'V', 'E_K'))
        self._operations.append(('d_L', 'minus', 'V', 'E_L'))

        # Ik, NA, K, L
        self._operations.append(('NA', 'var_mult', 'g_NAm3h', 'd_NA'))
        self._operations.append(('K', 'var_mult', 'g_Kn4', 'd_K'))
        self._operations.append(('L', 'var_mult', 'g_L', 'd_L'))

        self._operations.append(('Ik1', 'add', 'NA', 'K'))
        self._operations.append(('Ik2', 'add', 'Ik1', 'L'))
        self._operations.append(('Ik', 'var_mult', 'tauV', 'Ik2'))

        # I
        self._operations.append(('Vtemp', 'minus', 'Isyn[updated]', 'Ik'))
        self._operations.append(('V', 'add', 'V', 'Vtemp'))

        # O
        self._operations.append(('O', 'threshold', 'V[updated]', 'Vth'))


NeuronModel.register("hh", HodgkinHuxleyModel)


class LIFSTDPEXModel(NeuronModel):
    """
    LIF model:
    V(t) = decay_v * (v - v_rest) + v_rest + I^n[t]
    I^n[t] = V0 * Isyn^n[t]  #V0 = 1
    theta(t) = decay_th * theta[t-1]
    if v >= (vth + theta) then s_out = 1; else s_out = 0;
    Reset:
    V(t) = s_out * v_reset + (1 - s_out) * v; theta = theta + s_out * th_inc
    O^n[t] = spike_func(V^n[t-1])
    """
    def __init__(self, **kwargs):
        super(LIFSTDPEXModel, self).__init__()
        # initial value for state variables
        # self.neuron_parameters['decay_v'] = kwargs.get('decay_v', np.exp(-1/100))
        # self.neuron_parameters['decay_th'] = kwargs.get('decay_th', np.exp(-1/1e7))
        # self.neuron_parameters['th_inc'] = kwargs.get('th_inc', 0.05)
        # self.neuron_parameters['v_th'] = kwargs.get('v_th', -52.0)
        # self.neuron_parameters['v_rest'] = kwargs.get('v_rest', -65.0)
        # self.neuron_parameters['v_reset'] = kwargs.get('v_reset', -60.0)

        self._variables['I'] = 0.0
        self._variables['V'] = -65.0
        self._variables['O'] = 0.0
        self._variables['Isyn'] = 0.0
        self._variables['theta[stay]'] = 0.0
        self._variables['Vth_theta'] = 0.0

        # self._constant_variables['V0'] = 1

        self._parameter_variables['Vth'] = kwargs.get('v_th', -52.0)
        self._constant_variables['Vreset'] = kwargs.get('v_reset', -60.0)
        self._constant_variables['Vrest'] = kwargs.get('v_rest', -65.0)
        self._constant_variables['th_inc'] = kwargs.get('th_inc', 0.05)
        self._constant_variables['decay_th'] = kwargs.get('decay_th', np.exp(-1/1e7))
        self._constant_variables['decay_v'] = kwargs.get('decay_v', np.exp(-1/100))


        # self._operations.append(('I', 'var_mult', 'V0', 'I_synapse[updated]'))
        self._operations.append(('PSP1', 'minus', 'V', 'Vrest'))
        self._operations.append(('PSP2', 'var_linear', 'decay_v', 'PSP1', 'Vrest'))
        self._operations.append(('Vtemp', 'add', 'PSP2', 'Isyn[updated]'))
        self._operations.append(('theta_temp', 'var_mult', 'decay_th', 'theta[stay]'))
        self._operations.append(('Vth_theta', 'add', 'Vth', 'theta_temp'))
        self._operations.append(('O', 'threshold', 'Vtemp', 'Vth_theta'))
        self._operations.append(('Resetting1', 'var_mult', 'Vreset', 'O[updated]'))
        self._operations.append(('Resetting2', 'var_mult', 'Vtemp', 'O[updated]'))
        self._operations.append(('Resetting3', 'minus', 'Vtemp', 'Resetting2'))
        self._operations.append(('V', 'add', 'Resetting1', 'Resetting3'))
        self._operations.append(('Resetting_theta', 'var_mult', 'O[updated]', 'th_inc'))
        self._operations.append(('theta[stay]', 'add', 'theta_temp', 'Resetting_theta'))


NeuronModel.register("lifstdp_ex", LIFSTDPEXModel)


class ALIFSTDPEXModel(NeuronModel):
    """
    LIF model:
    V(t) = decay_v * (v - v_rest) + v_rest + I^n[t]
    I^n[t] = V0 * Isyn^n[t]  #V0 = 1
    theta(t) = decay_th * theta[t-1]
    if v >= (vth + theta) then s_out = 1; else s_out = 0;
    Reset:
    V(t) = s_out * v_reset + (1 - s_out) * v; theta = theta + s_out * th_inc

    O^n[t] = spike_func(V^n[t-1])
    """
    def __init__(self, **kwargs):
        super(ALIFSTDPEXModel, self).__init__()
        # initial value for state variables
        # self.neuron_parameters['decay_v'] = kwargs.get('decay_v', np.exp(-1/100))
        # self.neuron_parameters['decay_th'] = kwargs.get('decay_th', np.exp(-1/1e7))
        # self.neuron_parameters['th_inc'] = kwargs.get('th_inc', 0.05)
        # self.neuron_parameters['v_th'] = kwargs.get('v_th', -52.0)
        # self.neuron_parameters['v_rest'] = kwargs.get('v_rest', -65.0)
        # self.neuron_parameters['v_reset'] = kwargs.get('v_reset', -60.0)

        self._variables['I'] = 0.0
        self._variables['V'] = 0.0
        self._variables['O'] = 0.0
        self._variables['Isyn'] = 0.0
        self._variables['theta[stay]'] = 0.0
        self._variables['Vth_theta'] = 2.0

        # self._constant_variables['V0'] = 1

        self._parameter_variables['Vth'] = kwargs.get('v_th', 2.0)
        self._constant_variables['Vreset'] = kwargs.get('v_reset', 2.0)
        self._constant_variables['Vrest'] = kwargs.get('v_rest', 0.0)
        self._constant_variables['th_inc'] = kwargs.get('th_inc', 0.001)
        self._constant_variables['decay_th'] = kwargs.get('decay_th', np.exp(-1/100000.0))
        self._constant_variables['decay_v'] = kwargs.get('decay_v', np.exp(-1/100.0))


        # self._operations.append(('I', 'var_mult', 'V0', 'I_synapse[updated]'))
        self._operations.append(('PSP1', 'minus', 'V', 'Vrest'))
        self._operations.append(('PSP2', 'var_linear', 'decay_v', 'PSP1', 'Vrest'))
        self._operations.append(('Vtemp', 'add', 'PSP2', 'Isyn[updated]'))
        self._operations.append(('theta_temp', 'var_mult', 'decay_th', 'theta[stay]'))
        self._operations.append(('Vth_theta', 'add', 'Vth', 'theta_temp'))
        self._operations.append(('O', 'threshold', 'Vtemp', 'Vth_theta'))
        self._operations.append(('Resetting1', 'var_mult', 'Vreset', 'O[updated]'))
        self._operations.append(('V', 'minus', 'Vtemp', 'Resetting1'))
        self._operations.append(('Resetting_theta1', 'var_mult', 'O[updated]', 'th_inc'))
        self._operations.append(('Resetting_theta', 'var_mult', 'Resetting_theta1', 'Vth_theta'))
        self._operations.append(('theta[stay]', 'add', 'theta_temp', 'Resetting_theta'))


NeuronModel.register("alifstdp_ex", ALIFSTDPEXModel)


class LIFSTDPIHModel(NeuronModel):
    """
    LIF model:
    V(t) = decay_v * (v - v_rest) + v_rest + I^n[t]
    I^n[t] = V0 * Isyn^n[t]  #V0 = 1

    Reset:
    V(t) = s_out * v_reset + (1 - s_out) * v;

    O^n[t] = spike_func(V^n[t-1])
    """
    def __init__(self, **kwargs):
        super(LIFSTDPIHModel, self).__init__()
        # initial value for state variables
        # self.neuron_parameters['decay_v'] = kwargs.get('decay_v', np.exp(-1/10))
        # self.neuron_parameters['v_th'] = kwargs.get('v_th', -40.0)
        # self.neuron_parameters['v_rest'] = kwargs.get('v_rest', -60.0)
        # self.neuron_parameters['v_reset'] = kwargs.get('v_reset', -45.0)

        self._variables['I'] = 0.0
        self._variables['V'] = -60.0
        self._variables['O'] = 0.0
        self._variables['Isyn'] = 0.0

        # self._constant_variables['V0'] = 1

        self._parameter_variables['Vth'] = kwargs.get('v_th', -40.0)
        self._constant_variables['Vreset'] = kwargs.get('v_reset', -45.0)
        self._constant_variables['Vrest'] = kwargs.get('v_rest', -60.0)
        self._constant_variables['decay_v'] = kwargs.get('decay_v', np.exp(-1/10))


        # self._operations.append(('I', 'var_mult', 'V0', 'I_synapse[updated]'))
        self._operations.append(('PSP1', 'minus', 'V', 'Vrest'))
        self._operations.append(('PSP2', 'var_linear', 'decay_v', 'PSP1', 'Vrest'))
        self._operations.append(('Vtemp', 'add', 'PSP2', 'Isyn[updated]'))
        self._operations.append(('O', 'threshold', 'Vtemp', 'Vth'))
        self._operations.append(('Resetting1', 'var_mult', 'Vreset', 'O[updated]'))
        self._operations.append(('Resetting2', 'var_mult', 'Vtemp', 'O[updated]'))
        self._operations.append(('Resetting3', 'minus', 'Vtemp', 'Resetting2'))
        self._operations.append(('V', 'add', 'Resetting1', 'Resetting3'))


NeuronModel.register("lifstdp_ih", LIFSTDPIHModel)


class CANN_MeanFieldModel(NeuronModel):
    """
    Mean Field Model used in "Fung CC, Wong KY, Wu S. A moving bump in a continuous manifold: a comprehensive study of
    the tracking dynamics of continuous attractor neural networks. Neural Comput. 2010 Mar;22(3):752-92. doi: 10.1162/neco.2009.07-08-824. "
    "Wu S, Wong KY, Fung CC, Mi Y, Zhang W. Continuous Attractor Neural Networks: Candidate of a Canonical Model for
    Neural Information Representation.F1000Res. 2016 Feb 10;5:F1000 Faculty Rev-156. doi: 10.12688/f1000research.7387.1. "

    U = U + dt/tau * (Iext + rho*WgtSum - U)
    O = U^2/(1 + k*rho*sum(U^2))
    (WgtSum = weight*O_pre)
    """
    def __init__(self, **kwargs):
        super(CANN_MeanFieldModel, self).__init__()

        self._constant_variables['rho'] = kwargs.get('rho', 0.02)
        self._constant_variables['k_rho'] = kwargs.get('k', 0.01)
        self._constant_variables['1'] = 1
        self._constant_variables['2'] = 2

        self._membrane_variables['tau'] = kwargs.get('tau', 1.0)

        self._variables['Iext'] = 0.0
        self._variables['Isyn'] = 0.0
        self._variables['U'] = 0.0
        self._variables['O'] = 0.0

        self._operations.append(('Isum', 'var_linear', 'rho', 'Isyn', 'Iext'))
        self._operations.append(('dU', 'minus', 'Isum', 'U'))
        self._operations.append(('U', 'var_linear', 'tau', 'dU', 'U'))
        self._operations.append(('ReU', 'relu', 'U[updated]'))
        self._operations.append(('U2', 'var_mult', 'ReU', 'ReU'))
        self._operations.append(('SumU2', 'reduce_sum', 'U2', '1'))
        self._operations.append(('RBase','var_linear', 'k_rho', 'SumU2', '1'))
        self._operations.append(('O', 'div', 'U2', 'RBase'))


NeuronModel.register("cann_field", CANN_MeanFieldModel)


class MeanFieldModel(NeuronModel):
    """
    Mean Field Model of LIF neuron "

    U = U + dt/tau * (rho*(Iext + Isyn) - U)
    O = relu(U)
    (WgtSum = weight*O_pre)
    """
    def __init__(self, **kwargs):
        super(MeanFieldModel, self).__init__()

        self._constant_variables['rho'] = kwargs.get('rho', 0.1)
        self._constant_variables['1'] = 1
        self._constant_variables['2'] = 2
        self._membrane_variables['tau'] = kwargs.get('tau', 1.0)

        self._variables['Iext'] = 0.0
        self._variables['Isyn'] = 0.0
        self._variables['U'] = 0.0
        self._variables['O'] = 0.0

        self._operations.append(('Isum', 'var_linear', 'rho', 'Isyn', 'Iext'))
        self._operations.append(('dU', 'minus', 'Isum', 'U'))
        self._operations.append(('U', 'var_linear', 'tau', 'dU', 'U'))
        self._operations.append(('O', 'relu', 'U[updated]'))


NeuronModel.register("meanfield", MeanFieldModel)


class Darwin_CLIF(NeuronModel):
    """
    I = I*P4 + Wgt_sum
    V = V*P0 + I + P1
    """

    def __init__(self, **kwargs):
        super(Darwin_CLIF, self).__init__()
        self.dt = kwargs.get('dt', 0.1)
        self.tau_m = kwargs.get('tau_m', 12.0)
        self.tau_s = kwargs.get('tau_s', 8.0)
        self.bias = kwargs.get('bias', 0.0)
        self.v_th = kwargs.get('v_th', 16384)

        self._constant_variables['P0'] = np.round(np.exp(-self.dt / self.tau_m) * 2.0 ** 16) / 2.0 ** 16 #non-negative
        self._constant_variables['P4'] = np.round(np.exp(-self.dt / self.tau_s) * 2.0 ** 16) / 2.0 ** 16 #non-negative
        self._constant_variables['P1'] = np.round(self.bias*2.0**15)/2.0*15
        self._constant_variables['Vth'] = self.v_th

        self._variables['V'] = 0.0
        self._variables['I'] = 0.0
        self._variables['O'] = 0.0
        self._variables['Isyn'] = 0.0 # named WgtSum in Darwin

        self._operations.append((['V', 'I', 'O'], self.update, ['V', 'I', 'Isyn', 'P0', 'P4', 'P1', 'Vth']))

    def update(self, V, I, Isyn, P0, P4, P1, Vth):
        I = self.quantize_16(I*P4 + Isyn)
        V = self.quantize_16(V*P0 + I + P1)
        O = (V > Vth).to(torch.float)
        V = V - V*O
        return V, I, O

    def quantize_16(self, x):
        x = torch.round(torch.clamp(x+2**15, 0, 2**16)) - 2**15
        return x


NeuronModel.register("darwin_clif", Darwin_CLIF)








