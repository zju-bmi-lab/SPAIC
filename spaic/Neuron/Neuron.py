# -*- coding: utf-8 -*-
"""
Created on 2020/8/5
@project: SNNFlow
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
import re

# from brian2 import *

class NeuronGroup(Assembly):
    '''Class for a group of neurons.
    '''

    _class_label = '<neg>'
    _is_terminal = True
    def __init__(self, neuron_number=None,
                 neuron_shape=None,
                 neuron_type=('excitatory', 'inhibitory', 'pyramidal', '...'),
                 neuron_position=('x, y, z' or 'x, y'),
                 neuron_model=None,
                 name=None,
                 **kwargs
                 ):
        super(NeuronGroup, self).__init__(name=name)
        self.num = neuron_number
        # self.name = name
        self.shape = neuron_shape
        # self.neuron_model = neuron_model
        if neuron_type == ('excitatory', 'inhibitory', 'pyramidal', '...'):
            self.type = 'non_type'
        else:
            self.type = neuron_type

        if neuron_position == ('x, y, z' or 'x, y'):
            self.position = []
        else:
            neuron_position = np.array(neuron_position)
            assert neuron_position.size(0) == neuron_number, " Neuron_position not equal to neuron number"
            self.position = neuron_position

        self.parameters = kwargs

        self.model_class = NeuronModel.apply_model(neuron_model)
        self.model_name = neuron_model # self.neuron_model -> self.model_name
        self._var_names = list()
        self._operations = OrderedDict()



    def set_parameter(self):
        pass

    def get_model(self):
        return self.model

    def add_neuron_label(self, key: str):
        if '[updated]' in key:
            return self.id + ':' +'{'+key.replace('[updated]',"")+'}' + '[updated]'
        else:
            return self.id + ':' +'{'+key+'}'

    def build(self, simulator):
        '''
        Parameters
        ----------
        simulator : Backend.Backend
        Returns
        -------
        '''
        self._simulator = simulator
        batch_size = self._simulator.get_batch_size()
        # if(self.parameters is not None):
        #     self.model = self.model_class(**self.model_parameters)
        # else:
        #     self.model = self.model_class()
        self.model = self.model_class(**self.parameters)
        if self.shape is not None:
            num = np.prod(self.shape)
            if self.num is None:
                self.num = num
            else:
                assert self.num == num, "the neuron number is not accord with neuron shape"
        elif self.num is not None:
            self.shape = [self.num]
        else:
            ValueError("neither neuron number nor neuron shape is defined")

        dt = simulator.dt
        for (key, tau_var) in self.model._tau_constant_variables.items():
            key = self.add_neuron_label(key)
            tau_var = np.exp(-dt/tau_var)
            shape = ()
            simulator.add_variable(key, shape, value=tau_var)
            self._var_names.append(key)

        for (key, membrane_tau_var) in self.model._membrane_variables.items():
            key = self.add_neuron_label(key)
            membrane_tau_var = dt/membrane_tau_var
            shape = (1, *self.shape)  # (1, neuron_num)
            simulator.add_variable(key, shape, value=membrane_tau_var)
            self._var_names.append(key)

        for (key, var) in self.model._variables.items():
            # add the rule to extend new dimension before shape (for slif model)
            extend_tag = re.search("\[\d*\]", key)
            if extend_tag is not None:
                extend_tag = int(key[extend_tag.start()+1:extend_tag.end()-1])
            key = self.add_neuron_label(key)

            if extend_tag is not None:
                shape = (1, extend_tag, *self.shape)
            else:
                shape = (1, *self.shape)  # (batch_size, neuron_shape)
            simulator.add_variable(key, shape, value=var)
            self._var_names.append(key)

        for (key, var) in self.model._constant_variables.items():
            key = self.add_neuron_label(key)
            if isinstance(var, np.ndarray):
                if var.size > 1:
                    var_shape = var.shape
                    shape = (1, *var_shape)  # (1, neuron_shape)
                else:
                    shape = ()
            elif isinstance(var, list):
                if len(var) > 1:
                    var_len = len(var)
                    shape = (1, var_len)  # (1, neuron_shape)
                else:
                    shape = ()
            else:
                shape = ()
            simulator.add_variable(key, shape, value=var)
            self._var_names.append(key)

        op_count = 0
        for op in self.model._operations:
            addcode_op = []
            op_name = str(op_count) + ':' + op[1]
            op_count += 1
            for ind, name in enumerate(op):
                if ind != 1:
                    addcode_op.append(self.add_neuron_label(op[ind]))
                else:
                    addcode_op.append(op[ind])
            simulator.add_operation(addcode_op)
            self._operations[op_name] = addcode_op

        if self.model_name == "slif":
            self.model.build((1, *self.shape), simulator)
            update_code = self.model.update_op_code
            intital_code = self.model.initial_op_code
            simulator.register_initial(intital_code[0],intital_code[1],intital_code[2])
            simulator.register_standalone(self.add_neuron_label(update_code[0]), update_code[1], [self.add_neuron_label(update_code[2])])
            simulator.register_standalone(self.add_neuron_label('V'), self.model.return_V,  [])


class NeuronModel(ABC):
    '''
    op -> (return_name, operation_name, input_name1, input_name2...)
    '''

    #: A dictionary mapping neuron model names to `Model` objects
    neuron_models = dict()

    def __init__(self, **kwargs):
        super(NeuronModel, self).__init__()
        self._operations = []
        self._variables = dict()
        self._tau_constant_variables = dict()
        self._membrane_variables = dict()
        self._constant_variables = dict()
        self.neuron_parameters = dict()

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
            raise ValueError(('A neuron_model with the name "%s" has already been registered') % name)

        if not issubclass(model, NeuronModel):
            raise ValueError(('Given model of type %s does not seem to be a valid NeuronModel.' % str(type(model))))

        NeuronModel.neuron_models[name] = model

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
    I^n[t] = V0 * WgtSum^n[t-1] #sum(w * O^(n-1)[t])
    M^n[t] = betaM * M^n[t-1] + I^n[t-1]
    S^n[t] = betaS * S^n[t-1] + I^n[t-1]
    E^n[t] = betaM * E^n[t-1] + Vth * O^n[t-1]
    O^n[t] = spike_func(V^n[t-1])
    """

    def __init__(self, **kwargs):
        super(CLIFModel, self).__init__()
        self.neuron_parameters['tau_p'] = kwargs.get('tau_p', 12.0)
        self.neuron_parameters['tau_q'] = kwargs.get('tau_q', 8.0)
        self.neuron_parameters['tau_m'] = kwargs.get('tau_m', 20.0)
        self.neuron_parameters['v_th'] = kwargs.get('v_th', 1.0)

        self._variables['M'] = 0.0
        self._variables['S'] = 0.0
        self._variables['E'] = 0.0
        self._variables['I'] = 0.0
        self._variables['O'] = 0.0
        self._variables['V'] = 0.0
        self._variables['WgtSum'] = 0.0
        beta = self.neuron_parameters['tau_p'] / self.neuron_parameters['tau_q']
        V0 = (1 / (beta - 1)) * (beta ** (beta / (beta - 1)))
        self._constant_variables['V0'] = V0

        self._constant_variables['Vth'] = self.neuron_parameters['v_th']

        self._tau_constant_variables['tauM'] = self.neuron_parameters['tau_m']
        self._tau_constant_variables['tauP'] = self.neuron_parameters['tau_p']
        self._tau_constant_variables['tauQ'] = self.neuron_parameters['tau_q']

        self._operations.append(('I', 'var_mult', 'V0', 'WgtSum[updated]'))

        self._operations.append(('M', 'var_linear', 'tauP', 'M', 'I[updated]'))
        self._operations.append(('S', 'var_linear', 'tauQ', 'S', 'I[updated]'))

        self._operations.append(('PSP', 'minus', 'M[updated]', 'S[updated]'))
        self._operations.append(('V', 'minus', 'PSP', 'E'))
        self._operations.append(('O', 'threshold', 'V[updated]', 'Vth'))
        self._operations.append(('Resetting', 'var_mult', 'Vth', 'O[updated]'))
        self._operations.append(('E', 'var_linear', 'tauM', 'E', 'Resetting'))

NeuronModel.register("clif", CLIFModel)

class LIFModel(NeuronModel):
    """
    LIF model:
    # V(t) = -tuaM * V^n[t-1] + M^n[t] - S^n[t]   # tauM: constant membrane time (tauM=RmCm)
    V(t) = V^n[t-1] + (dt/taum) * (PSP-V^n[t-1])  # tauM: constant membrane time (tauM=RmCm)
    I^n[t] = V0 * WgtSum^n[t]                # sum(w * O^(n-1)[t])
    M^n[t] = tauP * M^n[t-1] + I^n[t-1]        # tauP: decaying time constants of membrane integration
    S^n[t] = tauQ * S^n[t-1] + I^n[t-1]        # tauQ: decaying time constants of synaptic currents
    PSP = M - S
    O^n[t] = spike_func(V^n[t-1])
    """

    def __init__(self, **kwargs):
        super(LIFModel, self).__init__()
        # initial value for state variables
        self.neuron_parameters['tau_p'] = kwargs.get('tau_p', 4.0)  # 12
        self.neuron_parameters['tau_q'] = kwargs.get('tau_q', 1.0)  # 8
        self.neuron_parameters['tau_m'] = kwargs.get('tau_m', 6.0)  # 20
        self.neuron_parameters['v_th'] = kwargs.get('v_th', 1)
        self.neuron_parameters['v_reset'] = kwargs.get('v_reset', 0.0)

        self._variables['I'] = 0.0
        self._variables['M'] = 0.0
        self._variables['S'] = 0.0
        self._variables['V'] = 0.0
        self._variables['O'] = 0.0
        self._variables['WgtSum'] = 0.0
        beta = self.neuron_parameters['tau_p'] / self.neuron_parameters['tau_q']
        V0 = (1 / (beta - 1)) * (beta ** (beta / (beta - 1)))
        self._constant_variables['V0'] = V0

        self._constant_variables['Vth'] = self.neuron_parameters['v_th']
        self._constant_variables['Vreset'] = self.neuron_parameters['v_reset']

        self._membrane_variables['tauM'] = self.neuron_parameters['tau_m']

        self._tau_constant_variables['tauP'] = self.neuron_parameters['tau_p']
        self._tau_constant_variables['tauQ'] = self.neuron_parameters['tau_q']

        self._operations.append(('I', 'var_mult', 'V0', 'WgtSum'))
        self._operations.append(('M', 'var_linear', 'tauP', 'M', 'I[updated]'))
        self._operations.append(('S', 'var_linear', 'tauQ', 'S', 'I[updated]'))
        self._operations.append(('PSP', 'minus', 'M[updated]', 'S[updated]'))
        self._operations.append(('decayV', 'minus', 'PSP', 'V'))
        self._operations.append(('Vtemp', 'var_linear', 'tauM', 'decayV', 'V'))
        self._operations.append(('O', 'threshold', 'Vtemp', 'Vth'))
        self._operations.append(('Resetting', 'var_mult', 'Vtemp', 'O[updated]'))
        self._operations.append(('V', 'minus', 'Vtemp', 'Resetting'))

NeuronModel.register("lif", LIFModel)

class ConstantCurrentLIFModel(NeuronModel):
    """
    ConstantCurrentLIF model:
    V(t) = V^n[t-1] + (dt/taum) * (Ureset-V^n[t-1]+I)  # tauM: constant membrane time (tauM=RmCm)  WgtSum = I*Weight
    O^n[t] = spike_func(V^n[t-1])
    """

    def __init__(self, **kwargs):
        super(ConstantCurrentLIFModel, self).__init__()
        # initial value for state variables
        self.neuron_parameters['tau_m'] = kwargs.get('tau_m', 20.0)
        self.neuron_parameters['v_th'] = kwargs.get('v_th', 1.0)
        self.neuron_parameters['v_reset'] = kwargs.get('v_reset', 0.0)

        self._variables['V'] = 0.0
        self._variables['O'] = 0.0
        self._variables['I'] = 0.0
        self._variables['WgtSum'] = 0.0
        # beta = self.neuron_parameters['tau_p'] / self.neuron_parameters['tau_q']
        V0 = 1.0  # (1 / (beta - 1)) * (beta ** (beta / (beta - 1)))
        self._constant_variables['V0'] = V0

        self._constant_variables['Vth'] = self.neuron_parameters['v_th']
        self._constant_variables['Vreset'] = self.neuron_parameters['v_reset']

        self._membrane_variables['tauM'] = self.neuron_parameters['tau_m']

        self._operations.append(('I', 'var_mult', 'V0', 'WgtSum'))
        self._operations.append(('decayV', 'minus', 'I', 'V'))
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
    I^n[t] = V0 * WgtSum^n[t-1]                # sum(w * O^(n-1)[t])
    M^n[t] = tauP * M^n[t-1] + I^n[t-1]        # tauP: decaying time constants of membrane integration
    S^n[t] = tauQ * S^n[t-1] + I^n[t-1]        # tauQ: decaying time constants of synaptic currents
    PSP = M - S
    """

    def __init__(self, **kwargs):
        super(NonSpikingLIFModel, self).__init__()
        # initial value for state variables
        self.neuron_parameters['tau_p'] = kwargs.get('tau_p', 4.0)
        self.neuron_parameters['tau_q'] = kwargs.get('tau_q', 1.0)
        self.neuron_parameters['tau_m'] = kwargs.get('tau_m', 1.0)
        self.neuron_parameters['v_th'] = kwargs.get('v_th', 1.0)
        self.neuron_parameters['v_reset'] = kwargs.get('v_reset', 0.0)

        self._variables['I'] = 0.0
        self._variables['M'] = 0.0
        self._variables['S'] = 0.0
        self._variables['V'] = 0.0
        self._variables['O'] = 0.0
        self._variables['WgtSum'] = 0.0
        beta = self.neuron_parameters['tau_p'] / self.neuron_parameters['tau_q']
        V0 = (1 / (beta - 1)) * (beta ** (beta / (beta - 1)))
        self._constant_variables['V0'] = V0

        self._membrane_variables['tauM'] = self.neuron_parameters['tau_m']
        # self._tau_constant_variables['tauM'] = self.neuron_parameters['tau_m']

        self._tau_constant_variables['tauP'] = self.neuron_parameters['tau_p']
        self._tau_constant_variables['tauQ'] = self.neuron_parameters['tau_q']

        self._operations.append(('I', 'var_mult', 'V0', 'WgtSum'))
        self._operations.append(('M', 'var_linear', 'tauP', 'M', 'I'))
        self._operations.append(('S', 'var_linear', 'tauQ', 'S', 'I'))
        self._operations.append(('PSP', 'minus', 'M', 'S'))
        self._operations.append(('decayV', 'minus', 'PSP', 'V'))
        self._operations.append(('V', 'var_linear', 'tauM', 'decayV', 'V'))


NeuronModel.register("nonspikinglif", NonSpikingLIFModel)

class LIFMModel(NeuronModel):
    """
    LIF model:
    # I = tauP*I + WgtSum^n[t-1] + b^n                         # sum(w * O^(n-1)[t])
    # F = tauM * exp(-O^n[t-1] / tauM)
    # V(t) = V^n[t-1] * F + I
    # O^(n)[t] = spike_func(V^n(t))
    """
    def __init__(self, **kwargs):
        super(LIFMModel, self).__init__()
        # initial value for state variables
        self.neuron_parameters['tau_p'] = kwargs.get('tau_p', 1.0)
        self.neuron_parameters['tau_m'] = kwargs.get('tau_m', 10.0)
        self.neuron_parameters['v_th']  = kwargs.get('v_th', 1.0)

        self._variables['V'] = 0.0
        self._variables['O'] = 0.0
        self._variables['WgtSum'] = 0.0
        self._variables['b'] = 0.0
        self._variables['I_che'] = 0.0
        self._variables['I_ele'] = 0.0
        self._variables['I'] = 0.0

        self._constant_variables['Vth'] = self.neuron_parameters['v_th']

        self._tau_constant_variables['tauM'] = self.neuron_parameters['tau_m']
        self._tau_constant_variables['tauP'] = self.neuron_parameters['tau_p']

        self._operations.append(('PSP', 'add', 'WgtSum[updated]', 'b'))

        self._operations.append(('I_che', 'var_linear', 'tauP', 'I_che', 'PSP'))
        self._operations.append(('I', 'add', 'I_che[updated]', 'I_ele'))
        self._operations.append(('Vtemp', 'var_linear', 'V', 'tauM', 'I[updated]'))
        self._operations.append(('O', 'threshold', 'Vtemp', 'Vth'))
        self._operations.append(('Vreset', 'var_mult', 'Vtemp', 'O[updated]'))
        self._operations.append(('V', 'minus', 'Vtemp', 'Vreset'))

NeuronModel.register("lifm", LIFMModel)

class IZHModel(NeuronModel):
    """
    IZH model:
    V = V + dt / tauM * (C1 * V * V + C2 * V + C3 - U + PSP)  # tauM=1 此处加tauM是为了添加op时和LIF模型保存一致
    V = V + dt / tauM * (V* (C1 * V + C2) + C3 - U + PSP)     # 由上式拆分而来
    U = U + a. * (b. * V - U)
    PSP = M^n[t] - S^n[t]
    M^n[t] = tauP * M^n[t-1] + I^n[t-1]        # tauP: decaying time constants of membrane integration
    S^n[t] = tauQ * S^n[t-1] + I^n[t-1]        # tauQ: decaying time constants of synaptic currents
    I^n[t] = V0 * WgtSum^n[t-1]                # WgtSum = sum(w * O^(n-1)[t])
    O^n[t] = spike_func(V^n[t-1])
    """

    def __init__(self, **kwargs):
        super(IZHModel, self).__init__()
        # initial value for state variables
        self.neuron_parameters['tau_p'] = kwargs.get('tau_p', 4.0)
        self.neuron_parameters['tau_q'] = kwargs.get('tau_q', 1.0)
        self.neuron_parameters['a'] = kwargs.get('a', 0.02)
        self.neuron_parameters['b'] = kwargs.get('b', 0.2)
        self.neuron_parameters['c'] = kwargs.get('c', -65.0)
        self.neuron_parameters['d'] = kwargs.get('d', 8.0)

        self._variables['I'] = 0.0
        self._variables['M'] = 0.0
        self._variables['S'] = 0.0
        self._variables['O'] = 0.0
        self._variables['V'] = -65.0  # self.neuron_parameters['c']
        self._variables['U'] = 1.0  # self.neuron_parameters['b']*self._variables['V']
        self._variables['WgtSum'] = 0.0  # 1.8
        beta = self.neuron_parameters['tau_p'] / self.neuron_parameters['tau_q']
        V0 = (1 / (beta - 1)) * (beta ** (beta / (beta - 1)))
        self._constant_variables['V0'] = V0

        self._constant_variables['a'] = self.neuron_parameters['a']
        self._constant_variables['b'] = self.neuron_parameters['b']
        self._constant_variables['Vth'] = 30.0
        self._variables['Vreset'] = self.neuron_parameters['c'] - self._constant_variables['Vth']  # 为了发放后将电压重置为-65
        self._constant_variables['Ureset'] = self.neuron_parameters['d']  # 8.0
        self._constant_variables['C1'] = 0.04
        self._constant_variables['C2'] = 5
        self._constant_variables['C3'] = 140

        self._membrane_variables['tauM'] = 1.0

        self._tau_constant_variables['tauP'] = self.neuron_parameters['tau_p']
        self._tau_constant_variables['tauQ'] = self.neuron_parameters['tau_q']

        # V = V + dt / tauM * (C1 * V * V + C2 * V + C3 - U + PSP)
        # V = V + dt / tauM * (V* (C1 * V + C2) + C3 - U + PSP)
        # U = U + dt /tauM * a. * (b. * V - U)

        self._operations.append(('I', 'var_mult', 'V0', 'WgtSum'))
        self._operations.append(('M', 'var_linear', 'tauP', 'M', 'I[updated]'))
        self._operations.append(('S', 'var_linear', 'tauQ', 'S', 'I[updated]'))
        self._operations.append(('PSP', 'minus', 'M[updated]', 'S[updated]'))

        self._operations.append(('O', 'threshold', 'V', 'Vth'))

        self._operations.append(('VResetting', 'var_mult', 'Vreset', 'O[updated]'))
        self._operations.append(('Vtemp', 'add', 'V', 'VResetting'))
        self._operations.append(('UResetting', 'var_mult', 'Ureset', 'O[updated]'))
        self._operations.append(('Utemp', 'add', 'U', 'UResetting'))

        self._operations.append(('temp_V1', 'var_linear', 'C1', 'Vtemp', 'C2'))
        self._operations.append(('temp_V2', 'var_linear', 'temp_V1', 'Vtemp', 'C3'))
        self._operations.append(('temp_V3', 'minus', 'temp_V2', 'Utemp'))
        self._operations.append(('temp_V4', 'add', 'temp_V3', 'PSP'))
        self._operations.append(('V', 'var_linear', 'tauM', 'temp_V4', 'Vtemp'))

        self._operations.append(('temp_U1', 'var_mult', 'b', 'V'))
        self._operations.append(('temp_U2', 'minus', 'temp_U1', 'Utemp'))
        self._operations.append(('U', 'var_linear', 'a', 'temp_U2', 'Utemp'))

        # self._operations.append(('V', 'izh_v', 'V', 'U', 'PSP'))
        # self._operations.append(('U', 'izh_u', 'a', 'b', 'V[updated]', 'U'))

NeuronModel.register("izh", IZHModel)

class aEIFModel(NeuronModel):
    """
    aEIF model:

    V = V + dt / tauM * (EL - V + EXP - U + I^n[t])
    U = U + dt / tauW * (a * (V - EL) - U)
    EXP = delta_t * delta_t2 * exp(du_th/delta_t2)
    du = V - EL
    du_th = V - Vth
    I^n[t] = V0 * WgtSum^n[t-1]

    O^n[t] = spike_func(V^n[t-1])
    """

    def __init__(self, **kwargs):
        super(aEIFModel, self).__init__()
        # initial value for state variables
        self.neuron_parameters['tau_p']   = kwargs.get('tau_p', 12.0)
        self.neuron_parameters['tau_q']   = kwargs.get('tau_q', 4.0)
        self.neuron_parameters['tau_w']   = kwargs.get('tau_w', 144)
        self.neuron_parameters['tau_m']   = kwargs.get('tau_m', 1.0)
        self.neuron_parameters['a']       = kwargs.get('a', 0.05)
        self.neuron_parameters['b']       = kwargs.get('b', 0.0805)
        self.neuron_parameters['delta_t'] = kwargs.get('delta_t', 30.0)
        self.neuron_parameters['delta_t2'] = kwargs.get('delta_t2', 1.0)
        self.neuron_parameters['EL']      = kwargs.get('EL', -70.6)

        self._variables['I'] = 0.0
        self._variables['I_che'] = 0.0
        self._variables['I_ele'] = 0.0
        self._variables['M'] = 0.0
        self._variables['S'] = 0.0
        self._variables['O'] = 0.0
        self._variables['V'] = -70.6
        # self._variables['Vt'] = -70.6
        self._variables['U'] = 0.0 # self.neuron_parameters['b'] * (-70.6)
        self._variables['WgtSum'] = 0.0

        self._variables['EXP'] = 0.0

        beta = self.neuron_parameters['tau_p'] / self.neuron_parameters['tau_q']
        V0 = (1 / (beta - 1)) * (beta ** (beta / (beta - 1)))
        self._constant_variables['V0'] = V0

        self._constant_variables['EL'] = self.neuron_parameters['EL']
        self._constant_variables['a'] = self.neuron_parameters['a']
        self._constant_variables['b'] = self.neuron_parameters['b']
        self._constant_variables['Vth'] = -50.4
        # self._constant_variables['Ureset'] = -8.0
        # self._constant_variables['C1'] = 0.6

        self._constant_variables['delta_t'] = self.neuron_parameters['delta_t'] * self.neuron_parameters['delta_t2']
        self._constant_variables['delta_t2'] = self.neuron_parameters['delta_t2']  # *10

        self._membrane_variables['tauM'] = self.neuron_parameters['tau_m']

        self._membrane_variables['tauW'] = self.neuron_parameters['tau_w']

        # V = V + dt / tauM * (EL - V + EXP - U + I ^ n[t])
        # U = U + dt / tauW * (a * (V - EL) - U)
        # EXP = delta_t * exp(du_th / delta_t2)
        # du = V - EL
        # du_th = V - Vth
        # I ^ n[t] = V0 * WgtSum ^ n[t - 1]
        #
        # O ^ n[t] = spike_func(V ^ n[t - 1])
        self._operations.append(('I_che', 'var_mult', 'V0', 'WgtSum[updated]'))
        self._operations.append(('I', 'add', 'I_che[updated]', 'I_ele[updated]'))

        self._operations.append(('dv', 'minus', 'V', 'EL'))
        self._operations.append(('dv_th', 'minus', 'V', 'Vth'))

        self._operations.append(('EXP_T1', 'div', 'dv_th', 'delta_t2'))
        self._operations.append(('EXP_T2', 'exp', 'EXP_T1'))
        self._operations.append(('EXP', 'var_mult', 'delta_t', 'EXP_T2'))

        self._operations.append(('temp_V1', 'minus', 'EXP[updated]', 'dv'))
        self._operations.append(('temp_V2', 'minus', 'temp_V1', 'U'))
        self._operations.append(('temp_V3', 'add', 'temp_V2', 'I'))
        self._operations.append(('Vt', 'var_linear', 'tauM', 'temp_V3', 'V'))

        self._operations.append(('temp_U1', 'var_mult', 'a', 'dv'))
        self._operations.append(('temp_U2', 'minus', 'temp_U1', 'U'))
        self._operations.append(('Ut', 'var_linear', 'tauW', 'temp_U2', 'U'))

        self._operations.append(('O', 'threshold', 'Vt', 'Vth'))

        self._operations.append(('Vtemp2', 'var_mult', 'Vt', 'O[updated]'))
        self._operations.append(('Vtemp3', 'minus', 'Vt', 'Vtemp2'))
        self._operations.append(('V', 'var_linear', 'EL', 'O[updated]', 'Vtemp3'))
        self._operations.append(('U', 'var_linear', 'b', 'O[updated]', 'Ut'))

NeuronModel.register("aeif", aEIFModel)
NeuronModel.register("adex", aEIFModel)


class GLIFModel(NeuronModel):
    """
    Current GLIF5 model:

    V = V + dt/tau_m * (R * (I + I1 + I2) - (V - E_L))
    Theta_s = Theta_s - b_s * Theta_s
    I_j = I_j - k_j * I_j (j = 1, 2)
    Theta_v = Theta_v + a_v * (V - E_L) - b_v * Theta_v
    v_th = Theta_v + Theta_s + Theta_inf
    O = spike_func(V)

    Reset function:

    V = E_L + f_v * (V - E_L) - delta_v
    Theta_s = Theta_s + delta_Theta_s
    I_j = f_j * I_j + delta_I_j (j = 1, 2; f_j = 1)
    Theta_v = Theta_v

    After transform:
    part I:
    I^n[t] = tau_p * I + WgtSum^n[t-1]
    I_sum = I1 + I2
    I_sum = I_sum + I^n[t]
    S^n[t] = tau_q * S^n[t-1] + I_sum

    dv = V^n[t-1] - E_L
    decay = S^n[t] - dv
    V = tauM * decay + V

    Theta_s = (-b_s) * Theta_s + Theta_s
    I_j = (-k_j) * I_j + I_j
    Theta_temp = a_v * dv + Theta_v
    Theta_v = (-b_v) * Theta_v + Theta_temp
    Theta = Theta_v + Theta_s
    v_th = Theta + Theta_inf
    O = spike_func(V, v_th)

    part II(reset part):
    dv_reset = V^n[t] - E_L
    deltaV(constant) = E_L - delta_v
    Vreset = f_v * dv_reset + deltaV
    Vreset = Vreset - V

    V = Vreset * O + V
    Theta_s = Theta_s + delta_Theta_s * O
    I_j = I_j + delta_I_j * O

    """
    def __init__(self, **kwargs):
        super(GLIFModel, self).__init__()
        self.neuron_parameters['R']             = kwargs.get('R', 0.2)
        self.neuron_parameters['C']             = kwargs.get('C', 1.0)
        self.neuron_parameters['E_L']           = kwargs.get('E_L', 0.0)
        self.neuron_parameters['Theta_inf']     = kwargs.get('Theta_inf', 1.0)
        # self.neuron_parameters['delta_t']       = kwargs.get('delta_t', 0.0)
        self.neuron_parameters['f_v']           = kwargs.get('f_v', 0.1)
        self.neuron_parameters['delta_v']       = kwargs.get('delta_v', 0.05)
        self.neuron_parameters['b_s']           = -kwargs.get('b_s', 0.02)
        self.neuron_parameters['delta_Theta_s'] = kwargs.get('delta_Theta_s', 0.02)
        self.neuron_parameters['k_1']           = -kwargs.get('k_1', 0.1)
        self.neuron_parameters['k_2']           = -kwargs.get('k_2', 0.1)
        self.neuron_parameters['delta_I1']      = kwargs.get('delta_I1', 1.0)
        self.neuron_parameters['delta_I2']      = kwargs.get('delta_I2', 1.0)
        self.neuron_parameters['a_v']           = kwargs.get('a_v', 0.05)
        self.neuron_parameters['b_v']           = -kwargs.get('b_v', 0.1)
        self.neuron_parameters['tau_p']         = kwargs.get('tau_p', 1.0)
        self.neuron_parameters['tau_q']         = kwargs.get('tau_q', 1.0)

        self._variables['V']       = 0.0
        self._variables['Theta_s'] = 0.1
        self._variables['Theta_v'] = 0.2
        self._variables['I1']      = 0.08
        self._variables['I2']      = 0.12
        self._variables['I']       = 0.0
        self._variables['S']       = 0.0
        self._variables['O']       = 0.0
        self._variables['v_th']    = 0.0

        self._constant_variables['deltaV'] = self.neuron_parameters['E_L'] - self.neuron_parameters['delta_v']
        self._constant_variables['R'] = self.neuron_parameters['R']
        self._constant_variables['C'] = self.neuron_parameters['C']
        self._constant_variables['E_L'] = self.neuron_parameters['E_L']
        self._constant_variables['Theta_inf'] = self.neuron_parameters['Theta_inf']
        self._constant_variables['f_v'] = self.neuron_parameters['f_v']
        self._constant_variables['delta_v'] = self.neuron_parameters['delta_v']
        self._constant_variables['b_s'] = self.neuron_parameters['b_s']
        self._constant_variables['delta_Theta_s'] = self.neuron_parameters['delta_Theta_s']
        self._constant_variables['k1'] = self.neuron_parameters['k_1']
        self._constant_variables['k2'] = self.neuron_parameters['k_2']
        self._constant_variables['delta_I1'] = self.neuron_parameters['delta_I1']
        self._constant_variables['delta_I2'] = self.neuron_parameters['delta_I2']
        self._constant_variables['a_v'] = self.neuron_parameters['a_v']
        self._constant_variables['b_v'] = self.neuron_parameters['b_v']

        self._membrane_variables['tau_m']   = self.neuron_parameters['R'] * self.neuron_parameters['C']
        self._tau_constant_variables['tauP']    = self.neuron_parameters['tau_p']
        self._tau_constant_variables['tauQ']    = self.neuron_parameters['tau_q']
        # self._tau_constant_variables['k1']      = self.neuron_parameters['k_1']
        # self._tau_constant_variables['k2']      = self.neuron_parameters['k_2']

        self._operations.append(('I', 'var_linear', 'tauP', 'I', 'WgtSum'))
        self._operations.append(('I_sum', 'add', 'I1', 'I2'))
        self._operations.append(('I_sum', 'add', 'I_sum', 'I[updated]'))
        self._operations.append(('S', 'var_linear', 'tauQ', 'S', 'I_sum'))

        self._operations.append(('dv', 'minus', 'V', 'E_L'))
        self._operations.append(('decay', 'minus', 'S[updated]', 'dv'))
        self._operations.append(('Vt', 'var_linear', 'tau_m', 'decay', 'V'))

        self._operations.append(('Theta_st', 'var_linear', 'b_s', 'Theta_s', 'Theta_s'))
        self._operations.append(('I1t', 'var_linear', 'k1', 'I1', 'I1'))
        self._operations.append(('I2t', 'var_linear', 'k2', 'I2', 'I2'))
        self._operations.append(('Theta_temp', 'var_linear', 'a_v', 'dv', 'Theta_v'))
        self._operations.append(('Theta_v', 'var_linear', 'b_v', 'Theta_v', 'Theta_temp'))
        self._operations.append(('Theta', 'add', 'Theta_v[updated]', 'Theta_st'))
        self._operations.append(('v_th', 'add', 'Theta', 'Theta_inf'))

        self._operations.append(('O', 'threshold', 'Vt', 'v_th[updated]'))

        self._operations.append(('I1', 'var_linear', 'delta_I1', 'O[updated]', 'I1t'))
        self._operations.append(('I2', 'var_linear', 'delta_I2', 'O[updated]', 'I2t'))
        self._operations.append(('Theta_s', 'var_linear', 'delta_Theta_s', 'O[updated]', 'Theta_st'))
        self._operations.append(('dv_reset', 'minus', 'Vt', 'E_L'))
        self._operations.append(('Vreset', 'var_linear', 'f_v', 'dv_reset', 'deltaV'))
        self._operations.append(('Vreset', 'add', 'Vreset', 'E_L'))
        self._operations.append(('Vreset', 'minus', 'Vreset', 'Vt'))

        self._operations.append(('V', 'var_linear', 'Vreset', 'O[updated]', 'Vt'))

NeuronModel.register("glif", GLIFModel)


class HodgkinHuxleyModel(NeuronModel):
    """
    Hodgkin-Huxley model:

    V = V + dt/tau_v * (I - Ik)
    Ik = NA + K + L
    NA = g_NA * m^3 * h * (V - V_NA)
    K = g_K * n^4 * (V - V_K)
    L = g_L * (V - V_L)

    m = m + dt/tau_m * (alpha_m * (1-m) - beta_m * m)
    n = n + dt/tau_n * (alpha_n * (1-n) - beta_n * n)
    h = h + dt/tau_h * (alpha_h * (1-h) - beta_h * h)

    original function:
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
        # initial value for state variables
        self.neuron_parameters['dt'] = kwargs.get('dt', 0.1)

        self.neuron_parameters['g_NA'] = kwargs.get('g_NA', 120.0)
        self.neuron_parameters['g_K'] = kwargs.get('g_K', 36.0)
        self.neuron_parameters['g_L'] = kwargs.get('g_L', 0.3)

        self.neuron_parameters['E_NA']      = kwargs.get('E_NA', 120.0)
        self.neuron_parameters['E_K']      = kwargs.get('E_K', -12.0)
        self.neuron_parameters['E_L']      = kwargs.get('E_L', 10.6)

        # a_m = a_m1 * (-V + a_m2) / (exp((-V + a_m2) / a_m3 - 1)
        # b_m = b_m1 * exp((-V + b_m2) / b_m3)
        self.neuron_parameters['a_m1'] = kwargs.get('alpha_m1', 0.1)
        self.neuron_parameters['a_m2'] = kwargs.get('alpha_m2', 25.0)
        self.neuron_parameters['a_m3'] = kwargs.get('alpha_m3', 10.0)
        # self.neuron_parameters['a_m4'] = kwargs.get('alpha_m4', -1)
        self.neuron_parameters['b_m1'] = kwargs.get('beta_m1', 4.0)
        self.neuron_parameters['b_m2'] = kwargs.get('beta_m2', 0.0)
        self.neuron_parameters['b_m3'] = kwargs.get('beta_m3', 18.0)

        # a_n = a_n1 * (-V + a_n2) / (exp((-V + a_n2) / a_n3 - 1)
        # b_n = b_n1 * exp((-V + b_n2) / b_n3)
        self.neuron_parameters['a_n1'] = kwargs.get('alpha_n1', 0.01)
        self.neuron_parameters['a_n2'] = kwargs.get('alpha_n2', 10.0)
        self.neuron_parameters['a_n3'] = kwargs.get('alpha_n3', 10.0)
        # self.neuron_parameters['a_n4'] = kwargs.get('alpha_n4', -1)
        self.neuron_parameters['b_n1'] = kwargs.get('beta_n1', 0.125)
        self.neuron_parameters['b_n2'] = kwargs.get('beta_n2', 0.0)
        self.neuron_parameters['b_n3'] = kwargs.get('beta_n3', 80.0)

        # a_h = a_h1 * exp((-V + a_h2) / a_h3)
        # b_h = b_h1 / (exp((-V + b_h2) / b_h3 + 1)
        self.neuron_parameters['a_h1'] = kwargs.get('alpha_h1', 0.07)
        self.neuron_parameters['a_h2'] = kwargs.get('alpha_h2', 0.0)
        self.neuron_parameters['a_h3'] = kwargs.get('alpha_h3', 20.0)
        self.neuron_parameters['b_h1'] = kwargs.get('beta_h1', 1.0) # 分子
        self.neuron_parameters['b_h2'] = kwargs.get('beta_h2', 30.0)
        self.neuron_parameters['b_h3'] = kwargs.get('beta_h3', 10.0)

        self.neuron_parameters['65'] = kwargs.get('V65', 0.0)
        self.neuron_parameters['m'] = kwargs.get('m', 0.5)
        self.neuron_parameters['n'] = kwargs.get('n', 0.5)
        self.neuron_parameters['h'] = kwargs.get('h', 0.06)
        # self.neuron_parameters['b_h4'] = kwargs.get('beta_h4', 1)

        self.neuron_parameters['V'] = kwargs.get('V', 0.0)
        self.neuron_parameters['vth'] = kwargs.get('vth', 1.0)

        self._variables['I'] = 0.0
        self._variables['O'] = 0.0
        self._variables['V'] = self.neuron_parameters['V']

        self._variables['m'] = self.neuron_parameters['m']
        self._variables['n'] = self.neuron_parameters['n']
        self._variables['h'] = self.neuron_parameters['h']

        # beta = self.neuron_parameters['tau_p'] / self.neuron_parameters['tau_q']
        # V0 = (1 / (beta - 1)) * (beta ** (beta / (beta - 1)))
        self._constant_variables['V0'] = 1.0
        self._constant_variables['Vth'] = self.neuron_parameters['vth']
        self._constant_variables['Vreset'] = 0.0

        self._membrane_variables['tauM'] = 1.0
        self._membrane_variables['tauN'] = 1.0
        self._membrane_variables['tauH'] = 1.0
        self._membrane_variables['tauV'] = 1.0

        self._constant_variables['1'] = 1.0
        self._constant_variables['65'] = self.neuron_parameters['65']
        self._constant_variables['dt'] = self.neuron_parameters['dt']

        self._constant_variables['g_NA'] = self.neuron_parameters['g_NA']
        self._constant_variables['g_K'] = self.neuron_parameters['g_K']
        self._constant_variables['g_L'] = self.neuron_parameters['g_L']

        self._constant_variables['E_NA'] = self.neuron_parameters['E_NA']
        self._constant_variables['E_K'] = self.neuron_parameters['E_K']
        self._constant_variables['E_L'] = self.neuron_parameters['E_L']

        self._constant_variables['a_m1'] = self.neuron_parameters['a_m1']
        self._constant_variables['a_m2'] = self.neuron_parameters['a_m2']
        self._constant_variables['a_m3'] = self.neuron_parameters['a_m3']
        self._constant_variables['b_m1'] = self.neuron_parameters['b_m1']
        self._constant_variables['b_m2'] = self.neuron_parameters['b_m2']
        self._constant_variables['b_m3'] = self.neuron_parameters['b_m3']

        self._constant_variables['a_n1'] = self.neuron_parameters['a_n1']
        self._constant_variables['a_n2'] = self.neuron_parameters['a_n2']
        self._constant_variables['a_n3'] = self.neuron_parameters['a_n3']
        self._constant_variables['b_n1'] = self.neuron_parameters['b_n1']
        self._constant_variables['b_n2'] = self.neuron_parameters['b_n2']
        self._constant_variables['b_n3'] = self.neuron_parameters['b_n3']

        self._constant_variables['a_h1'] = self.neuron_parameters['a_h1']
        self._constant_variables['a_h2'] = self.neuron_parameters['a_h2']
        self._constant_variables['a_h3'] = self.neuron_parameters['a_h3']
        self._constant_variables['b_h1'] = self.neuron_parameters['b_h1']
        self._constant_variables['b_h2'] = self.neuron_parameters['b_h2']
        self._constant_variables['b_h3'] = self.neuron_parameters['b_h3']

        self._operations.append(('V65', 'add', 'V', '65'))

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
        self._operations.append(('I', 'var_mult', 'V0', 'WgtSum'))
        self._operations.append(('Vtemp', 'minus', 'I[updated]', 'Ik'))
        self._operations.append(('V', 'add', 'V', 'Vtemp'))

        # O
        self._operations.append(('O', 'threshold', 'V[updated]', 'Vth'))


NeuronModel.register("hh", HodgkinHuxleyModel)



class LIFSTDPEXModel(NeuronModel):
    """
    LIF model:
    V(t) = decay_v * (v - v_rest) + v_rest + I^n[t]
    I^n[t] = V0 * WgtSum^n[t]  #V0 = 1
    theta(t) = decay_th * theta[t-1]
    if v >= (vth + theta) then s_out = 1; else s_out = 0;
    Reset:
    V(t) = s_out * v_reset + (1 - s_out) * v; theta = theta + s_out * th_inc

    O^n[t] = spike_func(V^n[t-1])
    """
    def __init__(self, **kwargs):
        super(LIFSTDPEXModel, self).__init__()
        # initial value for state variables
        self.neuron_parameters['decay_v'] = kwargs.get('decay_v', np.exp(-1/100))
        self.neuron_parameters['decay_th'] = kwargs.get('decay_th', np.exp(-1/1e7))
        self.neuron_parameters['th_inc'] = kwargs.get('th_inc', 0.05)
        self.neuron_parameters['v_th'] = kwargs.get('v_th', -52.0)
        self.neuron_parameters['v_rest'] = kwargs.get('v_rest', -65.0)
        self.neuron_parameters['v_reset'] = kwargs.get('v_reset', -60.0)

        self._variables['I'] = 0.0
        self._variables['V'] = -65.0
        self._variables['O'] = 0.0
        self._variables['WgtSum'] = 0.0
        self._variables['theta[stay]'] = 0.0
        self._variables['Vth_theta'] = 0.0

        self._constant_variables['V0'] = 1

        self._constant_variables['Vth'] = self.neuron_parameters['v_th']
        self._constant_variables['Vreset'] = self.neuron_parameters['v_reset']
        self._constant_variables['Vrest'] = self.neuron_parameters['v_rest']
        self._constant_variables['th_inc'] = self.neuron_parameters['th_inc']
        self._constant_variables['decay_th'] = self.neuron_parameters['decay_th']
        self._constant_variables['decay_v'] = self.neuron_parameters['decay_v']


        self._operations.append(('I', 'var_mult', 'V0', 'WgtSum[updated]'))
        self._operations.append(('PSP1', 'minus', 'V', 'Vrest'))
        self._operations.append(('PSP2', 'var_linear', 'decay_v', 'PSP1', 'Vrest'))
        self._operations.append(('Vtemp', 'add', 'PSP2', 'I[updated]'))
        self._operations.append(('theta_temp', 'var_mult', 'decay_th', 'theta[stay]'))
        self._operations.append(('Vth_theta', 'add', 'Vth', 'theta_temp'))
        self._operations.append(('O', 'threshold', 'Vtemp', 'Vth_theta'))
        self._operations.append(('Resetting1', 'var_mult', 'Vreset', 'O[updated]'))
        self._operations.append(('Resetting2', 'var_mult', 'Vtemp', 'O[updated]'))
        self._operations.append(('Resetting3', 'minus', 'Vtemp', 'Resetting2'))
        self._operations.append(('V', 'add', 'Resetting1', 'Resetting3'))
        self._operations.append(('Resetting_theta', 'var_mult', 'O[updated]', 'th_inc'))
        self._operations.append(('theta_stay', 'add', 'theta_temp', 'Resetting_theta'))


NeuronModel.register("lifstdp_ex", LIFSTDPEXModel)

class LIFSTDPIHModel(NeuronModel):
    """
    LIF model:
    V(t) = decay_v * (v - v_rest) + v_rest + I^n[t]
    I^n[t] = V0 * WgtSum^n[t]  #V0 = 1

    Reset:
    V(t) = s_out * v_reset + (1 - s_out) * v;

    O^n[t] = spike_func(V^n[t-1])
    """
    def __init__(self, **kwargs):
        super(LIFSTDPIHModel, self).__init__()
        # initial value for state variables
        self.neuron_parameters['decay_v'] = kwargs.get('decay_v', np.exp(-1/10))
        self.neuron_parameters['v_th'] = kwargs.get('v_th', -40.0)
        self.neuron_parameters['v_rest'] = kwargs.get('v_rest', -60.0)
        self.neuron_parameters['v_reset'] = kwargs.get('v_reset', -45.0)

        self._variables['I'] = 0.0
        self._variables['V'] = -60.0
        self._variables['O'] = 0.0
        self._variables['WgtSum'] = 0.0

        self._constant_variables['V0'] = 1

        self._constant_variables['Vth'] = self.neuron_parameters['v_th']
        self._constant_variables['Vreset'] = self.neuron_parameters['v_reset']
        self._constant_variables['Vrest'] = self.neuron_parameters['v_rest']
        self._constant_variables['decay_v'] = self.neuron_parameters['decay_v']


        self._operations.append(('I', 'var_mult', 'V0', 'WgtSum[updated]'))
        self._operations.append(('PSP1', 'minus', 'V', 'Vrest'))
        self._operations.append(('PSP2', 'var_linear', 'decay_v', 'PSP1', 'Vrest'))
        self._operations.append(('Vtemp', 'add', 'PSP2', 'I[updated]'))
        self._operations.append(('O', 'threshold', 'Vtemp', 'Vth'))
        self._operations.append(('Resetting1', 'var_mult', 'Vreset', 'O[updated]'))
        self._operations.append(('Resetting2', 'var_mult', 'Vtemp', 'O[updated]'))
        self._operations.append(('Resetting3', 'minus', 'Vtemp', 'Resetting2'))
        self._operations.append(('V', 'add', 'Resetting1', 'Resetting3'))


NeuronModel.register("lifstdp_ih", LIFSTDPIHModel)


class MeanFieldModel(NeuronModel):
    """
    Mean Field Model of LIF neuron "

    U = U + dt/tau * (rho*(Iext + WgtSum) - U)
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
        self._variables['WgtSum'] = 0.0
        self._variables['U'] = 0.0
        self._variables['O'] = 0.0

        self._operations.append(('Isum', 'var_linear', 'rho', 'WgtSum', 'Iext'))
        self._operations.append(('dU', 'minus', 'Isum', 'U'))
        self._operations.append(('U', 'var_linear', 'tau', 'dU', 'U'))
        self._operations.append(('O', 'relu', 'U[updated]'))

NeuronModel.register("meanfield", MeanFieldModel)