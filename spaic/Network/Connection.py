# -*- coding: utf-8 -*-
"""
Created on 2020/8/5
@project: SNNFlow
@filename: Connection
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
定义神经集群间的连接，包括记录神经元集群、连接的突触前、突触后神经元编号、连接形式（全连接、稀疏连接、卷积）、权值、延迟 以及连接产生函数、重连接函数等。
"""
from ..Network.BaseModule import BaseModule
from ..Network.Assembly import Assembly
from ..Network.ConnectPolicy import ConnectInformation, ConnectPolicy
from collections import OrderedDict
from typing import Dict, List, Tuple
from abc import abstractmethod
import numpy as np
import scipy.sparse as sp
import torch

class Connection(BaseModule):
    '''
    Base class for all kinds of connections, including full connection, sparse connection, conv connection,....

    Args:
        pre_assembly(Assembly): the assembly which needs to be connected.
        post_assembly(Assembly): the assembly which needs to connect the pre_assembly.
        link_type(str): the type for connection: full, sparse, conv...

    Attributes:
        pre_group(groups): the neuron group which need to be connected in the pre_assembly.
        post_group(groups): the neuron group which need to connect with pre_group neuron.
        unit_connections(list): a list contain unit_connect information: pre_group, post_group, link_num, var_code, op_code.
        _var_names(list): a list contain variable names.

    Methods:
        __new__: before build a new connection, do some checks.
        get_var_names: get variable names.
        register: register a connection class.
        build: add the connection variable, variable name and opperation to the simulator.
        get_str:
        unit_connect: define the basic connection information(the connection weight, the connection shape, the backend variable and the backend basic operation) and add them to the connection_information.
        condition_check: check whether the pre_group.type is equal to the post_group.type, only if they are equal, return flag=Ture.
        connect: connect the preg with postg.
        get_weight_name: give a name for each connection weight.
        get_target_name: give a name for each target group.
        get_input_name: give a name for each input group.

    Examples:
        when building the network:
        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full')

        '''

    _connection_subclasses = dict()
    _class_label = '<con>'

    def __init__(self, pre_assembly: Assembly, post_assembly: Assembly, name=None, link_type=('full', 'sparse', 'conv', '...'),
                  policies=[], max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):

        super(Connection, self).__init__()


        self.pre_assembly = pre_assembly#.get_assemblies()
        self.post_assembly = post_assembly#.get_assemblies()
        self.pre_var_name = pre_var_name
        self.post_var_name = post_var_name

        self.pre_groups = None #pre_assembly.get_groups()
        self.post_groups = None #post_assembly.get_groups()
        self.pre_assemblies = None
        self.post_assemblies = None

        self.connection_inforamtion = None
        self.unit_connections: List[(Assembly, Assembly, int, tuple, tuple)] = list()  # (pre_group, post_group, link_num, var_code, op_code) TODO: change to List[info_object]
        self.mask_info: List[(tuple, tuple)] = list()  # (var_code, op_code)

        self.link_type = link_type
        self.max_delay = max_delay
        self.sparse_with_mask = sparse_with_mask
        self._var_names = list()
        self._supers = list()
        self._link_var_codes = list()
        self._link_op_codes = list()
        self._policies = policies

        self.parameters = kwargs

        self.w_max = None
        self.w_min = None

        self.set_name(name)
        self.running_var = None
        self.running_mean = None
        self.decay = 0.9
        self._syn_operations = []
        self._syn_variables = dict()
        self._syn_tau_constant_variables = dict()
        self._syn_constant_variables = dict()
        self._var_names = list()
        self.synapse = kwargs.get('synapse', False)
        self.synapse_type = kwargs.get('synapse_type', 'chemistry_i_synapse')
        self.tau_p = kwargs.get('tau_p', 12.0)

        # construct unit connection information by policies,
        # construct in __init__ is potentially bad, as network structure may change before build. should add new function
        self.connection_inforamtion = ConnectInformation(self.pre_assembly, self.post_assembly)
        self.connection_inforamtion.expand_connection()
        for p in self._policies:
            self.connection_inforamtion = self.connection_inforamtion & p.generate_connection(self.pre_assembly, self.post_assembly)

    def norm_hook(self, grad):
        import torch
        if self.running_var is None:
            self.running_var = torch.norm(grad, dim=1,keepdim=True)
            self.running_mean = torch.mean(grad, dim=1,keepdim=True)
        else:
            self.running_var = self.decay * self.running_var + (1 - self.decay) * torch.norm(grad, dim=0)
            self.running_mean = self.decay * self.running_mean + (1 - self.decay) * torch.mean(grad, dim=0)
        return (grad - self.running_mean) / (1.0e-10 + self.running_var)



    def __new__(cls, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse', 'conv','...'), policies=[],
                max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):
        if cls is not Connection:
            return super().__new__(cls)

        if link_type in cls._connection_subclasses:
            return cls._connection_subclasses[link_type](pre_assembly, post_assembly, name, link_type, policies, max_delay, sparse_with_mask, pre_var_name, post_var_name, **kwargs)

        else:
            raise ValueError("No connection type: %s in Connection classes" %link_type)

    def get_var_names(self):
        return self._var_names

    @staticmethod
    def register(name, connection_class):
        '''
        Register a connection class. Registered connection classes can be referred to
        # via their name.
        Parameters
        ----------
        name : str
            A short name for the state updater (e.g. `'full'`)
        connection_class :
            The subclass of Connection object, e.g. an `FullConnection`, 'ConvConnection'.
        '''

        # only deal with lower case names
        name = name.lower()
        if name in Connection._connection_subclasses:
            raise ValueError(('A connection class with the name "%s" has already been registered') % name)

        if not issubclass(connection_class, Connection):
            raise ValueError(('Given model of type %s does not seem to be a valid ConnectionModel.' % str(type(connection_class))))

        Connection._connection_subclasses[name] = connection_class

    def assembly_linked(self, assembly):
        if (assembly is self.pre_assembly) or (assembly is self.post_assembly):
            return True
        else:
            return False

    def replace_assembly(self, old_assembly, new_assembly):
        if old_assembly is self.pre_assembly:
            self.pre_assembly = new_assembly
        elif old_assembly is self.post_assembly:
            self.post_assembly = new_assembly
        else:
            raise ValueError("the old_assembly is not in the connnection")

    def add_super(self, assembly):
        assert isinstance(assembly, Assembly), "the super is not Assembly"
        self._supers.append(assembly)

    def del_super(self, assembly):
        assert assembly in self._supers, "the assembly is not in supers"
        self._supers.remove(assembly)

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

    def get_str(self, level):

        level_space = "" + '-' * level
        repr_str = level_space + "|name:{}, type:{}, ".format(self.name, type(self).__name__)
        repr_str += "pre_assembly:{}, ".format(self.pre_assembly.name)
        repr_str += "post_assembly:{}\n ".format(self.post_assembly.name)
        # for c in self._connections.values():
        #     repr_str += c.get_str(level)
        return repr_str

    def set_delay(self, pre_group, post_group):
        # TODO: add to unit_connections information after it changed to List[dict]

        if self.max_delay > 0:
            # print("set delay")
            pre_num = pre_group.num
            post_num = post_group.num
            shape = (post_num, pre_num)
            delay_input_name, delay_output_name = self.get_delay_input_output(pre_group, post_group)

            # add delay container
            delay_queue = self._simulator.add_delay(delay_input_name, self.max_delay)
            delay_name = self.get_delay_name(pre_group, post_group)
            # ONLY FOR TEST  ===============================
            ini_delay = self.max_delay*np.random.rand(*shape)
            # ==============================================
            self._simulator.add_variable(delay_name, shape, ini_delay, True)
            self._var_names.append(delay_name)


            # add delay index
            self._simulator.register_standalone(delay_output_name, delay_queue.select, [delay_name])

            # add inital to transform initial delay_output
            self._simulator.register_initial(delay_output_name, delay_queue.transform_delay_output, [delay_input_name, delay_name])

        else:
            return


    def clamp_weight(self, weight):

        if (self.w_max is not None) and (self.w_min is not None):
            self._simulator.clamp_(weight, self.w_min, self.w_max)
        elif self.w_max is not None:
            self._simulator.clamp_max_(weight, self.w_max)
        elif self.w_min is not None:
            self._simulator.clamp_min_(weight, self.w_min)


    def build(self, simulator):
        '''
        add the connection variable, variable name and opperation to the simulator.
        '''
        self._simulator = simulator
        self.build_connect()

        for uc in self.unit_connections:
            # ToDO: This is too much! have to change the structure of unitconnection and its build

            # simulator.add_variable(uc[3][0], uc[3][1], uc[3][2], uc[3][3], uc[3][4])   # var_code
            simulator.add_variable(*uc[3])
            self._var_names.append(uc[3][0])

            if 'conv_max_pool2d' in uc[-1][1]:  # 如果需要max_pool,那么需要添加一个max_pool_size的参数和其他一些卷积参数
                for i in range(4, 9):
                    simulator.add_variable(*uc[i])  # var_code
                    self._var_names.append(uc[i][0])

                if self.synapse:
                    self.build_synapse()
                else:
                    simulator.add_operation(uc[-1])  # op_code

            elif 'conv_2d' in uc[-1][1]:# 不需要max_pool,添加卷积参数
                for i in range(4, 8):
                    simulator.add_variable(*uc[i])  # var_code
                    self._var_names.append(uc[i][0])

                if self.synapse:
                    self.build_synapse()
                else:
                    simulator.add_operation(uc[-1])# op_code

            else:
                if self.synapse:
                    self.build_synapse()
                else:
                    simulator.add_operation(uc[4])  # op_code


            if (self.w_min is not None) or (self.w_max is not None):
                simulator.register_initial(None, self.clamp_weight, [uc[3][0]])

        # for name in self._var_names:
        #     if 'weight' in name:
        #         self._simulator._parameters_dict[name].register_hook(self.norm_hook)

        for mask in self.mask_info:
            simulator.add_variable(*mask[0])
            self._var_names.append(mask[0][0])
            simulator.register_initial(*mask[1])

    @abstractmethod
    def unit_connect(self, pre_group, post_group):
        '''
        define the basic connection information(the connection weight, the connection shape, the backend variable and the backend basic operation),
        and add them to the connection_information.

        Args:
            pre_group(Groups): the neuron group which need to be connected in the pre_assembly.
            post_group(Groups): the neuron group which need to connect with pre_group neuron.

        Returns:
            (pre_group, post_group, link_num, var_code, op_code)

        '''
        # self.unit_connections.append((pre_group, post_goup, self.parameters?, number?, and building codes))
        NotImplemented()

    @abstractmethod
    def condition_check(self, pre_group, post_group):
        NotImplemented()

    def build_connect(self):

        for unit_con in self.connection_inforamtion.all_unit_connections:
            self.unit_connect(unit_con.pre, unit_con.post)
            self.set_delay(unit_con.pre, unit_con.post)

    def build_synapse(self):
        simulator = self._simulator
        self.dt = simulator.dt
        if self.synapse_type.lower() == 'chemistry_i_synapse':
            self.Chemistry_I_synapse()
        else:
            self.Electrical_synapse()

        for key, var in self._syn_tau_constant_variables.items():
            value = np.exp(-self.dt / var)
            simulator.add_variable(key, (), value)
            self._var_names.append(key)

        for key, value in self._syn_variables.items():

            simulator.add_variable(key, (), value)
            self._var_names.append(key)
        for key, value in self._syn_constant_variables.items():

            simulator.add_variable(key, (), value)
            self._var_names.append(key)

        for op in self._syn_operations:

            simulator.add_operation(op)




    def mask_operation(self, weight, mask):
        return weight*mask
    def get_link_name(self, pre_group: Assembly, post_group: Assembly, suffix_name: str):
        '''

        Args:
            pre_group(Assembly): the neuron group which needs to be connected
            post_group(Assembly): the neuron group which needs to connect with the pre_group
            suffix_name(str): represents the name of the object you want to retrieve, such as 'weight'
        Returns:
            name(str)
        '''

        name = self.id + ':' + post_group.id + '<-' + pre_group.id + ':' + '{' + suffix_name + '}'
        return name

    def get_pre_name(self, pre_group: Assembly, suffix_name: str):
        '''
        Args:
            pre_group(Assembly): the neuron group which needs to be connected
            suffix_name(str): represents the name of the object you want to retrieve, such as 'O'

        Returns:
            name(str)
        '''
        name = pre_group.id + ':' + '{' + suffix_name + '}'
        return name

    def get_post_name(self, post_group: Assembly, suffix_name: str):
        '''
        Args:
            post_group(Assembly): the neuron group which needs to connect with the pre_group
            suffix_name(str): represents the name of the object you want to retrieve, such as 'maxpool_kernel_size'

        Returns:
            name(str)
        '''
        name = post_group.id + ':' + '{' + suffix_name + '}'
        return name
    def get_weight_name(self, pre_group: Assembly, post_group: Assembly):

        '''
        give a name for each connection weight, the name consists of three parts: post_group.id + '<-' + pre_group.id + ':' + '{weight}'

        Args:
            pre_group(Assembly): the neuron group which needs to be connected
            post_group(Assembly): the neuron group which needs to connect with the pre_group
        Returns:
            name(str)
        '''


        name = self.id + ':' + post_group.id + '<-' + pre_group.id + ':' + '{weight}'

        return name

    def get_mask_name(self, pre_group: Assembly, post_group: Assembly):
        name = self.id + ':' + post_group.id + '<-' + pre_group.id + ':' + '{mask}'
        return name

    def get_delay_name(self, pre_group: Assembly, post_group: Assembly):
        name = self.id + ':' + post_group.id + '<-' + pre_group.id + ':' + '{delay}'
        return name

    def get_target_name(self, post_group:Assembly):
        '''
        Give a name for WgtSum,  the name consists of two parts: post_group.id + ':' + '{WgtSum}

        Args:
            post_group(Assembly): The neuron group which needs to connect with the pre_group

        Returns:
            name(str)
        '''
        name = post_group.id + ':' + '{' + self.post_var_name + '}'
        return name

    def get_input_name(self, pre_group: Assembly, post_group: Assembly):
        '''
        Give a name for input group's output spikes,  the name consists of two parts: pre_group.id + ':' + '{0}
        Args:
            pre_group(Assembly): The neuron group which need to connect with post_group neuron.

        Returns:
            name(str)
        '''

        if self.max_delay > 0:
            name = self.id + ':' + post_group.id + '<-' + pre_group.id + ':' + '{' + self.pre_var_name + '}'
        else:
            name = pre_group.id + ':' + '{' + self.pre_var_name + '}'

        return name



    def maxpool_kernel_size_name(self, pre_group: Assembly):
        '''
        Give a name for  maxpool_kernel_size,  the name consists of two parts: pre_group.id + '{maxpool_kernel_size}'
        Args:
            pre_group(Assembly): the neuron group which needs to be connected
            post_group(Assembly): the neuron group which needs to connect with the pre_group

        Returns:
            name(str)
        '''
        name = pre_group.id + ':' + '{maxpool_kernel_size}'
        return name
    def stride_name(self, pre_group: Assembly):
        name = pre_group.id + ':' + '{stride}'
        return name
    def padding_name(self, pre_group: Assembly):
        name = pre_group.id + ':' + '{padding}'
        return name
    def dilation_name(self, pre_group: Assembly):
        name = pre_group.id + ':' + '{dilation}'
        return name
    def groups_name(self, pre_group: Assembly):
        name = pre_group.id + ':' + '{groups}'
        return name

    def get_delay_input_output(self,  pre_group: Assembly, post_group: Assembly):
        input_name = pre_group.id + ':' + '{' + self.pre_var_name + '}'
        output_name = self.id + ':' + post_group.id + '<-' + pre_group.id + ':' + '{' + self.pre_var_name + '}'
        return input_name, output_name

    def get_target_output_name(self, output_group: Assembly):
        name = output_group.id + ':' + '{' + self.pre_var_name + '}'
        return name

    def get_V_name(self, post_group: Assembly): # target_output_name
        name = post_group.id + ':' + '{V}'
        return name
    def get_V_updated_name(self, post_group: Assembly): # target_output_name
        name = post_group.id + ':' + '{V}[updated]'
        return name
    def get_I_name(self, post_group: Assembly):
        name = post_group.id + ':' + '{I}'
        return name
    def get_Vtemp_name(self,  pre_group: Assembly, post_group: Assembly): # target_output_name
        name = post_group.id + '<-' + pre_group.id + ':' + '{Vtemp}'
        return name

    def get_I_ele_name(self, post_group: Assembly):
        name = post_group.id + ':' + '{I_ele}'
        return name

    def Chemistry_I_synapse(self):
        """
        Chemistry current synapse
        I = tauP*I + WgtSum
        """

        I = self.get_I_ele_name(self.post_assembly)
        WgtSum = self.get_target_name(self.post_assembly)
        tauP = self.post_assembly.id + ':' + '{tauP}'
        self._syn_variables[I] = 0
        self._syn_variables[WgtSum] = 0
        self._syn_tau_constant_variables[tauP] = self.tau_p

        self._syn_operations.append([I, 'var_linear', tauP, I, WgtSum])



    def Electrical_synapse(self):
        """
        Electrical synapse
        I_ele = weight *（V(l-1) - V(l)）
        """

        V_post = self.get_V_name(self.post_assembly)
        V_pre = self.get_V_name(self.pre_assembly)
        Vtemp1_post = self.get_Vtemp_name(self.pre_assembly, self.post_assembly)
        I_post = self.get_I_ele_name(self.post_assembly)
        weight = self.get_weight_name(self.pre_assembly, self.post_assembly)
        Vtemp1_pre = self.get_Vtemp_name(self.post_assembly, self.pre_assembly)
        I_pre = self.get_I_ele_name(self.pre_assembly)


        self._syn_variables[Vtemp1_post] = 0
        self._syn_variables[I_post] = 0
        self._syn_variables[Vtemp1_pre] = 0
        self._syn_variables[I_pre] = 0
        self._syn_operations.append([Vtemp1_post, 'minus', V_pre, V_post])
        self._syn_operations.append([I_post, 'var_mult', weight, Vtemp1_post + '[updated]'])
        self._syn_operations.append([Vtemp1_pre, 'minus', V_post, V_pre])
        self._syn_operations.append([I_pre, 'var_mult', weight, Vtemp1_pre + '[updated]'])




class FullConnection(Connection):

    '''
    each neuron in the first layer is connected to each neuron in the second layer.

    Args:
        pre_assembly(Assembly): The assembly which needs to be connected
        post_assembly(Assembly): The assembly which needs to connect the pre_assembly
        link_type(str): full

    Methods:
        unit_connect: define the basic connection information and add them to the connection_information.
        condition_check: check whether the pre_group.type is equal to the post_group.type, only if they are equal, return flag=Ture.

    '''



    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse', 'conv','...'), policies=[],
                 max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):

        super(FullConnection, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name, link_type=link_type,
                                             policies=policies, max_delay=max_delay, sparse_with_mask=sparse_with_mask,
                                             pre_var_name=pre_var_name, post_var_name=post_var_name, **kwargs)
        self.weight = kwargs.get('weight', None)
        self.mask = kwargs.get('mask', None)
        self.w_std = kwargs.get('w_std', 0.05)
        self.w_mean = kwargs.get('w_mean', 0.005)
        self.w_max = kwargs.get('w_max', None)
        self.w_min = kwargs.get('w_min', None)
        self.flatten_on = kwargs.get('flatten', False)
    def unit_connect(self, pre_group, post_group):
        # The number of neurons in neuron group
        pre_num = pre_group.num
        post_num = post_group.num
        link_num = pre_num * post_num
        shape = (post_num, pre_num)
        if self.weight is None:
            # Connection weight
            weight = self.w_std*np.random.randn(*shape) + self.w_mean
        else:
            assert (self.weight.shape == shape), f"The size of the given weight {self.weight.shape} does not correspond to the size of synaptic matrix {shape} "
            weight = self.weight

        # The name for backend variables
        input_name = self.get_input_name(pre_group, post_group)
        weight_name = self.get_weight_name(pre_group, post_group)
        target_name = self.get_target_name(post_group)

        # The backend variable
        # var_code = (weight_name, shape, weight, True, False, 'uniform')   # (var_name, shape, value, is_parameter, is_sparse, init)
        var_code = (weight_name, shape, weight, True, False)

        # The backend basic operation
        if self.max_delay > 0 and self.flatten_on != True:
            op_code = [target_name, 'mult_sum', input_name, weight_name]
        elif self.max_delay > 0 and self.flatten_on == True:
            raise ValueError("Conv_connectoin cannot do delay recently!")
        elif self.max_delay <= 0 and self.flatten_on == True:
            op_code = [target_name, 'reshape_mat_mult', input_name, weight_name]
        else:
            op_code = [target_name, 'mat_mult', input_name, weight_name]


        connection_information = (pre_group, post_group, link_num, var_code, op_code)
        self.unit_connections.append(connection_information)

        if self.sparse_with_mask:
            if self.mask is None:
                mask = (weight != 0)
            else:
                mask = self.mask
            mask_name = self.get_mask_name(pre_group, post_group)
            mask_var_code = (mask_name, shape, mask)
            mask_op = (weight_name, self.mask_operation, [weight_name, mask_name])
            mask_information = (mask_var_code, mask_op)
            self.mask_info.append(mask_information)
        pass

    def condition_check(self, pre_group, post_group):
        flag = False
        pre_type = pre_group.type
        post_type = post_group.type
        if pre_type == post_type:
            flag = True
        return flag
        pass

Connection.register('full',FullConnection)

class one_to_one_sparse(Connection):
    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse', 'conv','...'), policies=[],
                 max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):
        super(one_to_one_sparse, self).__init__(pre_assembly, post_assembly, name, link_type,
                                                policies, max_delay, sparse_with_mask, pre_var_name, post_var_name, **kwargs)
        self.w_std = kwargs.get('w_std', 0.05)

    def unit_connect(self, pre_group, post_group):
        pre_num = pre_group.num
        post_num = post_group.num
        try:
            assert pre_num == post_num
        except AssertionError:
            raise ValueError('One to One connection must be defined in two groups with the same size, but the pre_num %s is not equal to the post_num %s.'%(pre_num, post_num))
        link_num = pre_num * post_num
        shape = (post_num, pre_num)

        # Connection weight
        weight = self.w_std * np.eye(*shape)

        # The name for backend variables
        input_name = self.get_input_name(pre_group, post_group)
        weight_name = self.get_weight_name(pre_group, post_group)
        target_name = self.get_target_name(post_group)

        # The backend variable
        var_code = (weight_name, shape, weight, False, True)  # (var_name, shape, value, is_parameter, is_sparse)

        # The backend basic operation
        op_code = [target_name, 'sparse_mat_mult', weight_name, input_name]

        connection_information = (pre_group, post_group, link_num, var_code, op_code)
        self.unit_connections.append(connection_information)

        pass

    def condition_check(self, pre_group, post_group):
        flag = False
        pre_type = pre_group.type
        post_type = post_group.type
        if pre_type == post_type:
            flag = True
        return flag
        pass

Connection.register('one_to_one_sparse', one_to_one_sparse)

class one_to_one(Connection):
    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse', 'conv','...'), policies=[],
                 max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):
        super(one_to_one, self).__init__(pre_assembly, post_assembly, name, link_type,
                                         policies, max_delay, sparse_with_mask, pre_var_name, post_var_name, **kwargs)
        self.w_std = kwargs.get('w_std', 0.05)

    def unit_connect(self, pre_group, post_group):
        pre_num = pre_group.num
        post_num = post_group.num
        try:
            assert pre_num == post_num
        except AssertionError:
            raise ValueError('One to One connection must be defined in two groups with the same size, '
                             'but the pre_num %s is not equal to the post_num %s.'%(pre_num, post_num))

        link_num = pre_num * post_num
        shape = (post_num, pre_num)

        # Connection weight
        weight = self.w_std * np.eye(*shape)


        # The name for backend variables
        input_name = self.get_input_name(pre_group, post_group)
        weight_name = self.get_weight_name(pre_group, post_group)
        target_name = self.get_target_name(post_group)

        # The backend variable
        var_code = (weight_name, shape, weight, True, False)  # (var_name, shape, value, is_parameter, is_sparse)

        # The backend basic operation
        op_code = [target_name, 'mat_mult', input_name, weight_name]

        connection_information = (pre_group, post_group, link_num, var_code, op_code)
        self.unit_connections.append(connection_information)

        if self.sparse_with_mask:
            mask = (weight != 0)
            mask_name = self.get_mask_name(pre_group, post_group)
            mask_var_code = (mask_name, shape, mask)
            mask_op = (weight_name, self.mask_operation, [weight_name, mask_name])
            mask_information = (mask_var_code, mask_op)
            self.mask_info.append(mask_information)

        pass

    def condition_check(self, pre_group, post_group):
        flag = False
        pre_type = pre_group.type
        post_type = post_group.type
        if pre_type == post_type:
            flag = True
        return flag
        pass
Connection.register('one_to_one', one_to_one)

class conv_connect(Connection):

    '''
    do the convolution connection.

    Args:
        pre_assembly(Assembly): the assembly which needs to be connected
        post_assembly(Assembly): the assembly which needs to connect the pre_assembly
        link_type(str): Conv
    Methods:
        unit_connect: define the basic connection information and add them to the connection_information.
        condition_check: check whether the pre_group.type is equal to the post_group.type, only if they are equal, return flag=Ture.

    '''
    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse', 'conv','...'),  policies=[],  max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum',**kwargs):
        super(conv_connect, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name, link_type=link_type,
                                             policies=policies, max_delay=max_delay, sparse_with_mask=sparse_with_mask, pre_var_name=pre_var_name, post_var_name=post_var_name, **kwargs)
        self.out_channels = kwargs.get('out_channels', 4)
        self.in_channels = kwargs.get('in_channels', 1)
        self.kernel_size = kwargs.get('kernel_size', (3, 3))
        self.maxpool_on = kwargs.get('maxpool_on', True)
        self.maxpool_kernel_size = kwargs.get('maxpool_kernel_size', (2, 2))
        self.w_std = kwargs.get('w_std', 0.05)
        self.w_mean = kwargs.get('w_mean', 0.05)

        self.weight = kwargs.get('weight', None)
        self.mask = kwargs.get('mask', None)
        self.stride = kwargs.get('stride', 1)
        self.padding = kwargs.get('padding', 0)
        self.dilation = kwargs.get('dilation', 1)
        self.groups = kwargs.get('groups', 1)


    def unit_connect(self, pre_group, post_group):
        '''
        set the basic parameters, for example: link_length, connection weight, connection shape, the name for backend variables, the backend variable,the backend basic operation.

        Args:
            pre_group(Groups): the neuron group which need to be connected in the pre_assembly.
            post_group(Groups): the neuron group which need to be connected with the pre_group neuron.

        '''
        shape = (self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        if self.weight is None:
            # Connection weight
            weight = self.w_std * np.random.randn(*shape) + self.w_mean
        else:
            assert (self.weight.shape == shape), f"The size of the given weight {self.weight.shape} does not correspond to the size of synaptic matrix {shape} "
            weight = self.weight


        Hin = pre_group.shape[-2]
        Win = pre_group.shape[-1]

        Ho_conv = round((Hin + 2 * self.padding - self.kernel_size[0]) / self.stride + 1)  # Ho = (Hin + 2 * padding[0] - kernel_size[0]) / stride[0] + 1
        Wo_conv = round((Win + 2 * self.padding - self.kernel_size[1]) / self.stride + 1)  # Wo = (Win + 2 * padding[0] - kernel_size[1]) / stride[0] + 1


        if self.maxpool_on:  # 池化

            Ho = int(Ho_conv / self.maxpool_kernel_size[0])
            Wo = int(Wo_conv / self.maxpool_kernel_size[1])

        else:

            Ho = int(Ho_conv)
            Wo = int(Wo_conv)

        post_num = int(Ho * Wo * self.out_channels)

        if post_group.num == None:
            post_group.num = post_num
            post_group.shape = (self.out_channels, Ho, Wo)

        if post_group.num != None:
            if post_group.num != post_num:
                raise ValueError(
                    "The post_group num is not equal to the output num, cannot achieve the conv connection, "
                    "the output num is %d * %d * %d " % (self.out_channels, Ho, Wo))

            else:
                post_group.shape = (self.out_channels, Ho, Wo)
        link_num = post_num
        # The name for backend variables
        input_name = self.get_input_name(pre_group,post_group)
        weight_name = self.get_weight_name(pre_group, post_group)
        target_name = self.get_target_name(post_group)


        # The backend variable
        var_code_weight= (weight_name, shape, weight, True)
        stride = self.stride_name(pre_group)
        stride_value = np.array(self.stride)
        var_code_stride = (stride, stride_value.shape, stride_value)
        padding = self.padding_name(pre_group)
        padding_value = np.array(self.padding)
        var_code_padding = (padding, padding_value.shape, padding_value)
        dilation = self.dilation_name(pre_group)
        dilation_value = np.array(self.dilation)
        var_code_dilation = (dilation, dilation_value.shape, dilation_value)
        groups = self.groups_name(pre_group)
        groups_value = np.array(self.groups)
        var_code_groups = (groups, groups_value.shape, groups_value)

        maxpool_kernel_size_name = self.maxpool_kernel_size_name(pre_group)
        maxpool_kernel_size_value = np.array(self.maxpool_kernel_size)
        var_code_maxpool = (maxpool_kernel_size_name, maxpool_kernel_size_value.shape, maxpool_kernel_size_value)

        if self.maxpool_on:

            op_code1 = [target_name, 'conv_max_pool2d', input_name, weight_name, maxpool_kernel_size_name,
                        stride, padding, dilation, groups]
            connection_information = (pre_group, post_group, link_num, var_code_weight, var_code_maxpool,
                                      var_code_stride, var_code_padding, var_code_dilation, var_code_groups, op_code1)
        else:
            op_code1 = [target_name, 'conv_2d', input_name, weight_name, stride, padding, dilation, groups]
            connection_information = (pre_group, post_group, link_num, var_code_weight,
                                      var_code_stride, var_code_padding, var_code_dilation, var_code_groups, op_code1)


        self.unit_connections.append(connection_information)


    def condition_check(self, pre_group, post_group):
        '''
        check whether the pre_group.type is equal to the post_group.type, only if they are equal, return flag=Ture.

        Args:
            pre_group(Groups): the neuron group which need to be connected in the pre_assembly.
                   post_group(Groups): the neuron group which need to connect the pre_group in the post_assembly.

        Returns: flag

        '''

        flag = False
        pre_type = pre_group.type
        post_type = post_group.type
        if pre_type == post_type:
            flag = True
        return flag
Connection.register('conv',conv_connect)


class sparse_connect_sparse(Connection):

    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse', 'conv', '...'), policies=[],
                 max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):
        super(sparse_connect_sparse, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name,
                                                    link_type=link_type,
                                                    policies=policies, max_delay=max_delay,
                                                    sparse_with_mask=sparse_with_mask, pre_var_name=pre_var_name,
                                                    post_var_name=post_var_name, **kwargs)
        self.w_std = kwargs.get('w_std', 0.05)
        self.density = kwargs.get('density', 0.1)

    def unit_connect(self, pre_group, post_group):
        # The number of neurons in neuron group
        pre_num = pre_group.num
        post_num = post_group.num
        link_num = pre_num * post_num
        shape = (post_num, pre_num)

        # Connection weight
        sparse_matrix = self.w_std * sp.rand(post_num, pre_num, density=self.density, format='csr')
        weight = sparse_matrix.toarray()

        # The name for backend variables
        input_name = self.get_input_name(pre_group, post_group)
        weight_name = self.get_weight_name(pre_group, post_group)
        target_name = self.get_target_name(post_group)

        # The backend variable
        var_code = (weight_name, shape, weight, False, True)  # (var_name, shape, value, is_parameter, is_sparse)

        # The backend basic operation
        op_code = [target_name, 'sparse_mat_mult', weight_name, input_name]
        connection_information = (pre_group, post_group, link_num, var_code, op_code)
        self.unit_connections.append(connection_information)

        pass

    def condition_check(self, pre_group, post_group):
        flag = False
        pre_type = pre_group.type
        post_type = post_group.type
        if pre_type == post_type:
            flag = True
        return flag
        pass

Connection.register('sparse_connect_sparse', sparse_connect_sparse)

class sparse_connect(Connection):

    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse', 'conv','...'), policies=[],
                 max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):
        super(sparse_connect, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name, link_type=link_type,
                                             policies=policies, max_delay=max_delay, sparse_with_mask=sparse_with_mask, pre_var_name=pre_var_name, post_var_name=post_var_name, **kwargs)
        self.w_std = kwargs.get('w_std', 0.05)
        self.density = kwargs.get('density', 0.1)

    def unit_connect(self, pre_group, post_group):
        # The number of neurons in neuron group
        pre_num = pre_group.num
        post_num = post_group.num
        link_num = pre_num * post_num
        shape = (post_num, pre_num)

        # Connection weight
        sparse_matrix = self.w_std * sp.rand(post_num, pre_num, density=self.density, format='csr')
        weight = sparse_matrix.toarray()

        # The name for backend variables
        input_name = self.get_input_name(pre_group, post_group)
        weight_name = self.get_weight_name(pre_group, post_group)
        target_name = self.get_target_name(post_group)

        # The backend variable
        var_code = (weight_name, shape, weight, True, False)  # (var_name, shape, value, is_parameter, is_sparse)

        # The backend basic operation
        op_code = [target_name, 'mat_mult', input_name, weight_name]

        connection_information = (pre_group, post_group, link_num, var_code, op_code)
        self.unit_connections.append(connection_information)

        mask = (weight != 0)
        mask_name = self.get_mask_name(pre_group, post_group)
        mask_var_code = (mask_name, shape, mask)
        mask_op = (weight_name, self.mask_operation, [weight_name, mask_name])
        mask_information = (mask_var_code, mask_op)
        self.mask_info.append(mask_information)

        pass

    def condition_check(self, pre_group, post_group):
        flag = False
        pre_type = pre_group.type
        post_type = post_group.type
        if pre_type == post_type:
            flag = True
        return flag
        pass
Connection.register('sparse_connect', sparse_connect)

class random_connect_sparse(Connection):

    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse', 'conv','...'), policies=[],
                 max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):
        super(random_connect_sparse, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name, link_type=link_type,
                                             policies=policies, max_delay=max_delay, sparse_with_mask=sparse_with_mask, pre_var_name=pre_var_name, post_var_name=post_var_name, **kwargs)
        self.probability = kwargs.get('probability', 0.1)

    def unit_connect(self, pre_group, post_group):
        # The number of neurons in neuron group
        pre_num = pre_group.num
        post_num = post_group.num
        link_num = pre_num * post_num
        shape = (post_num, pre_num)

        # Link_parameters

        prob_weight = np.random.rand(*shape)
        diag_index = np.arange(min([pre_num, post_num]))
        prob_weight[diag_index, diag_index] = 1
        index = (prob_weight < self.probability)

        weight = np.zeros(shape)
        weight[index] = prob_weight[index]

        # The name for backend variables
        input_name = self.get_input_name(pre_group, post_group)
        weight_name = self.get_weight_name(pre_group, post_group)
        target_name = self.get_target_name(post_group)

        # The backend variable
        var_code = (weight_name, shape, weight, False, True)   # (var_name, shape, value, is_parameter, is_sparse)

        # The backend basic operation
        op_code = [target_name, 'sparse_mat_mult', weight_name, input_name]

        connection_information = (pre_group, post_group, link_num, var_code, op_code)
        self.unit_connections.append(connection_information)
        pass

    def condition_check(self, pre_group, post_group):
        flag = False
        pre_type = pre_group.type
        post_type = post_group.type
        if pre_type == post_type:
            flag = True
        return flag
        pass
Connection.register('random_connect_sparse', random_connect_sparse)

class random_connect(Connection):

    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse', 'conv', '...'), policies=[],
                 max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):
        super(random_connect, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name, link_type=link_type,
                                             policies=policies, max_delay=max_delay, sparse_with_mask=sparse_with_mask, pre_var_name=pre_var_name, post_var_name=post_var_name, **kwargs)
        self.probability = kwargs.get('probability', 0.1)

    def unit_connect(self, pre_group, post_group):
        # The number of neurons in neuron group
        pre_num = pre_group.num
        post_num = post_group.num
        link_num = pre_num * post_num
        shape = (post_num, pre_num)

        prob_weight = np.random.rand(*shape)
        diag_index = np.arange(min([pre_num, post_num]))
        prob_weight[diag_index, diag_index] = 1
        index = (prob_weight < self.probability)

        weight = np.zeros(shape)
        weight[index] = prob_weight[index]

        # The name for backend variables
        input_name = self.get_input_name(pre_group, post_group)
        weight_name = self.get_weight_name(pre_group, post_group)
        target_name = self.get_target_name(post_group)

        # The backend variable
        var_code = (weight_name, shape, weight, True, False)    # (var_name, shape, value, is_parameter, is_sparse)

        # The backend basic operation
        op_code = [target_name, 'mat_mult', input_name, weight_name]

        connection_information = (pre_group, post_group, link_num, var_code, op_code)
        self.unit_connections.append(connection_information)

        mask = (weight != 0)
        mask_name = self.get_mask_name(pre_group, post_group)
        mask_var_code = (mask_name, shape, mask)
        mask_op = (weight_name, self.mask_operation, [weight_name, mask_name])
        mask_information = (mask_var_code, mask_op)
        self.mask_info.append(mask_information)
        pass

    def condition_check(self, pre_group, post_group):
        flag = False
        pre_type = pre_group.type
        post_type = post_group.type
        if pre_type == post_type:
            flag = True
        return flag
        pass
Connection.register('random_connect', random_connect)


class DistDepd_connect(Connection):

    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse', 'conv','...'), policies=[],
                 max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):

        super(DistDepd_connect, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name, link_type=link_type,
                                             policies=policies, max_delay=max_delay, sparse_with_mask=sparse_with_mask,
                                             pre_var_name=pre_var_name, post_var_name=post_var_name, **kwargs)
        self.distance_weight_function = kwargs.get('distance_weight_function', None)
        self.zero_self = kwargs.get('zero_self', False)
        if self.distance_weight_function is None:
            self.distance_weight_function = self.default_dist_weight_function
        self.dist_function = kwargs.get('dist_function', 'euclidean')
        if self.dist_function == 'euclidean':
            self.dist_function = self.euclidean_dist_function
        elif self.dist_function == 'circular':
            self.dist_function = self.circular_dist_function

        self.dist_a = kwargs.get('dist_a', 0.2)
        self.dist_b = kwargs.get('dist_b', 0.4)
        self.w_amp = kwargs.get('w_amp', -0.1)
        self.pos_range = kwargs.get('pos_range', 1.0)

    def unit_connect(self, pre_group, post_group):
        # The number of neurons in neuron group
        pre_num = pre_group.num
        post_num = post_group.num
        assert len(pre_group.position) > 0
        assert len(post_group.position) > 0
        link_num = pre_num * post_num
        shape = (post_num, pre_num)
        weight = np.zeros(shape)
        post_pos = np.expand_dims(post_group.position,  axis=1)
        pre_pos = np.expand_dims(pre_group.position, axis=0)
        weight = self.distance_weight_function(self.dist_function(pre_pos, post_pos))
        # from matplotlib import pyplot as plt
        # plt.imshow(weight)
        # plt.show()

        # The name for backend variables
        input_name = self.get_input_name(pre_group, post_group)
        weight_name = self.get_weight_name(pre_group, post_group)
        target_name = self.get_target_name(post_group)

        # The backend variable
        var_code = (weight_name, shape, weight, True, False) # (var_name, shape, value, is_parameter, is_sparse, init)
        op_code = [target_name, 'mat_mult', input_name, weight_name]
        connection_information = (pre_group, post_group, link_num, var_code, op_code)
        self.unit_connections.append(connection_information)

    def circular_dist_function(self, pre_pos, post_pos):
        if not isinstance(pre_pos, torch.Tensor):
            pre_pos = torch.tensor(pre_pos)
        if not  isinstance(post_pos, torch.Tensor):
            post_pos = torch.tensor(post_pos)

        z = torch.maximum(pre_pos, post_pos)
        k = torch.minimum(pre_pos, post_pos)
        dist = torch.minimum(z - k, self.pos_range + k - z)
        dist = torch.norm(dist, p=2, dim=-1)
        return dist

    def euclidean_dist_function(self, pre_pos, post_pos):
        if isinstance(pre_pos, torch.Tensor) and isinstance(post_pos, torch.Tensor):
            diff = pre_pos - post_pos
        else:
            diff = torch.tensor(pre_pos-post_pos)
        dist = torch.norm(diff, p=2, dim=-1)
        return dist

    def default_dist_weight_function(self, dist):
        weights = self.w_amp * (torch.exp(-dist / self.dist_a)/self.dist_a - 0.5*torch.exp(-dist / self.dist_b)/self.dist_b)
        if self.zero_self:
            weights = weights * (dist!=0).float()
        # import matplotlib.pyplot as plt
        # plt.imshow(weights, aspect='auto')
        # plt.show()
        return weights



Connection.register('dist_depd', DistDepd_connect)

class reconnect(Connection):
    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse', 'conv', '...'), policies=[],
                 max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):
        super(reconnect, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name, link_type=link_type,
                                             policies=policies, max_delay=max_delay, sparse_with_mask=sparse_with_mask, pre_var_name=pre_var_name, post_var_name=post_var_name, **kwargs)
    def unit_connect(self, pre_group, post_group):
        pass

    def condition_check(self, pre_group, post_group):
        pass