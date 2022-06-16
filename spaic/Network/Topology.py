# -*- coding: utf-8 -*-
"""
Created on 2022/1/18
@project: SPAIC
@filename: Topology
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description: 
"""
from ..Network.BaseModule import BaseModule
from ..Network.Assembly import Assembly
# from ..Network.Synapse import Flatten_synapse
from ..Network.BaseModule import VariableAgent
from ..Backend.Backend import Backend
from abc import abstractmethod
from typing import List
import numpy as np
from abc import ABC

class Projection(BaseModule):
    '''
    Class for projection between assemblies, which contain multiple connections between sub neurongroups
    '''
    _class_label = '<prj>'
    _con_count = 0

    def __init__(self, pre: Assembly, post: Assembly,  policies=[], link_type=None, ConnectionParameters=None, name=None):
        super(Projection, self).__init__()
        assert isinstance(pre, Assembly)
        assert isinstance(post, Assembly)
        self.name = name
        self.super = None
        self.abstract_level = 0
        self._backend: Backend = None
        self.pre_assembly = pre
        self.post_assembly = post
        self._connections = dict()
        self._projections = dict()

        if (pre.id is not None) and (post.id is not None):
            self.key = pre.id + "->" + post.id
        else:
            self.key = 'default' + str(Projection._con_count)
        Projection._con_count += 1
        self._leaf_connections = dict()

        if isinstance(ConnectionParameters, dict):
            self.ConnectionParameters = ConnectionParameters
        else:
            self.ConnectionParameters = dict()
        if pre._is_terminal and post._is_terminal:
            self.is_unit = True
        else:
            self.is_unit = False

        self._policies = policies
        if self._policies:
            assert link_type is not None
        if isinstance(self._policies, ConnectPolicy):
            self._policies = [self._policies]
        self.link_type = link_type
        self._supers = []

    def __setattr__(self, name, value):
        super(Projection, self).__setattr__(name, value)
        if name == 'super':
            return
        if isinstance(value, Connection):
            if self._backend: self._backend.builded = False
            self._connections[name] = value
            self._leaf_connections[value.key] = value
            value.set_name(name)
            value.add_super(self)
        elif isinstance(value, Projection):
            if self._backend: self._backend.builded = False
            self._projections[name] = value
            self._leaf_connections[value.key] = value
            value.set_name(name)
            value.add_super(self)


    def homologous(self, other):
        if (other.pre_assembly is self.pre_assembly) and (other.post_assembly is self.post_assembly):
            return True
        else:
            return False



    def is_empty(self):
        if self.is_unit:
            return False
        elif len(self._connections) + len(self._projections) == 0:
            return True
        else:
            return False

    def __and__(self, other):
        if not self.homologous(other):
            raise ValueError("can't do & operation for nonhomologous connections")
        if self.is_empty():
            new_connect = self
        elif other.is_empty():
            new_connect = other
        else:
            new_connect = Projection(self.pre_assembly, self.pre_assembly)
            # unit connections
            key1 = set(self._connections.keys())
            key2 = set(other._connections.keys())
            unite_keys = key1.intersection(key2)
            for key in unite_keys:
                new_connect._connections[key] = self._connections[key]
                new_connect._connections[key].super = new_connect

            #assb connections
            key1 = set(self._projections.keys())
            key2 = set(other._projections.keys())
            unite_keys = key1.intersection(key2)
            for key in unite_keys:
                new_connect._projections[key] = self._projections[key] & other._projections[key]
                new_connect._projections[key].super = new_connect

            # leaf_connections
            key1 = set(self._leaf_connections.keys())
            key2 = set(other._leaf_connections.keys())
            unite_keys = key1.intersection(key2)
            for key in unite_keys:
                new_connect._leaf_connections[key] = self._leaf_connections[key] & other._leaf_connections[key]

        return new_connect

    def add_connection(self, con):
        assert isinstance(con, Projection)
        con.super = self
        key = con.key
        if not (con.pre_assembly in self.pre_assembly and con.post_assembly in self.post_assembly):
            raise ValueError("the sub connection is not belong to this connection group (pre and post is not a member of the connected Assemblies)")

        con.set_name(None)
        if con.is_unit:
            self._connections[key] = con
        else:
            self._projections[key] = con
    
    def del_connection(self):
        if self.super is not None:
            if self.is_unit:
                self.super._connections.pop(self.key)
            else:
                self.super._projections.pop(self.key)
            if self.key in self.super._leaf_connections:
                self.super._leaf_connections.pop(self.key)

            top_leaf = self.top._leaf_connections
            if self.key in top_leaf:
                top_leaf.pop(self.key)
            if self.super.is_empty():
                self.super.del_connection()
        else:
            self.is_unit = False
            
    def expand_connection(self, to_level=-1):
        assert isinstance(to_level, int), ValueError("level is not int")
        if not bool(self._leaf_connections):
            self._leaf_connections[self.key] = self
        if self.is_unit:
            return self._leaf_connections.values()
        else:
            new_leaf_connections = dict()
            assb_connections = self._leaf_connections.values()
            while(len(assb_connections)>0):
                if to_level >= 0 and to_level<=self.abstract_level:
                    break
                self.abstract_level += 1
                new_assb_connections = []
                for con in assb_connections:
                    if con.is_unit:
                        new_leaf_connections[con.key] = con
                    else:
                        pre_groups = con.pre_assembly.get_groups(recursive=False)
                        post_groups = con.post_assembly.get_groups(recursive=False)
                        for pre in pre_groups:
                            for post in post_groups:
                                new_con = Projection(pre, post)
                                con.add_connection(new_con)
                                if new_con.is_unit:
                                    new_leaf_connections[new_con.key] = new_con
                                else:
                                    new_assb_connections.append(new_con)

                assb_connections = new_assb_connections

            for con in assb_connections:
                #对于超过abstract_level的connection，都认为是leaf_connection
                new_leaf_connections[con.key] = con
            self._leaf_connections = new_leaf_connections
            return list(new_leaf_connections.values())

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
        assert isinstance(assembly, Assembly) or isinstance(assembly, Projection), "the super is not Assembly or Projection"
        self._supers.append(assembly)

    def del_super(self, assembly):
        assert assembly in self._supers, "the assembly is not in supers"
        self._supers.remove(assembly)

    def get_connections(self, recursive=True):
        """
            Get the Connections in this assembly
        Args:
            recursive(bool): flag to decide if member connections of the member assemblies should be returned.

        Returns:
            List of Connections
        """
        if not recursive:
            return self._connections.values()
        else:
            connections = list(self._connections.values())
            for proj in self.sub_projections:
                connections.extend(proj.get_connections(recursive=False))
            return connections

    @property
    def top(self):
        if self.super is None:
            return self
        else:
            return self.super.top

    @property
    def sub_connections(self):
        if self.is_unit:
            raise ValueError("no sub_connections for unit connections")
        else:
            sub_connections = []
            sub_connections.extend(self._connections.values())
            return sub_connections

    @property
    def sub_projections(self):
        if self.is_unit:
            raise ValueError("no sub assb_connections for unit connections")
        else:
            return list(self._projections.values())


    @property
    def all_connections(self):
        if self.is_unit:
            return [self]
        else:
            unit_connections = []
            unit_connections.extend(self._connections.values())
            for con in self._projections.values():
                unit_connections.extend(con.all_connections)
            return unit_connections

    @property
    def all_projections(self):
        if self.is_unit:
            raise ValueError("no sub_connections for unit connections")
        else:
            assb_connections = []
            assb_connections.extend(self._projections.values())
            for con in self._projections.values():
                assb_connections.extend(con.all_projections)
            return assb_connections
    @property
    def leaf_connections(self):
        return list(self._leaf_connections.values())


    def get_str(self, level):

        # TODO: add str of subconnections
        level_space = "" + '-' * level
        repr_str = level_space + "|name:{}, type:{}, ".format(self.name, type(self).__name__)
        repr_str += "pre_assembly:{}, ".format(self.pre_assembly.name)
        repr_str += "post_assembly:{}\n ".format(self.post_assembly.name)
        level += 1
        for c in self._connections.values():
            repr_str += c.get_str(level)
        return repr_str

    def build(self, backend=None):
        self._backend = backend

        # clear previous builded connections
        if backend.builded == True:
            for super_asb in self._supers:
                for con in self._connections:
                    super_asb.del_connection(con)

        connection_inforamtion = self
        connection_inforamtion.expand_connection()
        for p in self._policies:
            connection_inforamtion = connection_inforamtion & p.generate_connection(self.pre_assembly, self.post_assembly)

        self._connections = connection_inforamtion._connections
        self._projections = connection_inforamtion._projections
        self._leaf_connections = connection_inforamtion._leaf_connections
        new_connections = dict()
        for key, con in connection_inforamtion._connections.items():
            assert con.is_unit
            if con.link_type is not None:
                link_type = con.link_type
            else:
                assert self.link_type is not None
                link_type = self.link_type
            new_connections[key] = Connection(con.pre_assembly, con.post_assembly, link_type=link_type, **self.ConnectionParameters)
            new_connections[key].add_super(self)
        self._connections = new_connections


class SynapseModel(ABC):
    '''
    op -> (return_name, operation_name, input_name1, input_name2...)
    '''

    #: A dictionary mapping synapse model names to `Model` objects
    synapse_models = dict()

    def __init__(self, conn, **kwargs):
        super(SynapseModel, self).__init__()
        self.name = 'none'
        self._syn_operations = []
        self._syn_variables = dict()
        self._syn_constant_variables = dict()
        updated_input = conn.updated_input
        if updated_input:
            self.input_name = conn.pre_var_name + '[input][updated]'
        else:
            self.input_name = conn.pre_var_name + '[input]'

    @staticmethod
    def register(name, model):
        '''
        Register a synapse model. Registered synapse models can be referred to
        # via their name.
        Parameters
        ----------
        name : str
            A short name for the state updater (e.g. `'basic'`)
        model : `SynapseModel`
            The synapse model object, e.g. an `Basic_synapse`.
        '''

        # only deal with lower case names -- we don't want to have 'LIF' and
        # 'lif', for example
        name = name.lower()
        if name in SynapseModel.synapse_models:
            raise ValueError(('A synapse_model with the name "%s" has already been registered') % name)

        if not issubclass(model, SynapseModel):
            raise ValueError(('Given model of type %s does not seem to be a valid SynapseModel.' % str(type(model))))

        SynapseModel.synapse_models[name] = model
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
        if model_name not in SynapseModel.synapse_models:
            raise ValueError(('Given model name is not in the model list'))
        else:
            return SynapseModel.synapse_models[model_name]



class Connection(Projection):
    '''
    Base class for all kinds of connections, including full connection, sparse connection, conv connection,....
    Ten connection methods are provided, as shown below (key: class):
        'full', FullConnection
        'one_to_one_sparse', one_to_one_sparse
        'one_to_one', one_to_one_mask
        'conv', conv_connect
        'sparse_connect_sparse', sparse_connect_sparse
        'sparse_connect', sparse_connect_mask
        'random_connect_sparse', random_connect_sparse
        'random_connect', random_connect_mask
    Args:
        pre_assembly(Assembly): the assembly which needs to be connected.
        post_assembly(Assembly): the assembly which needs to connect the pre_assembly.
        link_type(str): the type for connection: full, sparse, conv...

    Attributes:
        pre_group(groups): the neuron group which need to be connected in the pre_assembly.
        post_group(groups): the neuron group which need to connect with pre_group neuron.
        _var_names(list): a list contain variable names.

    Methods:
        __new__: before build a new connection, do some checks.
        get_var_names: get variable names.
        register: register a connection class.
        build: add the connection variable, variable name and opperation to the backend.
        get_str:
        condition_check: check whether the pre_group.type is equal to the post_group.type, only if they are equal, return flag=Ture.
        connect: connect the preg with postg.
        get_weight_name: give a name for each connection weight.
        get_post_name: give a name for each post group.
        get_input_name: give a name for each input group.

    Examples:
        when building the network:
        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full')

        '''

    _connection_subclasses = dict()
    _class_label = '<con>'

    def __init__(self, pre_assembly: Assembly, post_assembly: Assembly, name=None, link_type=('full', 'sparse_connect', 'conv', '...'),
                 syn_type=['basic_synapse'], max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn', **kwargs):

        super(Connection, self).__init__(pre_assembly, post_assembly)
        # assert pre_assembly._is_terminal
        # assert post_assembly._is_terminal

        self.pre_assembly = pre_assembly  #.get_assemblies()
        self.post_assembly = post_assembly  #.get_assemblies()
        self.pre_var_name = pre_var_name
        self.post_var_name = post_var_name

        self.pre_num = pre_assembly.num
        self.post_num = post_assembly.num
        self.link_num = self.pre_num * self.post_num
        self.shape = (self.post_num, self.pre_num)

        self.pre_groups = None   # pre_assembly.get_groups()
        self.post_groups = None  # post_assembly.get_groups()
        self.pre_assemblies = None
        self.post_assemblies = None
        self.connection_information = None
        self.mask_info: List[(tuple, tuple)] = list()  # (var_code, op_code)

        self.link_type = link_type
        self.max_delay = max_delay
        self.sparse_with_mask = sparse_with_mask
        self._var_names = list()
        self._var_dict = dict()
        self._supers = list()
        self._link_var_codes = list()
        # self._link_op_codes = list()

        self.parameters = kwargs
        self.init = kwargs.get('init', None)
        self.init_param = kwargs.get('init_param', None)
        self.bias = kwargs.get('bias', False)
        self.updated_input = kwargs.get('updated_input', True)

        self.w_max = None
        self.w_min = None

        self.set_name(name)
        self.running_var = None
        self.running_mean = None
        self.decay = 0.9
        self._variables = dict()
        self._constant_variables = dict()

        self.flatten = kwargs.get('flatten', False)

        if len(syn_type) == 0:
            raise ValueError('Please set the param syn_type, you can set it as \'basic_synapse\'')

        if isinstance(syn_type, list):
            self.synapse_type = syn_type
        else:
            self.synapse_type = [syn_type]
        self.synapse_class = []
        self.synapse_name = []

        for i in range(len(self.synapse_type)):
            if isinstance(self.synapse_type[i], str):
                self.synapse_class.append(SynapseModel.apply_model(self.synapse_type[i]))
                self.synapse_name.append(self.synapse_type[i])  # self.neuron_model -> self.model_name
                self.synapse = []
            else:
                raise ValueError("only support set synapse model with string")

        # construct unit connection information by policies,
        # construct in __init__ is potentially bad, as network structure may change before build. should add new function
        # self.connection_inforamtion = ConnectInformation(self.pre_assembly, self.post_assembly)
        # self.connection_inforamtion.expand_connection()
        # for p in self._policies:
        #     self.connection_inforamtion = self.connection_inforamtion & p.generate_connection(self.pre_assembly, self.post_assembly)

    def norm_hook(self, grad):
        import torch
        if self.running_var is None:
            self.running_var = torch.norm(grad, dim=1,keepdim=True)
            self.running_mean = torch.mean(grad, dim=1,keepdim=True)
        else:
            self.running_var = self.decay * self.running_var + (1 - self.decay) * torch.norm(grad, dim=0)
            self.running_mean = self.decay * self.running_mean + (1 - self.decay) * torch.mean(grad, dim=0)
        return (grad - self.running_mean) / (1.0e-10 + self.running_var)

    def __new__(cls, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse_connect', 'conv','...'),
                syn_type=['basic_synapse'], max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn', **kwargs):
        if cls is not Connection:
            return super().__new__(cls)

        if link_type in cls._connection_subclasses:
            return cls._connection_subclasses[link_type](pre_assembly, post_assembly, name, link_type, syn_type, max_delay, sparse_with_mask, pre_var_name, post_var_name, **kwargs)

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
        assert isinstance(assembly, Assembly) or isinstance(assembly, Projection), "the super is not Assembly"
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
            # pre_num = pre_group.num
            # post_num = post_group.num
            # shape = (post_num, pre_num)
            delay_input_name, delay_output_name = self.get_delay_input_output(pre_group, post_group)

            # add delay container
            delay_queue = self._backend.add_delay(delay_input_name, self.max_delay)
            delay_name = self.get_link_name(pre_group, post_group, 'delay')
            # ONLY FOR TEST  ===============================
            ini_delay = self.max_delay*np.random.rand(*self.shape)
            # ==============================================
            self.variable_to_backend(delay_name,  self.shape, ini_delay, True)


            # add delay index
            self._backend.register_standalone(delay_output_name, delay_queue.select, [delay_name])

            # add inital to transform initial delay_output
            self._backend.register_initial(delay_output_name, delay_queue.transform_delay_output, [delay_input_name, delay_name])

        else:
            return


    def clamp_weight(self, weight):

        if (self.w_max is not None) and (self.w_min is not None):
            self._backend.clamp_(weight, self.w_min, self.w_max)
        elif self.w_max is not None:
            self._backend.clamp_max_(weight, self.w_max)
        elif self.w_min is not None:
            self._backend.clamp_min_(weight, self.w_min)

    def _add_label(self, var_name: str):
        if '[pre]' in var_name:
            var_name = var_name.replace("[pre]", "")
            var_name = self.get_group_name(self.pre_assembly, var_name)
        elif '[link]' in var_name:
            var_name = var_name.replace("[link]", "")
            var_name = self.get_link_name(self.pre_assembly, self.post_assembly, var_name)
        elif '[post]' in var_name:
            var_name = var_name.replace("[post]", "")
            var_name = self.get_group_name(self.post_assembly, var_name)
        elif '[input]' in var_name:
            var_name = self.get_input_name(self.pre_assembly, self.post_assembly)
        else:
            var_name = self.get_name(var_name)

        return var_name

    def add_conn_label(self, var_name: str):
        if isinstance(var_name, str):
            if '[updated]' in var_name:
                var_name = var_name.replace("[updated]", "")
                name = self._add_label(var_name)
                return name + '[updated]'
            else:
                name = self._add_label(var_name)
                return name

        elif isinstance(var_name, list) or isinstance(var_name, tuple):
            var_names = []
            for name in var_name:
                if isinstance(name, str):
                    if '[updated]' in name:
                        name = name.replace("[updated]", "")
                        name = self._add_label(name)
                        name = name + '[updated]'
                    else:
                        name = self._add_label(name)
                    var_names.append(name)
            return var_names


    def decode_syn_op(self, syn_ops, synapse_name, op_len):
        for i in range(1, op_len):
            pre_op_return_name = syn_ops[i-1][0]
            post_op_first_input = syn_ops[i][2]
            if '[updated]' in post_op_first_input:
                post_op_first_input = post_op_first_input.replace('[updated]', '')
            if post_op_first_input == pre_op_return_name:
                temp_name = synapse_name[i-1]+'_' + str(i-1)
                syn_ops[i-1][0] = temp_name
                syn_ops[i][2] = temp_name
        return syn_ops

    def build(self, backend):
        '''
        add the connection variable, variable name and opperation to the backend.
        '''

        self._backend = backend
        # Add weight

        self.set_delay(self.pre_assembly, self.post_assembly)

        for (key, value) in self._variables.items():
            var_name = self.add_conn_label(key)
            if isinstance(value, np.ndarray):
                if value.size > 1:
                    shape = value.shape
                else:
                    shape = ()
            elif hasattr(value, '__iter__'):
                value = np.array(value)
                if value.size > 1:
                    shape = value.shape
                else:
                    shape = ()
            else:
                shape = ()

            if 'weight' in key:
                self.variable_to_backend(name=var_name, shape=self.shape, value=value, is_parameter=self.is_parameter,
                                         is_sparse=self.is_sparse, init=self.init, init_param=self.init_param)  # (var_name, shape, value, is_parameter, is_sparse, init)
            elif 'bias' in key:
                self.variable_to_backend(name=var_name, shape=value.shape, value=value, is_parameter=self.is_parameter)
            else:
                self.variable_to_backend(var_name, shape, value=value)
            # del[self._variables[key]]
            # self._var_names.append(var_name)
            # self._var_dict[var_name] = VariableAgent(backend, var_name)

        for (key, value) in self._constant_variables.items():
            var_name = self.add_conn_label(key)
            self.variable_to_backend(name=var_name, shape=None, value=value, is_constant=True)

        weight_name = self.get_link_name(self.pre_assembly, self.post_assembly, 'weight')
        if self.sparse_with_mask:
            mask = (self.weight != 0)
            mask_name = self.get_link_name(self.pre_assembly, self.post_assembly, 'mask')
            self.variable_to_backend(mask_name, self.shape, mask)
            backend.register_initial(weight_name, self.mask_operation, [weight_name, mask_name])

        if (self.w_min is not None) or (self.w_max is not None):
            backend.register_initial(None, self.clamp_weight, [weight_name])

        syn_ops = []
        for i in range(len(self.synapse_class)):
            if self.synapse_class[i] is not None:
                self.synapse.append(self.synapse_class[i](self))

            for (key, value) in self.synapse[i]._syn_variables.items():
                key = self.add_conn_label(key)
                self.variable_to_backend(key, value.shape, value=value)

            for (key, value) in self.synapse[i]._syn_constant_variables.items():
                key = self.add_conn_label(key)
                self.variable_to_backend(key, shape=None, value=value, is_constant=True)

            for op in self.synapse[i]._syn_operations:
                syn_ops.append(op)

        syn_ops = self.decode_syn_op(syn_ops, self.synapse_name, len(syn_ops))


        for sop in syn_ops:
            addcode_op = []
            for ind, name in enumerate(sop):
                if ind != 1:
                    addcode_op.append(self.add_conn_label(sop[ind]))
                else:
                    addcode_op.append(sop[ind])
            backend.add_operation(addcode_op)


    @abstractmethod
    def condition_check(self, pre_group, post_group):
        NotImplemented()


    def mask_operation(self, weight, mask):
        return weight*mask

    def get_name(self, suffix_name: str):
        '''

        Args:
            pre_group(Assembly): the neuron group which needs to be connected
            post_group(Assembly): the neuron group which needs to connect with the pre_group
            suffix_name(str): represents the name of the object you want to retrieve, such as 'weight'
        Returns:
            name(str)
        '''

        name = self.id + ':' + '{' + suffix_name + '}'
        return name

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

    def get_group_name(self, group: Assembly, suffix_name: str):
        '''
        Args:
            group(Assembly): the neuron group which needs to be connected
            suffix_name(str): represents the name of the object you want to retrieve, such as 'O'

        Returns:
            name(str)
        '''
        name = group.id + ':' + '{' + suffix_name + '}'
        return name

    # def get_post_name(self, post_group: Assembly, suffix_name: str):
    #     '''
    #     Args:
    #         post_group(Assembly): the neuron group which needs to connect with the pre_group
    #         suffix_name(str): represents the name of the object you want to retrieve, such as 'maxpool_kernel_size'
    #
    #     Returns:
    #         name(str)
    #     '''
    #     name = post_group.id + ':' + '{' + suffix_name + '}'
    #     return name

    # def get_mask_name(self, pre_group: Assembly, post_group: Assembly):
    #     name = self.id + ':' + post_group.id + '<-' + pre_group.id + ':' + '{mask}'
    #     return name
    #
    # def get_delay_name(self, pre_group: Assembly, post_group: Assembly):
    #     name = self.id + ':' + post_group.id + '<-' + pre_group.id + ':' + '{delay}'
    #     return name

    # 这两个特例应该要留着
    def get_delay_input_output(self,  pre_group: Assembly, post_group: Assembly):
        input_name = pre_group.id + ':' + '{' + self.pre_var_name + '}'
        output_name = self.id + ':' + post_group.id + '<-' + pre_group.id + ':' + '{' + self.pre_var_name + '}'
        return input_name, output_name

    def get_input_name(self, pre_group: Assembly, post_group: Assembly):
        '''
        Give a name for input group's output spikes,  the name consists of two parts: pre_group.id + ':' + '{0}
        Args:
            pre_group(Assembly): The neuron group which need to connect with post_group neuron.
            post_group(Assembly): the neuron group which needs to connect with the pre_group
        Returns:
            name(str)
        '''
        if self.max_delay > 0:
            name = self.id + ':' + post_group.id + '<-' + pre_group.id + ':' + '{' + self.pre_var_name + '}'
        else:
            name = pre_group.id + ':' + '{' + self.pre_var_name + '}'
        return name

    def get_target_output_name(self, output_group: Assembly):
        name = output_group.id + ':' + '{' + self.pre_var_name + '}'
        return name

    # def Chemistry_Isyn(self):
    #     """
    #     Chemistry current synapse
    #     I = tauP*I + WgtSum
    #     """
    #     I = self.get_post_name(self.post_assembly, 'I')
    #     WgtSum = self.get_post_name(self.post_assembly, 'WgtSum')
    #     tauP = self.get_post_name(self.post_assembly, 'tauP')
    #     # I = self.get_I_ele_name(self.post_assembly)
    #     # WgtSum = self.get_target_name(self.post_assembly)
    #     # tauP = self.post_assembly.id + ':' + '{tauP}'
    #     self._syn_variables[I] = 0
    #     self._syn_variables[WgtSum] = 0
    #     self._syn_tau_constant_variables[tauP] = self.tau_p
    #
    #     self._syn_operations.append([I, 'var_linear', tauP, I, WgtSum])

    # def Electrical_synapse(self):
    #     """
    #     Electrical synapse
    #     WgtSum = weight *（V(l-1) - V(l)）
    #     """
    #     V_post = self.get_post_name(self.post_assembly, 'V')
    #     V_pre = self.get_pre_name(self.post_assembly, 'V')
    #     Vtemp1_post = self.get_link_name(self.pre_assembly, self.post_assembly, 'Vtemp')
    #     I_post = self.get_post_name(self.post_assembly, 'I')
    #     weight = self.get_link_name(self.pre_assembly, self.post_assembly, 'weight')
    #     Vtemp1_pre = self.get_link_name(self.post_assembly, self.pre_assembly, 'Vtemp')
    #     I_pre = self.get_pre_name(self.post_assembly, 'I_ele')
    #     # V_post = self.get_V_name(self.post_assembly)
    #     # V_pre = self.get_V_name(self.pre_assembly)
    #     # Vtemp1_post = self.get_Vtemp_name(self.pre_assembly, self.post_assembly)
    #     # I_post = self.get_I_ele_name(self.post_assembly)
    #     # weight = self.get_weight_name(self.pre_assembly, self.post_assembly)
    #     # Vtemp1_pre = self.get_Vtemp_name(self.post_assembly, self.pre_assembly)
    #     # I_pre = self.get_I_ele_name(self.pre_assembly)
    #
    #     self._syn_variables[Vtemp1_post] = 0
    #     self._syn_variables[I_post] = 0
    #     self._syn_variables[Vtemp1_pre] = 0
    #     self._syn_variables[I_pre] = 0
    #     self._syn_operations.append([Vtemp1_post, 'minus', V_pre, V_post])
    #     self._syn_operations.append([I_post, 'var_mult', weight, Vtemp1_post + '[updated]'])
    #     self._syn_operations.append([Vtemp1_pre, 'minus', V_post, V_pre])
    #     self._syn_operations.append([I_pre, 'var_mult', weight, Vtemp1_pre + '[updated]'])


class ConnectPolicy():

    def __init__(self, level=-1):
        super(ConnectPolicy, self).__init__()
        self.level = level
        self.policies = [self, ]

    @abstractmethod
    def checked_connection(self, new_connection: Projection):
        NotImplementedError()

    def extend_policy(self, policies):
        self.policies.extend(policies)

    def __and__(self, other):
        self.extend_policy(other.policies)
        return self


    def generate_connection(self, pre: Assembly, post: Assembly):

        candidates = []
        for p in self.policies:
            candidates.append(p.checked_connection(Projection(pre, post)))
        res = candidates[-1]
        candidates.pop()
        for con in candidates:
            res = res & con
        return res
