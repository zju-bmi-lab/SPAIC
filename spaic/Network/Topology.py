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
# from .Synapse import Flatten_synapse
from ..IO.Initializer import BaseInitializer
from ..Network.BaseModule import VariableAgent
from ..Network.Operator import Op
from ..Backend.Backend import Backend
from abc import abstractmethod
from typing import List
import numpy as np
from abc import ABC
# from ..utils.memory import get_cpu_mem, get_object_size

class Projection(BaseModule):
    '''
    Class for projection between assemblies, which contain multiple connections between sub neurongroups
    '''
    _class_label = '<prj>'
    _con_count = 0

    def __init__(self, pre: Assembly, post: Assembly, policies=[], link_type=None, ConnectionParameters=None,
                 name=None):
        super(Projection, self).__init__()
        assert isinstance(pre, Assembly)
        assert isinstance(post, Assembly)
        self.name = name
        self.super = None
        self.abstract_level = 0
        self._backend: Backend = None
        self.pre = pre
        self.post = post
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
            # self._leaf_connections[value.key] = value
            value.set_name(name)
            value.add_super(self)
        elif isinstance(value, Projection):
            if self._backend: self._backend.builded = False
            self._projections[name] = value
            # self._leaf_connections[value.key] = value
            value.set_name(name)
            value.add_super(self)

    def homologous(self, other):
        if (other.pre is self.pre) and (other.post is self.post):
            return True
        else:
            return False

    def is_empty(self):
        if self.is_unit:
            return False
        elif len(self._connections) + len(self._projections) + len(self._leaf_connections) == 0:
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
            new_connect = Projection(self.pre, self.post)
            # unit connections
            key1 = set(self._connections.keys())
            key2 = set(other._connections.keys())
            unite_keys = key1.intersection(key2)
            for key in unite_keys:
                new_connect._connections[key] = self._connections[key]
                new_connect._connections[key].super = new_connect

            # assb connections
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

    def add_connection(self, con, name=None):
        assert isinstance(con, Projection)
        con.super = self
        key = con.key
        if not (con.pre in self.pre and con.post in self.post):
            raise ValueError(
                "the sub connection is not belong to this connection group (pre and post is not a member of the connected Assemblies)")

        con.set_name(name)
        if con.is_unit:
            self.__setattr__(name, con)
            # self._connections[key] = con
        else:
            self._projections[key] = con

    def add_leaf_connection(self, con):
        assert isinstance(con, Projection)
        con.super = self
        key = con.key
        if not (con.pre in self.pre and con.post in self.post):
            raise ValueError("the sub connection is not belong to this connection group (pre and post is not a member of the connected Assemblies)")
        con.set_name(None)
        self._leaf_connections[key] = con

    def del_connection(self):
        if self.super is not None:
            if self.key in self.super._connections:
                self.super._connections.pop(self.key)
            if self.key in self.super._projections:
                self.super._projections.pop(self.key)
            if self.key in self.super._leaf_connections:
                self.super._leaf_connections.pop(self.key)

            top_leaf = self.top._leaf_connections
            if self.key in top_leaf:
                top_leaf.pop(self.key)
            if self.super.is_empty():
                self.super.del_connection()
            self.super = None
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
            while (len(assb_connections) > 0):
                if to_level >= 0 and to_level <= self.abstract_level:
                    break
                self.abstract_level += 1
                new_assb_connections = []
                for con in assb_connections:
                    if con.is_unit:
                        new_leaf_connections[con.key] = con
                    else:
                        pre_groups = con.pre.get_groups(recursive=False)
                        post_groups = con.post.get_groups(recursive=False)
                        for pre in pre_groups:
                            for post in post_groups:
                                new_con = Projection(pre, post)
                                # con.add_connection(new_con)
                                if new_con.is_unit:
                                    new_leaf_connections[new_con.key] = new_con
                                else:
                                    new_assb_connections.append(new_con)

                assb_connections = new_assb_connections

            for con in assb_connections:
                # 对于超过abstract_level的connection，都认为是leaf_connection
                new_leaf_connections[con.key] = con

            # self._leaf_connections = new_leaf_connections
            self._leaf_connections.clear()
            for key, con in new_leaf_connections.items():
                self.add_leaf_connection(con)

            return list(new_leaf_connections.values())

    def assembly_linked(self, assembly):
        if (assembly is self.pre) or (assembly is self.post):
            return True
        else:
            return False

    def replace_assembly(self, old_assembly, new_assembly):
        if old_assembly is self.pre:
            self.pre = new_assembly
        elif old_assembly is self.post:
            self.post = new_assembly
        else:
            raise ValueError("the old_assembly is not in the connnection")

    def add_super(self, assembly):
        assert isinstance(assembly, Assembly) or isinstance(assembly,
                                                            Projection), "the super is not Assembly or Projection"
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
        repr_str += "pre:{}, ".format(self.pre.name)
        repr_str += "post:{}\n ".format(self.post.name)
        level += 1
        for c in self._projections.values():
            repr_str += c.get_str(level)
        for c in self._connections.values():
            repr_str += c.get_str(level)
        return repr_str

    def train(self, mode=True):
        self.training = mode
        for p in self._projections.values():
            p.train(mode)
        for c in self._connections.values():
            c.training = mode

    def build(self, backend=None):
        self._backend = backend

        # clear previous builded connections
        if backend.builded == True:
            for super_asb in self._supers:
                for con in self._connections:
                    super_asb.del_connection(con)
        for sub_proj in self._projections.values():
            sub_proj.build(backend)
        # if not policy is not given then it does not have connection policy (no default policy)
        if not self._policies:
            return
        connection_inforamtion = None
        for p in self._policies:
            if connection_inforamtion is None:
                connection_inforamtion = p.generate_connection(self.pre, self.post)
            else:
                connection_inforamtion = connection_inforamtion & p.generate_connection(self.pre,
                                                                                        self.post)

        self._connections = connection_inforamtion._connections
        self._projections = connection_inforamtion._projections
        self._leaf_connections = connection_inforamtion._leaf_connections
        all_connections = dict()
        all_connections.update(self._connections)
        all_connections.update(self._leaf_connections)
        new_connections = dict()
        auto_con_num = 0
        for key, con in all_connections.items():
            assert con.is_unit
            if con.link_type is not None:
                link_type = con.link_type
                con_name = con.name
                con_kwargs = con.con_kwargs
                con_kwargs.update(self.ConnectionParameters)
                syn_kwargs = con.syn_kwargs
            else:
                assert self.link_type is not None
                link_type = self.link_type
                con_name = 'auto_con' + str(auto_con_num)
                auto_con_num += 1
                con_kwargs = self.ConnectionParameters
                syn_kwargs = None

            new_connections[key] = Connection(pre=con.pre, post=con.post,
                                              name=con_name, link_type=link_type, syn_kwargs=syn_kwargs, **con_kwargs)
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
        self._syn_tau_variables = dict()
        if conn is not None:
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

        # only deal with lower case names
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
            raise ValueError(('Given synapse model name is not in the model list'))
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
        'sparse_connection_sparse', sparse_connect_sparse
        'sparse_connection', sparse_connect_mask
        'random_connection_sparse', random_connect_sparse
        'random_connection', random_connect_mask
    Args:
        pre(Assembly): the assembly which needs to be connected.
        post(Assembly): the assembly which needs to connect the pre.
        link_type(str): the type for connection: full, sparse, conv...

    Attributes:
        pre_group(groups): the neuron group which need to be connected in the pre.
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

    def __init__(self, pre: Assembly, post: Assembly, name=None,
                 link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=None, max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn',
                 syn_kwargs=None, prefer_device=None, **kwargs):

        super(Connection, self).__init__(pre, post)
        # self.param_init = kwargs.get('param_init', None)
        self.con_kwargs = kwargs
        self.is_parameter = kwargs.get('is_parameter', True)
        self.is_sparse = kwargs.get('is_sparse', False)
        self.weight_quantization = kwargs.get('weight_quantization', False)
        if syn_kwargs is None:
            self.syn_kwargs = dict()
        else:
            self.syn_kwargs = syn_kwargs
        # assert pre._is_terminal
        # assert post._is_terminal
        self.prefer_device = prefer_device
        self.pre = pre  # .get_assemblies()
        self.post = post  # .get_assemblies()
        self.pre_var_name = pre_var_name
        self.post_var_name = post_var_name

        self.pre_num = pre.num
        self.post_num = post.num
        # self.link_num = self.pre_num * self.post_num
        self.shape = (self.post_num, self.pre_num)

        self.pre_groups = None  # pre.get_groups()
        self.post_groups = None  # post.get_groups()
        self.pre_assemblies = None
        self.post_assemblies = None
        self.connection_information = None
        self.mask_info: List[(tuple, tuple)] = list()  # (var_code, op_code)

        self.link_type = link_type
        self.max_delay = max_delay
        self.min_delay = kwargs.get('min_delay', 1.0)
        self.sparse_with_mask = sparse_with_mask
        self._var_names = list()
        self._var_dict = dict()
        self._supers = list()
        self._link_var_codes = list()
        # self._policies = policies
        self._variables = dict()
        self._operations = list()
        self._init_operations = list()
        # self._link_op_codes = list()
        self.w_init = None
        self.b_init = None
        self.w_init_param = dict()
        self.b_init_param = dict()

        self.parameters = kwargs
        # self.bias = kwargs.get('bias', False)

        # self.updated_input = kwargs.get('updated_input', True)
        self.weight_norm = kwargs.get('weight_norm', False)

        self.w_max = None
        self.w_min = None

        self.set_name(name)
        self.running_var = None
        self.running_mean = None
        self.decay = 0.9
        self._variables = dict()
        self._constant_variables = dict()

        self.flatten = kwargs.get('flatten', False)

        # if len(syn_type) == 0:
        #     raise ValueError('Please set the param syn_type, you can set it as \'basic\'')

        if isinstance(syn_type, list):
            self.synapse_type = syn_type
        else:
            self.synapse_type = [syn_type]
        self.synapse_name = []
        self.synapse_class = []

        for i in range(len(self.synapse_type)):
            if isinstance(self.synapse_type[i], str):
                self.synapse_class.append(SynapseModel.apply_model(self.synapse_type[i]))
                self.synapse_name.append(self.synapse_type[i])  # self.model -> self.model_name

            else:
                raise ValueError("only support set synapse model with string")

    def __new__(cls, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                syn_type=None, max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn',
                syn_kwargs=None, **kwargs):
        if cls is not Connection:
            return super().__new__(cls)

        if link_type in cls._connection_subclasses:
            return cls._connection_subclasses[link_type](pre, post, name, link_type, syn_type,
                                                         max_delay, sparse_with_mask, pre_var_name, post_var_name,
                                                         syn_kwargs, **kwargs)

        else:
            raise ValueError("No connection type: %s in Connection classes" % link_type)

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
            raise ValueError(
                ('Given model of type %s does not seem to be a valid ConnectionModel.' % str(type(connection_class))))

        Connection._connection_subclasses[name] = connection_class

    def assembly_linked(self, assembly):
        if (assembly is self.pre) or (assembly is self.post):
            return True
        else:
            return False

    def replace_assembly(self, old_assembly, new_assembly):
        if old_assembly is self.pre:
            self.pre = new_assembly
        elif old_assembly is self.post:
            self.post = new_assembly
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
        repr_str += "pre:{}, ".format(self.pre.name)
        repr_str += "post:{}\n ".format(self.post.name)
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
            ini_delay = (self.max_delay-self.min_delay) * np.random.rand(*self.shape) + self.min_delay
            # ==============================================
            self.variable_to_backend(delay_name, self.shape, ini_delay, True)
            self.variable_to_backend(delay_output_name, (1, *self.shape), 0)

            # add delay index
            self._backend.register_standalone(Op(delay_output_name, delay_queue.select, [delay_name] ,owner=self))

            # add inital to transform initial delay_output
            # self.init_op_to_backend(delay_output_name, delay_queue.transform_delay_output,
            #                                [delay_input_name, delay_name])

        else:
            return

    def clamp_weight(self, weight):
        if not weight.is_sparse:
            if (self.w_max is not None) and (self.w_min is not None):
                self._backend.clamp_(weight, self.w_min, self.w_max)
            elif self.w_max is not None:
                self._backend.clamp_max_(weight, self.w_max)
            elif self.w_min is not None:
                self._backend.clamp_min_(weight, self.w_min)
    def quantize_weight(self, weight):
        return np.round(np.clip(weight + 2 ** 7, 0, 2 ** 8)) - 2 ** 7

    def _add_label(self, var_name: str):
        if '[pre]' in var_name:
            var_name = var_name.replace("[pre]", "")
            var_name = self.get_group_name(self.pre, var_name)
        elif '[link]' in var_name:
            var_name = var_name.replace("[link]", "")
            var_name = self.get_link_name(self.pre, self.post, var_name)
        elif '[post]' in var_name:
            var_name = var_name.replace("[post]", "")
            var_name = self.get_group_name(self.post, var_name)
        elif '[input]' in var_name:
            tmp_var_name = var_name.replace("[input]", "")
            var_name = self.get_input_name(self.pre, self.post)
            if tmp_var_name not in var_name:
                raise ValueError(" the var_name tagged as [input] is not pre_var_name of the connection")
        elif var_name[0] == '[' and var_name[-1] == ']':
            var_name = var_name
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

    @property
    def dt(self):
        return self._backend.dt

    # def decode_syn_op(self, syn_ops, synapse_name, op_len):
    #     if len(synapse_name) > 1:
    #         for i in range(1, op_len):
    #             pre_op_return_name = syn_ops[i-1][0]
    #             post_op_first_input = syn_ops[i][2]
    #             if '[updated]' in post_op_first_input:
    #                 post_op_first_input = post_op_first_input.replace('[updated]', '')
    #             if post_op_first_input == pre_op_return_name:
    #                 temp_name = synapse_name[i-1]+'_' + str(i-1)
    #                 syn_ops[i-1][0] = temp_name
    #                 syn_ops[i][2] = temp_name
    #     return syn_ops

    def decode_syn_op(self, syn_cls, synapse_name):
        if len(synapse_name) > 1:
            for i in range(1, len(synapse_name)):
                pre_op_return_name = syn_cls[i - 1]._syn_operations[-1][0]  # 前一个突触最后一条操作的返回名
                post_op_first_input = syn_cls[i]._syn_operations[0][2]   # 后一个突触第一条操作的第一个输入名
                if '[updated]' in post_op_first_input:
                    post_op_first_input = post_op_first_input.replace('[updated]', '')
                if post_op_first_input == pre_op_return_name:
                    temp_name = synapse_name[i-1]+'_' + str(i-1)
                    syn_cls[i - 1]._syn_operations[-1][0] = temp_name
                    syn_cls[i]._syn_operations[0][2] = temp_name
        syn_ops = []
        for cls in syn_cls:
            for op in cls._syn_operations:
                syn_ops.append(op)
        return syn_ops

    def decode_initializer(self, initial):
        init_name = initial.__class__.__name__
        init_param = initial.__dict__
        return init_name, init_param

    def build(self, backend):
        '''
        add the connection variable, variable name and operation to the backend.
        '''

        self.synapse = []
        self._backend = backend
        # Add weight
        self.assigned_weight = False
        self.assigned_bias = False
        prefer_device=self.prefer_device if self.prefer_device != None else None

        self.set_delay(self.pre, self.post)


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
                if self.assigned_weight is False:
                    self.assigned_weight = True
                else:
                    raise ValueError("connection have multiple weight variable")
                if self.weight_quantization:
                    value = self.quantize_weight(value)
                self.weight = self.variable_to_backend(name=var_name, shape=self.shape, value=value, is_parameter=self.is_parameter,
                                         is_sparse=self.is_sparse, init=self.w_init, init_param=self.w_init_param, prefer_device=prefer_device)  # (var_name, shape, value, is_parameter, is_sparse, init)
            elif 'bias' in key:
                if self.assigned_bias is False:
                    self.assigned_bias = True
                else:
                    raise ValueError("connection have multiple bias variable")
                self.bias = self.variable_to_backend(name=var_name, shape=value.shape, value=value, is_parameter=self.is_parameter,
                                         init=self.b_init, init_param=self.b_init_param, prefer_device=prefer_device)
            elif hasattr(value, 'requires_grad'):
                self.variable_to_backend(var_name, shape=value.shape, value=value, is_parameter=value.requires_grad,
                                         prefer_device=prefer_device)
            else:
                self.variable_to_backend(var_name, shape, value=value, prefer_device=prefer_device)
            # self._var_names.append(var_name)
            # self._var_dict[var_name] = VariableAgent(backend, var_name)

        for (key, value) in self._constant_variables.items():
            var_name = self.add_conn_label(key)
            self.variable_to_backend(name=var_name, shape=None, value=value, is_constant=True, prefer_device=prefer_device)

        weight_name = self.get_link_name(self.pre, self.post, 'weight')
        if self.sparse_with_mask:
            mask = (self.weight != 0)
            mask_name = self.get_link_name(self.pre, self.post, 'mask')
            self.variable_to_backend(mask_name, self.shape, mask, prefer_device=prefer_device)
            self.init_op_to_backend(weight_name, self.mask_operation, [weight_name, mask_name], prefer_device)
            # del self.sparse_with_mask

        # del self.weight
        if (self.w_min is not None) or (self.w_max is not None):
            self.init_op_to_backend(None, self.clamp_weight, [weight_name], prefer_device)

        # TODO: 这里input 和 output的 name 默认没有加label,同时都加上conn的label是不是过于局限了？
        for op in self._operations:
            addcode_op = Op(owner=self)
            addcode_op.func_name = op[1]
            addcode_op.output = self.add_conn_label(op[0])
            if len(op) > 3:  # 为了解决历史的单一list格式的问题
                addcode_op.input = self.add_conn_label(op[2:])
            else:
                addcode_op.input = self.add_conn_label(op[2])
            addcode_op.place = prefer_device
            # if len(op) > 3:
            #     addcode_op[2] = addcode_op[2:]
            #     addcode_op = addcode_op[:3]
            backend.add_operation(addcode_op)

        for op in self._init_operations:
            addcode_op = []
            for ind, name in enumerate(op):
                if ind != 1:
                    addcode_op.append(self.add_conn_label(op[ind]))
                else:
                    addcode_op.append(op[ind])
            if len(op) > 3:
                addcode_op[2] = addcode_op[2:]
                addcode_op = addcode_op[:3]

            self.init_op_to_backend(addcode_op[0], addcode_op[1], addcode_op[2], self.prefer_device)

        # the initialzation of synapse with forward_build kwarg can be read
        if backend.forward_build:
            self.updated_input = True
        else:
            self.updated_input = False
        # syn_ops = []
        for i in range(len(self.synapse_class)):
            if self.synapse_class[i] is not None:
                self.synapse.append(self.synapse_class[i](self, **self.syn_kwargs))

            for (key, value) in self.synapse[i]._syn_variables.items():
                key = self.add_conn_label(key)
                self.variable_to_backend(key, value.shape, value=value)

            for (key, value) in self.synapse[i]._syn_constant_variables.items():
                key = self.add_conn_label(key)
                self.variable_to_backend(key, shape=None, value=value, is_constant=True)
            dt = backend.dt
            for (key, value) in self.synapse[i]._syn_tau_variables.items():
                key = self.add_conn_label(key)
                tau_value = np.exp(-dt / value)
                # 暂时只考虑scalar值
                self.variable_to_backend(key, shape=[1, ], value=tau_value)

            # for op in self.synapse[i]._syn_operations:
            #     syn_ops.append(op)

        #toDO: make sure synapse with multiple ops
        # if 'basic' in self.synapse_name:
        # syn_ops = self.decode_syn_op(syn_ops, self.synapse_name, len(syn_ops))
        syn_ops = self.decode_syn_op(self.synapse, self.synapse_name)

        for sop in syn_ops:
            addcode_op = Op(owner=self)
            addcode_op.func_name = sop[1]
            addcode_op.output = self.add_conn_label(sop[0])
            if len(sop) > 3:  # 为了解决历史的单一list格式的问题
                addcode_op.input = self.add_conn_label(sop[2:])
            else:
                addcode_op.input = self.add_conn_label(sop[2])
            addcode_op.place = prefer_device
            backend.add_operation(addcode_op)

    @abstractmethod
    def condition_check(self, pre_group, post_group):
        NotImplemented()

    def mask_operation(self, weight, mask):
        return weight * mask

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
    def get_delay_input_output(self, pre_group: Assembly, post_group: Assembly):
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
    #     I = self.get_post_name(self.post, 'I')
    #     WgtSum = self.get_post_name(self.post, 'WgtSum')
    #     tauP = self.get_post_name(self.post, 'tauP')
    #     # I = self.get_I_ele_name(self.post)
    #     # WgtSum = self.get_target_name(self.post)
    #     # tauP = self.post.id + ':' + '{tauP}'
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
    #     V_post = self.get_post_name(self.post, 'V')
    #     V_pre = self.get_pre_name(self.post, 'V')
    #     Vtemp1_post = self.get_link_name(self.pre, self.post, 'Vtemp')
    #     I_post = self.get_post_name(self.post, 'I')
    #     weight = self.get_link_name(self.pre, self.post, 'weight')
    #     Vtemp1_pre = self.get_link_name(self.post, self.pre, 'Vtemp')
    #     I_pre = self.get_pre_name(self.post, 'I_ele')
    #     # V_post = self.get_V_name(self.post)
    #     # V_pre = self.get_V_name(self.pre)
    #     # Vtemp1_post = self.get_Vtemp_name(self.pre, self.post)
    #     # I_post = self.get_I_ele_name(self.post)
    #     # weight = self.get_weight_name(self.pre, self.post)
    #     # Vtemp1_pre = self.get_Vtemp_name(self.post, self.pre)
    #     # I_pre = self.get_I_ele_name(self.pre)
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