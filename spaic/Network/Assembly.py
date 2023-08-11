# -*- coding: utf-8 -*-
"""
Created on 2020/8/5
@project: SPAIC
@filename: Assembly
@author: Hong Chaofei
@contact: hongchf@gmail.com
@description:
"""
# import spaic

from collections import OrderedDict
from .BaseModule import BaseModule, VariableAgent
# from .Topology import Projection, Connection
from ..Backend.Backend import Backend
# from ..Neuron.Neuron import NeuronGroup
# from ..Neuron.Node import Node
from abc import ABC, abstractmethod
from torch import nn
from typing import List
import inspect

from .. import global_assembly_context_list, global_assembly_init_count
from .. import global_assembly_context_omit_start, global_assembly_context_omit_end
global global_assembly_context_list, global_assembly_init_count

# class ContextMetaClass(type):
#
#     def __call__(self, *args, **kwargs):
#         spaic.global_assembly_init_count += 1
#         rv = super(ContextMetaClass, self).__call__(*args, **kwargs)
#         spaic.global_assembly_init_count -= 1
#         return rv

class Assembly(BaseModule):
    r"""Base class for all network units.

        The Assembly represent an abstract network structure, it defines the basic network construct behavior and attributes.
        It can contain any neural network units like neurongroups, nodes, other assemblies and their connections.
        The classes like Node, NeuronGroup, and Network are special cases of Assembly.

        Methods:
            add_assembly: Add a new assembly to this assembly as its member.
            del_assembly: Delete an existed member assembly of this assembly.
            add_connection: Add the connection between two member assemblies of this assembly.
            del_connection: Delete an existed connection between member assemblies of this assembly.
            copy_assembly: Copy an existed assembly structure into this assembly.
            replace_assembly: Replace an existed member assembly with a new assembly.
            merge_assembly:   Add the member assemblies and connections of the target assembly, which are not already included in this assembly, to this assembly.
            select_assembly:   Select a list of member assemblies in this assembly, and form a new assembly that contains those selected assemblies.



        Attributes:
            _class_label(str): the label is a static variable to imply the class of this object
            _backend(Backend): the backend backend this assembly runs on
            _groups(OrderedDict): the container of member assemblies
            _connections(OrderedDict): the container of member connections
            _supers(list): the super assemblies that add this assembly as their member assemblies
            _input_connections(list): the connections that use this assembly as post-synaptic target
            _output_connections(list): the connections that use this assembly as pre-synaptic target
            num(int): the total number of neurons this assembly contains
            position(Tuple(int, int)): the top level positon of this assembly
            _var_names: The backend variable names this assembly and its members contains


    """

    _class_label = '<asb>'
    _is_terminal = False

    def __init__(self, name=None):
        '''Base class for  neural network units, and defines the basic attributes of a group object.

        It contains other Assemblies (including neuron_groups, networks and nodes) and their connections.
        This class provides a abstract representation of network topology.

        Args:
           name(str): name of the network assembly

        Attributes:
            _backend(Backend): the backend backend this assembly runs on
            _groups(OrderedDict): the container of member assemblies
            _connections(OrderedDict): the container of member connections
            _supers(list): the super assemblies that add this assembly as their member assemblies
            _input_connections(list): the connections that use this assembly as post-synaptic target
            _output_connections(list): the connections that use this assembly as pre-synaptic target
            num(int): the total number of neurons this assembly contains
            position(Tuple(int, int)): the top level positon of this assembly
            _var_names: The backend variable names this assembly and its members contains
        '''
        super(Assembly, self).__init__()
        context = global_assembly_context_list
        init_count = global_assembly_init_count

        self.set_name(name)
        self._backend: Backend = None
        # self._groups: OrderedDict[str, Assembly] = OrderedDict()
        # self._connections: OrderedDict[str, Connection] = OrderedDict()
        # self._projections: OrderedDict[str, Projection] = OrderedDict()
        self._groups: OrderedDict[str] = OrderedDict()
        self._connections: OrderedDict[str] = OrderedDict()
        self._projections: OrderedDict[str] = OrderedDict()
        self._supers = list()
        self._input_connections = list()
        self._output_connections = list()
        self._input_modules = list()
        self._output_modules = list()
        self.num = 0
        self.position = None
        self.model_name = None
        # _var_names 和 _var_dict移动到了BaseModule 作为网络各模块的通用属性
        # self._var_names = []
        # self._var_dict = dict()
        self.context_enterpoint = 0
        self.type = []

    # front-end functions
    def add_type(self, type):
        self.type.append(type)

    def get_labeled_name(self, key: str):
        if isinstance(key, str):
            if '[updated]' in key:
                return self.id + ':' +'{'+key.replace('[updated]',"")+'}' + '[updated]'
            else:
                return self.id + ':' +'{'+key+'}'
        elif isinstance(key, VariableAgent):
            return key.var_name
        elif isinstance(key, list) or isinstance(key, tuple):
            keys = []
            for k in key:
                if isinstance(k, str):
                    if '[updated]' in k:
                        mk = self.id + ':' + '{' + k.replace('[updated]', "") + '}' + '[updated]'
                    else:
                        mk = self.id + ':' + '{' + k + '}'
                    keys.append(mk)
                elif isinstance(k, VariableAgent):
                    keys.append(k.var_name)
            return keys

    def add_assembly(self, name, assembly):
        """
        Add a new assembly to this assembly as its member.
        This is the basic method to build the network structure, neurongroups, nodes and other assemblies is added
        to this assembly by calling add_assembly function, explicitly or implicitly.

        Args:
            name (str): the attribute name of the added assembly
            assembly (Assembly): the assembly to be added

        Returns:
            None

        Examples:

            if you want to add a new assembly to in assembly initialization, you can explicitly use add_assembly:

            >>> def __init__(self, name=None):
            >>>    ...
            >>>    self.add_assembly(name='layer1', assembly=Assembly())

            or implicitly use add_assembly:

            >>> def __init__(self, name=None):
            >>>     ...
            >>>     self.layer1 = Assembly()

            you can also add new assemblies after the target assembly has been created:

            >>> TestAsb = Assembly()
            >>> TestAsb.add_assembly(name='layer1', assembly=Assembly())
        """

        if self._backend: self._backend.builded = False

        assert assembly not in self._groups.values(), "assembly %s is already in the assembly %s"%(name, self.name)
        assert name not in self._groups, "assembly with name: %s have the same name with assembly already in the assembly %s"%(name, self.name)
        self.__setattr__(name, assembly)

    def del_assembly(self, assembly=None, name=None):
        """
        Delete an existed member assembly of this assembly.

        User can delete by assembly object or delete by assembly name.

        Args:
            assembly(Assembly): the member assembly to be deleted
            name(str): the member name of the assembly to be deleted
        Returns:
            None

        Examples:
            User can delete member assembly by object:

            >>> TestAsb = Assembly() # assuming contains neurongroups and network structure
            >>> TestAsb.del_assembly(assembly=TestAsb.layer1)

            User can also delete member assembly by the name:

            >>> TestAsb = Assembly() # assuming contains neurongroups and network structure
            >>> TestAsb.del_assembly(name='layer1')
        """
        if self._backend: self._backend.builded = False

        if assembly is not None:
            deleted_assembly = False
            for gkey, value in self._groups.items():
                if value is assembly:
                    del self._groups[gkey]
                    del self.__dict__[gkey]
                    deleted_assembly = True
            assert deleted_assembly, " try to delete an assembly that is not in the group"

            for ckey in self._connections.keys():
                if self._connections[ckey].assembly_linked(assembly):
                    del self._connections[ckey]

        elif name is not None:
            assert name in self._groups, " try to delete an assembly that is not in the group"
            assembly = self._groups[name]
            del self._groups[name]
            del self.__dict__[name]
            for ckey in self._connections.keys():
                if self._connections[ckey].assembly_linked(assembly):
                    del self._connections[ckey]

    def add_connection(self, name, connection):
        """
        Add the connection between two member assemblies of this assembly.

        Args:
            name(str): name of this connection
            connection(Connection): the new connection to be added to the assembly

        Returns:
            None

        Examples:

            >>> TestAsb = Assembly() # assuming contains neurongroups and network structure
            >>> TestAsb.add_connection(name='con1', connection=Connection(self.layer1, self.layer2, link_type='full'))

        """
        if self._backend: self._backend.builded = False
        # assert connection.pre in self.get_groups(), 'pre %s is not in the group' % connection.pre.name
        # assert connection.post in self.get_groups(), 'post %s is not in the group' % connection.post.name
        if name in self._connections:
            if connection is self._connections[name]:
                raise ValueError(" connection is already in the assembly's connection list")
            else:
                raise ValueError("duplicated name for the connection")
        else:
            # self._connections[name] = connection
            self.__setattr__(name, connection)

    def del_connection(self, connection=None, name=None):
        """
        Delete an existed connection between member assemblies of this assembly.

        User can delete the connection by connection object or by connection name.

        Args:
            connection(Connection): the connection object to be deleted
            name(str): the name of connection to be deleted

        Returns:
            None

        Examples:
            Delete by object:

            >>> TestAsb = Assembly() # assuming it contains neurongroups and network structure
            >>> TestAsb.del_connection(connection=TestAsb.con1)

            Delete by name:

            >>> TestAsb = Assembly()
            >>> TestAsb.del_connection(name='con1')
        """
        if self._backend: self._backend.builded = False
        if connection is not None:
            deleted_connection = False
            for ckey, value in self._connections.items():
                if value is connection:
                    dict.__getitem__()
                    del self._connections[ckey]
                    del self.__dict__[ckey]
                    deleted_connection = True 
            assert deleted_connection,  " try to delete an connection that is not in the group"
        elif name is not None:
            assert name in self._connections, " try to delete an connection that is not in the group"
            del self._connections[name]
            del self.__dict__[name]

    def add_projection(self, name, projection):
        """
        Add the projection between two member assemblies of this assembly.

        Args:
            name(str): name of this projection
            projection(Projection): the new projection to be added to the assembly

        Returns:
            None

        Examples:

            >>> TestAsb = Assembly() # assuming contains neurongroups and network structure
            >>> TestAsb.add_projection(name='prj1', projection=Projection(self.layer1, self.layer2, link_type='full'))

        """
        if self._backend: self._backend.builded = False
        assert projection.pre in self._groups.values(), 'pre %s is not in the group' % projection.pre.name
        assert projection.post in self._groups.values(), 'post %s is not in the group' % projection.post.name
        if name in self._projections:
            if projection is self._projections[name]:
                raise ValueError(" projection is already in the assembly's projection list")
            else:
                raise ValueError("duplicated name for the projection")
        else:
            self.__setattr__(name, projection)

    def copy_assembly(self, name, assembly):
        """
        Copy an existed assembly structure into this assembly.

        A new assembly is initialized which copy the structure(type and connection of the assembly members) of the original assembly,
        and the new assembly is added to this assembly.

        Args:
            name(str): the name of the new copy assembly
            assembly(Assembly): the assembly object to be copied

        Returns:
            None

        Examples:

            >>> Asb1 = Assembly() # assuming it contains neurongroups and network structure
            >>> Asb2 = Assembly() # assuming it contains neurongroups and network structure
            >>> Asb1.copy_assembly(name='layer2', assembly=Asb2)
        """
        if self._backend: self._backend.builded = False
        rv = assembly.structure_copy(name)
        self.__setattr__(name, rv)

    def replace_assembly(self, old_assembly, new_assembly):
        """
        Replace an existed member assembly with a new assembly.

        Delete the existed old member assembly, add the new assembly to this assembly,
        and redirect related connections from the old assembly to the new assembly.

        Args:
            old_assembly(Assembly): the old member assembly to be replaced
            new_assembly(Assembly): the new assembly

        Returns:
            None

        Examples:

            >>> templateAsb = Assembly() # assuming templateAsb contains a member assembly called asb1
            >>> asb2 = Assembly() # assuming it contains neurongroups and network structure
            >>> templateAsb.replace_assembly(templateAsb.asb1, asb2)
        """
        if self._backend: self._backend.builded = False
        replaced_assembly = False
        for gkey in self._groups.keys():
            if self._groups[gkey] is old_assembly:
                self._groups[gkey] = new_assembly
                self.__dict__[gkey] = new_assembly
                replaced_assembly = True
        assert replaced_assembly, " try to repalce an assembly that is not in the group"

        for con in self._connections.values():
            if con.assembly_linked(old_assembly):
                con.replace_assembly(old_assembly, new_assembly)

    def merge_assembly(self, assembly):
        """
        Add the member assemblies and connections of the target assembly, which are not already included in this assembly, to this assembly.


        Args:
            assembly(Assembly): the target assembly, from which this assembly will copy member assemblies and connections.

        Returns:
            None

        Examples:

            >>> target_asb = Assembly()  # assuming it contains neurongroups and network structure
            >>> test_asb = Assembly() # assuming it contains neurongroups and network structure
            >>> test_asb.merge_assembly(target_asb)
        """
        if self._backend: self._backend.builded = False

        for key, value in assembly._groups.item():
            if key in self._groups:
                # for the different sub_assembly with different name:
                if value is not self._groups[key]:
                    # set a duplicated name with suffix (1),(2),(3),...
                    for num in range(1, 1000):
                        tmp_key = key + '(' + str(num) +')'
                        if tmp_key not in self._groups:
                            self._groups[tmp_key] = value
                            break
            else:
                # for the sub_assembly that is not in self
                self._groups[key] = value

        for key, value in assembly._connections.item():
            if key in self._connections:
                if value is not self._connections[key]:
                    # set a duplicated name with suffix (1),(2),(3),...
                    for num in range(1, 1000):
                        tmp_key = key + '(' + str(num) +')'
                        if tmp_key not in self._connections:
                            self._connections[tmp_key] = value
                            break
            else:
                self._connections[key] = value

    def select_assembly(self, assemblies, name=None, with_connection=True):
        """
        Select a list of member assemblies in this assembly, and form a new assembly that contains those selected assemblies
        and their connections(if with_connection is True).

        Args:
            assemblies(List[Assembly]): list of assemblies (or member assembly names) to be selected to form a new assembly
            name: the name of the new assembly

        Returns:
            a new assembly that contains the selected assemblies and their connections (if with_coonection is True)

        Examples:

            >>> testAsb = Assembly() # assuming it contains member assemblies named asb1, asb2, asb3...
            >>> newAsb1 = testAsb.select_assembly(['asb1', 'asb2'], 'newAsb') # using names
            >>> newAsb2 = testAsb.select_assembly([testAsb.asb2, testAsb.asb3], 'newAsb') # using assembly objects
        """
        if self._backend: self._backend.builded = False
        new_asb = Assembly(name)
        for asb in assemblies:
            if isinstance(asb, str):
                asb_name= asb
                if asb_name in self._groups:
                    asb = self._groups[asb_name]
                    new_asb.add_assembly(asb_name, asb)
                    if with_connection:
                        for key, con in self._connections.items():
                            if con.assembly_linked(asb):
                                new_asb.add_connection(key, con)
                else:
                    ValueError("No assembly name in the groups")
            else:
                assert isinstance(asb, Assembly), "selected object that is not Assembly"
                for gkey, value in self._groups:
                    if value is asb:
                        new_asb.add_assembly(asb.name, asb)
                        if with_connection:
                            for key, con in self._connections.items():
                                if con.assembly_linked(asb):
                                    new_asb.add_connection(key, con)
                else:
                    ValueError("No assembly in the groups")

        return new_asb

    def assembly_hide(self):
        """
        Prohibit this assembly from building and display, but keep this assembly for later use.

        The set this assembly and its member assemblies with the flag hided = True.

        Returns:
            None

        Examples:
            >>> TestAsb = Assembly()
            >>> TestAsb.assembly_hide()
        """
        if self._backend: self._backend.builded = False
        self.enabled = False
        for key, value in self._groups.items():
            value.assembly_hide()
        for key, value in self._connections.items():
            value.enabled = False

    def assembly_show(self):
        """
        Make the hided assembly to normal assembly.

        Returns:
            None

        Examples:
        >>> TestAsb = Assembly() # assuming hided
        >>> TestAsb.assembly_show()

        """
        if self._backend: self._backend.builded = False
        self.enabled = True
        for key, value in self._groups.items():
            value.assembly_show()
        for key, value in self._connections.items():
            value.enabled = True

    def get_groups(self, recursive=True):
        """
        Get all member neurongroups and neurongroups in member assemblies in a list.
        Args:
            recursive(bool): flag to decide if members of the member assemblies should be returned.

        Returns:
            list of all member groups

        """
        if self._groups and recursive:
            all_groups = []
            for g in self._groups.values():
                all_groups.extend(g.get_groups(recursive))
            return all_groups
        elif self._groups and not recursive:
            return list(self._groups.values())
        elif self._class_label == '<asb>' or self._class_label == '<net>':
            return []
        else:
            return [self]

    def get_leveled_groups(self):
        """
        Get list of all sup groups in leveled order, such as [ [self], [subgroups], [subgroup of subgroups], ...]
        Returns:

        """
        if self._is_terminal:
            return [[self]]
        else:
            leveled_groups = [[self]]
            for k, g in self._groups.items():
                new_level_groups = g.get_leveled_groups()
                for level, groups in enumerate(new_level_groups):
                    level += 1
                    if level < len(leveled_groups):
                        leveled_groups[level].extend(groups)
                    else:
                        leveled_groups.append(groups)
        return leveled_groups

    def get_assemblies(self, recursive=True, include_empty=False):
        """
        Get all the member assemblies and assemblies in member assemblies.
        Args:
            recursive(bool): flag to decide if members of the member assemblies should be returned.

        Returns:
            list of all member assemblies
        """

        if type(recursive) is int:
            # use recursive as a level label
            recursive -= 1

        if self._groups and recursive:
            all_assemblies = {self,}
            for g in self._groups.values():
                all_assemblies.update(g.get_assemblies(recursive, include_empty))
            return all_assemblies
        elif self._groups and not recursive:
            return {self}
        elif not self._groups and include_empty:
            return {self}
        else:
            return []

    def get_assembly_key(self, assembly):
        """
        Get the key of the target assembly if it is a member of this assembly
        Args:
            assembly: the target assembly

        Returns:
            the key of target assembly
        """
        for gkey in self._groups.keys():
            if self._groups[gkey] is assembly:
                return gkey
        return False

    def get_super_assemblies(self, assembly):
        """
        Get all the super assembly of the target assembly if it is a member or member's member of self assembly
        Args:
            assembly: the target assembly

        Returns:
            list of super assemblies or []
        """
        if len(self._groups) == 0:
            return []
        elif assembly in self._groups.values():
            return [self]
        else:
            for g in self._groups.values():
                sup_list = g.get_super_assemblies(assembly)
                if sup_list:
                    return_list = [self]
                    return_list.extend(sup_list)
                    return return_list
            return []

    def __contains__(self, item):
        if isinstance(item, Assembly):
            if item is self:
                return True
            elif item in self._groups.values():
                return True
            else:
                for g in self._groups.values():
                    if item in g:
                        return True
                return False
        else:
            return item in self._connections.values()

    def get_connections(self, recursive=True):
        """
            Get the Connections in this assembly
        Args:
            recursive(bool): flag to decide if member connections of the member assemblies should be returned.

        Returns:
            List of Connections
        """
        if not recursive:
            return list(self._connections.values())
        else:
            all_assmblies = self.get_assemblies(recursive=2)
            connections = list()
            connections = self.update_connection(connections, self._connections)
            # connections.update(self._connections.values())
            for asb in all_assmblies:
                if asb is self:
                    # connections.update(asb.get_connections(recursive=False))
                    connections = self.update_connection(connections, asb.get_connections(recursive=False))
                else:
                    # connections.update(asb.get_connections(recursive=True))
                    connections = self.update_connection(connections, asb.get_connections(recursive=True))
            for proj in self._projections.values():
                # connections.update(proj.get_connections(recursive=True))
                connections = self.update_connection(connections, proj.get_connections(recursive=True))
            return connections
    def update_connection(self, container, connections):
        assert isinstance(container, list)
        if isinstance(connections, OrderedDict):
            for con in connections.values():
                if con not in container:
                    container.append(con)
        elif isinstance(connections, list):
            for con in connections:
                if con not in container:
                    container.append(con)
        else:
            raise ValueError("connections type not right")
        return container


    def get_var_names(self):
        """
        Get a list of variable names the assembly member contains.

        """
        return self._var_names

    def get_str(self, level):
        """
        Get a string description of the strcuture of this assembly
        Args:
            level: the deepth of this assembly relative to the top network

        Returns:
            String representations
        """

        level_space = "" + '-'*level
        repr_str = level_space + "|name:{}, type:{}, ".format(self.name, type(self).__name__)
        repr_str += "total_neuron_num:{}\n ".format(self.num)
        level += 1
        for g in self._groups.values():
            repr_str += g.get_str(level)
        for c in self._connections.values():
            repr_str += c.get_str(level)
        for p in self._projections.values():
            repr_str += p.get_str(level)
        return repr_str

    # back-end functions
    def build(self, backend=None, strategy=0):
        """
        Build the front-end network structure into a back-end computation graph.

        Args:
            backend(Backend): the backend backend to be builded into

        Returns:
            None

        """
        self._backend = backend
        print("builder for assembly has been called")
        # for asb in self.get_groups():
        #     asb.set_id()
        # for con in self.get_connections():
        #     con.set_id()

        for key, value in self._connections.items():
            value.build(backend)
        for key, value in self._groups.items():
            value.build(backend)

    def build_projections(self, backend):
        for proj in self._projections.values():
            proj.build(backend)
        for group in self._groups.values():
            group.build_projections(backend)

    def set_id(self):
        """
        Get the ID of this assembly
        """

        if self.id is not None:
            return self.id

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

    def register_connection(self, connection_obj, presynaptic):
        '''
        Register input or output connection of this assembly
        Args:
            connection_obj (Connection.Connection): the connection
            presynaptic (bool) : if this assembly is presynaptic neuron
        Returns:
            None

        '''
        # ###对复杂结构的assembly也进行注册
        # for key, i in self._groups.items():
        #     if presynaptic:
        #         if connection_obj not in i._output_connections:
        #             i._output_connections.append(connection_obj)
        #     else:
        #         if connection_obj not in i._input_connections:
        #             i._input_connections.append(connection_obj)
        # # connection_obj.post_groups
        if presynaptic:
            if connection_obj not in self._output_connections:
                self._output_connections.append(connection_obj)
        else:
            if connection_obj not in self._input_connections:
                self._input_connections.append(connection_obj)

    def register_module(self, module_obj, pre):
        if pre:
            if module_obj not in self._output_modules:
                self._output_modules.append(module_obj)
        else:
            if module_obj not in self._input_modules:
                self._input_modules.append(module_obj)


    def structure_copy(self, name=None):
        """
        Copy the structure of this assembly with new members
        Args:
            name: name of the new Assembly

        Returns:
            the new assembly
        """
        # define a new object but remain the structure
        from copy import deepcopy
        rv = self.__class__.__new__(self.__class__)
        rv.__init__(name)
        tmp_dict = deepcopy(self.__dict__)
        del tmp_dict['name']
        del tmp_dict['_supers']
        del tmp_dict['_input_connections']
        del tmp_dict['_output_connections']
        rv.__dict__.update(tmp_dict)
        return rv

    def add_super(self, assembly):
        '''
        Tell this assemlby  the target assembly is it's super assembly
        Args:
            assembly: the target super assembly
        '''
        assert isinstance(assembly, Assembly), "the super is not Assembly"
        self._supers.append(assembly)

    def del_super(self, assembly):
        """
        Delete certain assembly from super assemblies of this assembly
        Args:
            assembly: the target super assembly

        """
        assert  assembly in self._supers, "the assembly is not in supers"
        self._supers.remove(assembly)


    # def __getattr__(self, name):

    def train(self, mode=True):
        self.training = mode
        for g in self._groups.values():
            g.train(mode)
        for p in self._projections.values():
            p.train(mode)
        for c in self._connections.values():
            c.training = mode

    def eval(self):
        self.train(False)


    def __enter__(self):
        import __main__

        global_assembly_context_list.append(self)
        main_vars = vars(__main__)
        NoInMain = True
        for key in main_vars:
            value = main_vars[key]
            if value is self:
               NoInMain = False
               self.set_name(key)
               break

        if NoInMain:
            raise ValueError("can only construct network using with at __main__")
        else:
            # record the variable number before enter the context
            self.context_enterpoint = main_vars.__len__() -1

    def __exit__(self, exc_type, exc_val, exc_tb):
        import __main__

        global global_assembly_context_omit_start, global_assembly_context_omit_end
        main_vars = vars(__main__)
        endpoint_num = main_vars.__len__() -1

        # depending on the feature that python >=3.7 , dict is insertion ordered, so we can get the subunits by order
        for ind, key in enumerate(main_vars):
            if ind > self.context_enterpoint and ind <= global_assembly_context_omit_start:
                self.__setattr__(key, main_vars[key])
            elif ind > self.context_enterpoint and ind > global_assembly_context_omit_end:
                self.__setattr__(key, main_vars[key])

        global_assembly_context_list.pop()
        global_assembly_context_omit_start = self.context_enterpoint
        global_assembly_context_omit_end = endpoint_num
        # keys = list(globals().keys())
        # print(keys)

    def __setattr__(self, name, value):
        super(Assembly, self).__setattr__(name, value)
        from ..Network.Topology import Connection, Projection
        from ..Neuron.Neuron import NeuronGroup
        from ..Neuron.Node import Node
        if (self.__class__ is NeuronGroup) or (issubclass(self.__class__, Node)):
            # If class is NeuronGroup or the subclass of Node, do not add other object to it.
            return

        if isinstance(value, Assembly):
            if self._backend: self._backend.builded = False
            value.set_name(name)
            self._groups[name] = value
            # self.num += value.num
            value.add_super(self)
        elif isinstance(value, Connection):
            if self._backend: self._backend.builded = False
            self._connections[name] = value
            value.set_name(name)
            value.add_super(self)
        elif isinstance(value, Projection):
            # if it is not Connection but belongs to projection (pure projection)
            if self._backend: self._backend.builded = False
            self._projections[name] = value
            value.set_name(name)
            value.add_super(self)

    def __delattr__(self, name):
        super(Assembly, self).__delattr__(name)
        if name in self._groups:
            if self._backend: self._backend.builded = False
            self._groups[name].del_super(self)
            del self._groups[name]
        elif name in self._connections:
            if self._backend: self._backend.builded = False
            self._connections[name].del_super(self)
            del self._connections[name]

    def __repr__(self):

        repr_str = self.get_str(0)
        return repr_str
