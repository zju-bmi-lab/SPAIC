# -*- coding: utf-8 -*-
"""
Created on 2021/5/10
@project: SPAIC
@filename: ConnectionPolicy
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description: 
"""

from .Assembly import Assembly
from .Topology import ConnectPolicy, Projection


# class ConnectInformation():
#     _con_count = 0
#
#
#     def __init__(self, pre: Assembly, post: Assembly, level=0):
#         super(ConnectInformation, self).__init__()
#         self.pre = pre
#         self.post = post
#         self.abstract_level = 0
#         self.my_level = level
#         self.super = None
#         if (self.pre.id is not None) and (self.post.id is not None):
#             self.key = self.pre.id + "->" + self.post.id
#         else:
#             self.key = 'default' + str(ConnectInformation._con_count)
#         ConnectInformation._con_count += 1
#
#         self._link_num = 0
#         self._sub_unit_connections = dict()
#         self._sub_assb_connections = dict()
#         self._leaf_connections = {self.key: self}
#
#         if pre._is_terminal and post._is_terminal:
#             self.is_unit = True
#         else:
#             self.is_unit = False
#
#
#     def homologous(self, other):
#         if (other.pre is self.pre) and (other.post is self.post):
#             return True
#         else:
#             return False
#
#     def is_empty(self):
#         if self.is_unit:
#             return False
#         elif len(self._sub_unit_connections) + len(self._sub_assb_connections) == 0:
#             return True
#         else:
#             return False
#
#     def __and__(self, other):
#         if not self.homologous(other):
#             raise ValueError("can't do & operation for nonhomologous connections")
#         if self.is_empty():
#             new_connect = other
#         elif other.is_empty():
#             new_connect = self
#         else:
#             new_connect = ConnectInformation(self.pre, self.post)
#             # unit connections
#             key1 = set(self._sub_unit_connections.keys())
#             key2 = set(other._sub_unit_connections.keys())
#             unite_keys = key1.intersection(key2)
#             for key in unite_keys:
#                 new_connect._sub_unit_connections[key] = self._sub_unit_connections[key]
#                 new_connect._sub_unit_connections[key].super = new_connect
#
#             #assb connections
#             key1 = set(self._sub_assb_connections.keys())
#             key2 = set(other._sub_assb_connections.keys())
#             unite_keys = key1.intersection(key2)
#             for key in unite_keys:
#                 new_connect._sub_assb_connections[key] = self._sub_assb_connections[key] & other._sub_assb_connections[key]
#                 new_connect._sub_assb_connections[key].super = new_connect
#
#         return new_connect
#
#
#     def add_connection(self, con):
#         con.super = self
#         key = con.key
#         if not (con.pre in self.pre and con.post in self.post):
#             raise ValueError("the sub connection is not belong to this connection group (pre and post is not a member of the connected Assemblies)")
#
#         if con.is_unit:
#             self._sub_unit_connections[key] = con
#         else:
#             self._sub_assb_connections[key] = con
#
#         # if len(self._level_connections) >= 2:
#         #     self._level_connections[1].append(con)
#         # else:
#         #     self._level_connections.append([con])
#
#         # new_level = self.my_level + 1
#         # if len(self.super._level_connections) >= new_level+1:
#         #     self.super._level_connections[new_level].append(con)
#         # else:
#         #     self.super._level_connections.append([con])
#
#
#     def expand_connection(self, to_level=-1):
#         assert isinstance(to_level, int), ValueError("level is not int")
#         if self.is_unit:
#             return self._leaf_connections.values()
#         else:
#             new_leaf_connections = dict()
#             assb_connections = self._leaf_connections.values()
#             while(len(assb_connections)>0):
#                 if to_level >= 0 and to_level<=self.abstract_level:
#                     break
#                 self.abstract_level += 1
#                 new_assb_connections = []
#                 for con in assb_connections:
#                     if con.is_unit:
#                         new_leaf_connections[con.key] = con
#                     else:
#                         pre_groups = con.pre.get_groups(recursive=False)
#                         post_groups = con.post.get_groups(recursive=False)
#                         for pre in pre_groups:
#                             for post in post_groups:
#                                 new_con = ConnectInformation(pre, post, self.abstract_level)
#                                 con.add_connection(new_con)
#                                 if new_con.is_unit:
#                                     new_leaf_connections[new_con.key] = new_con
#                                 else:
#                                     new_assb_connections.append(new_con)
#
#                 assb_connections = new_assb_connections
#
#             for con in assb_connections:
#                 #对于超过abstract_level的connection，都认为是leaf_connection
#                 new_leaf_connections[con.key] = con
#             self._leaf_connections = new_leaf_connections
#             return list(new_leaf_connections.values())
#
#
#     def del_connection(self):
#         if self.super is not None:
#             if self.is_unit:
#                 self.super._sub_unit_connections.pop(self.key)
#             else:
#                 self.super._sub_assb_connections.pop(self.key)
#             top_leaf = self.top._leaf_connections
#             if self.key in top_leaf:
#                 top_leaf.pop(self.key)
#             if self.super.is_empty():
#                 self.super.del_connection()
#         else:
#             self.is_unit = False
#
#
#     # def add_var_code(self, var_code):
#     #     if self.is_unit:
#     #         self._var_codes.append(var_code)
#     #     else:
#     #         raise ValueError("can't add var_code for non-unit connections")
#     #
#     # def add_op_code(self, op_code):
#     #     if self.is_unit:
#     #         self._op_codes.append(op_code)
#     #     else:
#     #         raise ValueError("can't add op_code for non-unit connections")
#
#     def set_link_num(self, link_num):
#         if self.is_unit:
#             self._link_num = link_num
#             if self.super is not None:
#                 self.super.add_link_num(link_num)
#         else:
#             raise ValueError("can't set link_num for non-unit connections")
#
#     def add_link_num(self, link_num):
#         self._link_num += link_num
#         if self.super is not None:
#             self.super.add_link_num(link_num)
#
#
#     @property
#     def top(self):
#         if self.super is None:
#             return self
#         else:
#             return self.super.top
#
#     @property
#     def link_num(self):
#         return self._link_num
#
#     @property
#     def sub_connections(self):
#         if self.is_unit:
#             raise ValueError("no sub_connections for unit connections")
#         else:
#             sub_connections = []
#             sub_connections.extend(self._sub_assb_connections.values())
#             sub_connections.extend(self._sub_unit_connections.values())
#             return sub_connections
#
#     @property
#     def leaf_connections(self):
#         return list(self._leaf_connections.values())
#
#     @property
#     def sub_unit_connections(self):
#         if self.is_unit:
#             raise ValueError("no sub_connections for unit connections")
#         else:
#             return list(self._sub_unit_connections.values())
#
#     @property
#     def sub_assb_connections(self):
#         if self.is_unit:
#             raise ValueError("no sub assb_connections for unit connections")
#         else:
#             return list(self._sub_assb_connections.values())
#
#     @property
#     def all_unit_connections(self):
#         if self.is_unit:
#             return [self]
#         else:
#             unit_connections = []
#             unit_connections.extend(self._sub_unit_connections.values())
#             for con in self._sub_assb_connections.values():
#                 unit_connections.extend(con.all_unit_connections)
#             return unit_connections
#
#     @property
#     def all_assb_connections(self):
#         if self.is_unit:
#             raise ValueError("no sub_connections for unit connections")
#         else:
#             assb_connections = []
#             assb_connections.extend(self._sub_assb_connections.values())
#             for con in self._sub_assb_connections.values():
#                 assb_connections.extend(con.all_assb_connections)
#             return assb_connections


class IncludedTypePolicy(ConnectPolicy):

    def __init__(self, pre_types=None, post_types=None, level=-1):
        super(IncludedTypePolicy, self).__init__(level=level)
        self.name = 'Included_policy'
        if isinstance(pre_types, list):
            self.pre_types = pre_types
        elif pre_types is not None:
            self.pre_types = [pre_types]
        else:
            self.pre_types = None

        if isinstance(post_types, list):
            self.post_types = post_types
        elif post_types is not None:
            self.post_types = [post_types]
        else:
            self.post_types = None

    def checked_connection(self, new_connection: Projection):

        leaf_connections = new_connection.expand_connection(self.level)
        if self.post_types is not None:
            self.post_types = set(self.post_types)
            for con in leaf_connections:
                fit_type = self.post_types.intersection(con.post.type)
                if not fit_type:
                    con.del_connection()

        if self.pre_types is not None:
            leaf_connections = new_connection.leaf_connections
            self.pre_types = set(self.pre_types)
            for con in leaf_connections:
                fit_type = self.pre_types.intersection(con.pre.type)
                if not fit_type:
                    con.del_connection()

        return new_connection


class ExcludedTypePolicy(ConnectPolicy):

    def __init__(self, pre_types=None, post_types=None, level=-1):
        super(ExcludedTypePolicy, self).__init__(level=level)
        self.name = 'Excluded_policy'
        if isinstance(pre_types, list):
            self.pre_types = pre_types
        elif pre_types is not None:
            self.pre_types = [pre_types]
        else:
            self.pre_types = None

        if isinstance(post_types, list):
            self.post_types = post_types
        elif post_types is not None:
            self.post_types = [post_types]
        else:
            self.post_types = None

    def checked_connection(self, new_connection: Projection):

        leaf_connections = new_connection.expand_connection(self.level)
        if isinstance(leaf_connections, dict):
            leaf_connections = leaf_connections.values()

        if self.post_types is not None:
            self.post_types = set(self.post_types)
            for con in leaf_connections:
                fit_type = self.post_types.intersection(con.post.type)
                if fit_type:
                    con.del_connection()

        if self.pre_types is not None:
            leaf_connections = new_connection._leaf_connections
            self.pre_types = set(self.pre_types)
            for con in leaf_connections:
                fit_type = self.pre_types.intersection(con.pre.type)
                if fit_type:
                    con.del_connection()
        return new_connection


class IndexConnectPolicy(ConnectPolicy):

    def __init__(self, pre_indexs=None, post_indexs=None, level=-1):
        super(IndexConnectPolicy, self).__init__(level=level)
        self.name = 'Index_policy'
        if type(pre_indexs) is list:
            self.pre_indexs = pre_indexs
        else:
            raise ValueError("pre_indexs should be list")

        if type(post_indexs) is list:
            self.post_indexs = post_indexs
        else:
            raise ValueError("post_indexs should be list")

        if len(pre_indexs) != len(post_indexs):
            raise ValueError(" the length of pre and post index is not equal")
        else:
            self.index_len = len(pre_indexs)

    def checked_connection(self, new_connection: Projection):

        pre_level_groups = new_connection.pre.get_leveled_groups()
        post_level_groups = new_connection.post.get_leveled_groups()
        for ind in range(self.index_len):
            pre_ind = self.pre_indexs[ind]
            post_ind = self.post_indexs[ind]
            if not hasattr(pre_ind, '__iter__'):
                pre_group = pre_level_groups[2][pre_ind]
            else:
                pre_group = pre_level_groups[pre_ind[0]][pre_ind[1]]
            if not hasattr(post_ind, '__iter__'):
                post_group = post_level_groups[2][post_ind]
            else:
                post_group = post_level_groups[post_ind[0]][post_ind[1]]
            new_connection.add_connection(Projection(pre_group, post_group, level=1))

        return new_connection

#
# asb1 = Assembly()
# asb2 = Assembly()
#
#
# class Group1(Assembly):
#     def __init__(self, name=None):
#         super(Group1, self).__init__(name)
#
#         self.n1 = NeuronGroup(100 ,model='clif')
#         self.n2 = NeuronGroup(100, model='clif')
#         self.n1.add_type('in')
#         self.n2.add_type('out')
#
#
# with asb1:
#     g1 = NeuronGroup(100 ,model='clif')
#     g2 = NeuronGroup(100 ,model='clif')
#     g3 = Group1()
# asb1.g1.add_type('in')
#
#
# with asb2:
#     g4 = NeuronGroup(100 ,model='clif')
#     g5 = NeuronGroup(100 ,model='clif')
#     g6 = Group1()
#
# asb2.g4.add_type('out')
#
#
#
#
#
#
# p1 = IndexConnectPolicy(pre_indexs=[(1,1), (2,0)], post_indexs=[(2,1), (1,0)])
# c = p1.generate_connection(asb1, asb2)
# print(c)
#
