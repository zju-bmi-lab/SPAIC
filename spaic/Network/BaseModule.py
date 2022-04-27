# -*- coding: utf-8 -*-
"""
Created on 2020/9/9
@project: SPAIC
@filename: BaseModule
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description: 
"""
from abc import  abstractmethod
from collections import OrderedDict
import spaic

class BaseModule():
    '''
    Base class for all snn modules (assemblies, connection, learner, monitor, piplines).

    '''
    _Module_Count = 0
    _class_label = '<bm>'
    def __init__(self):
        self.id = None
        self.name = None
        self.hided = False
        self._supers = []
        pass

    @abstractmethod
    def build(self, backend):
        NotImplementedError()

    @abstractmethod
    def get_str(self, level):
        NotImplementedError()

    def set_name(self, given_name):

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
            spaic.global_module_name_count += 1
            self.name = 'autoname' + str(spaic.global_module_name_count)

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


class VariableAgent(object):
    def __init__(self, backend, var_name):
        super(VariableAgent, self).__init__()
        assert isinstance(backend, spaic.Backend)
        self._backend = backend
        self._var_name = var_name

    @property
    def var_name(self):
        return self._var_name

    @property
    def value(self):
        return self._backend.get_varialble(self._var_name)

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


