# -*- coding: utf-8 -*-
"""
Created on 2020/9/9
@project: SNNFlow
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
        pass

    @abstractmethod
    def build(self, simulator):
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


class NetModule(BaseModule):
    '''
    Base class for snn network modules: assemblies, connection
    '''

    def __init__(self):
        super(NetModule, self).__init__()

        self.trainable_parameter_names = OrderedDict()

    def add_trainable_names(self, name):
        pass


