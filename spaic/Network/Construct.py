# -*- coding: utf-8 -*-
"""
Created on 2020/12/23
@project: SPAIC
@filename: WithBuildFuncs
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description: 
"""
from ..Network.Assembly import Assembly
import spaic

# ================= Assembly Functions used in with ==========================

def add_assembly(name, assembly):
    context = spaic.global_assembly_context_list
    assert len(context) >0, "use global assembly build functions outside of the 'with' context"
    assert isinstance(context[-1], Assembly), "the object of the context is not an Assembly"
    spaic.global_assembly_init_count += 1
    self = context[-1]
    self.add_assembly(name, assembly)
    spaic.global_assembly_init_count -= 1



def del_assembly(assembly=None, name=None):

    context = spaic.global_assembly_context_list
    assert len(context) >0, "use global assembly build functions outside of the 'with' context"
    assert isinstance(context[-1], Assembly), "the object of the context is not an Assembly"
    spaic.global_assembly_init_count += 1
    self = context[-1]
    self.del_assembly(name, assembly)
    spaic.global_assembly_init_count -= 1


def add_connection(name, connection):
    context = spaic.global_assembly_context_list
    assert len(context) >0, "use global assembly build functions outside of the 'with' context"
    assert isinstance(context[-1], Assembly), "the object of the context is not an Assembly"
    spaic.global_assembly_init_count += 1
    self = context[-1]
    self.add_connection(name, connection)
    spaic.global_assembly_init_count -= 1


def del_connection(connection=None, name=None):

    context = spaic.global_assembly_context_list
    assert len(context) >0, "use global assembly build functions outside of the 'with' context"
    assert isinstance(context[-1], Assembly), "the object of the context is not an Assembly"
    spaic.global_assembly_init_count += 1
    self = context[-1]
    self.del_connection(connection, name)
    spaic.global_assembly_init_count -= 1



def copy_assembly(name, assembly):
    context = spaic.global_assembly_context_list
    assert len(context) > 0, "use global assembly build functions outside of the 'with' context"
    assert isinstance(context[-1], Assembly), "the object of the context is not an Assembly"
    spaic.global_assembly_init_count += 1
    self = context[-1]
    self.copy_assembly(name, assembly)
    spaic.global_assembly_init_count -= 1


def replace_assembly(old_assembly, new_assembly):
    context = spaic.global_assembly_context_list
    assert len(context) > 0, "use global assembly build functions outside of the 'with' context"
    assert isinstance(context[-1], Assembly), "the object of the context is not an Assembly"
    spaic.global_assembly_init_count += 1
    self = context[-1]
    self.replace_assembly(old_assembly, new_assembly)
    spaic.global_assembly_init_count -= 1


def merge_assembly(assembly):
    context = spaic.global_assembly_context_list
    assert len(context) > 0, "use global assembly build functions outside of the 'with' context"
    assert isinstance(context[-1], Assembly), "the object of the context is not an Assembly"
    spaic.global_assembly_init_count += 1
    self = context[-1]
    self.merge_assembly(assembly)
    spaic.global_assembly_init_count -= 1

def select_assembly( assemblies, name=None):
    context = spaic.global_assembly_context_list
    assert len(context) > 0, "use global assembly build functions outside of the 'with' context"
    assert isinstance(context[-1], Assembly), "the object of the context is not an Assembly"
    spaic.global_assembly_init_count += 1
    self = context[-1]
    if name is None:
        new_asb = self.select_assembly(assemblies, context)
    else:
        new_asb = self.select_assembly(assemblies, name)
    spaic.global_assembly_init_count -= 1
    return new_asb