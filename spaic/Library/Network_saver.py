# -*- coding: utf-8 -*-
"""
Created on 2020/8/17
@project: SPAIC
@filename: Network_saver
@author: Mengxiao Zhang
@contact: mxzhangice@gmail.com

@description:
对既定格式网络的存储
"""

import os
from ..Network.Assembly import Assembly
from ..Neuron.Neuron import NeuronGroup
from ..Neuron.Node import Node
from ..Network.Topology import Connection
from ..Backend.Backend import Backend
from ..Network.Topology import Projection
from ..Monitor.Monitor import Monitor

import time


def network_save(Net: Assembly, filename=None, trans_format='json', combine=False, save=True, save_weight=True):
    '''
        Save network to files.

        Args:
            Net(Assembly) : The network needed to be saved.
            filename(Str) : The filename of the file that save target network.
            trans_format(str) : The format of file, could be 'json' or 'yaml'
            combine(Boolen) : Whether combine weight and structure of the Network into on file, False by default.
            save(Boolen) : Whether need to save the structure.

        Return:
            filename(str) : The filename of the file we save network, since it
                will give an auto name if no name given.

        Examples:
            >>> save_file = network_save(Net, "TestNetwork", trans_format='yaml', combine=True, save=True)
            ("TestNetwork", a dict of the structure of the network)
            or
            >>> save_file = network_save(Net, "TestNetwork", trans_format='json', combine=True, save=False)
            a dict of the structure of the network

    '''

    if filename is None:
        if Net.name:
            filename = Net.name + str(time.time())
        else:
            filename = "autoname" + str(time.time())

    path = './NetData/' + filename

    if save:
        if 'NetData' not in os.listdir(os.getcwd()):
            os.mkdir('NetData')
        if filename not in os.listdir(os.getcwd() + '/NetData'):
            os.mkdir(path)

    result_dict = trans_net(Net=Net, path=path, combine=combine, save=save, save_weight=save_weight)

    if trans_format == "yaml":
        import yaml
        result = yaml.dump(result_dict, indent=4)
        ends = '.yml'
    elif trans_format == 'json':
        import json
        result = json.dumps(result_dict, indent=4)
        ends = '.json'
    else:
        raise ValueError("Wrong data format. Only support yaml/json format.")

    if save:
        with open(path+'/'+filename+ends, 'w+') as f:
            f.write(result)
        print("Save Complete.")
        return filename

    print("Complete.")

    return filename, result_dict


def trans_net(Net: Assembly, path: str, combine: bool, save: bool, save_weight: bool):
    '''
        Transform the structure of the network for saving.

        Args:
            Net(Assembly): target network.
            path(string): Target path for saving net data.

        return:
            result_dict(dictionary) : the result diction of the whole Network.

        Example:
            yaml_net = trans_net(Net)

    '''
    result_dict = dict()
    net_name = Net.name
    result_dict[net_name] = []

    for g in Net._groups.values():
        if g._class_label == '<asb>' or g._class_label == '<net>':  # translate other nets
            sub_net_name = g.name
            result_dict[net_name].append(trans_net(g, path+'/'+str(sub_net_name), combine, save, save_weight=False))
        elif g._class_label == '<neg>':  # translate layers
            result_dict[net_name].append(trans_layer(g))
        elif g._class_label == '<nod>':  # translate nodes
            result_dict[net_name].append(trans_node(g))
        else:
            # TODO: if get wrong _class_label, need check the type of
            #  this element
            pass

    for p in Net._projections.values():
        result_dict[net_name].append(trans_projection(p))

    for c in Net._connections.values():  # translate connections
        result_dict[net_name].append(trans_connection(c, combine))

    if '_monitors' in dir(Net):
        mon_dict = {'monitor': []}
        result_dict[net_name].append(mon_dict)
        for monitor in Net._monitors.items():
            mon_dict['monitor'].append(trans_monitor(monitor))

    if '_learners' in dir(Net):
        for key, g in Net._learners.items():  # translate learners
            result_dict[net_name].append({key: trans_learner(g, key)})
    # result_dict[net_name].append({'learners':trans_learner(Net._learners)})

    if (not combine) and save_weight:
        if Net._backend:
            result_dict[net_name].append(
                {'backend': trans_backend(Net._backend, path, save)}
            )
        else:
            import warnings
            warnings.warn("Net._backend not exist. Please check whether need save weight")

    return result_dict


def trans_node(node: Node):
    '''
        Transform the structure of the Node layer for saving and extract the
            parameters.

        Args:
            node (Node): target node layer, like input layer and output layer

        return:
            result_dict (dictionary): the result diction with necessary
                parameters of the layer.

    '''

    needed = ['shape', 'num', 'dec_target', '_time', '_dt', 'coding_method',
              'coding_var_name', 'node_type', 'name', ]

    result_dict = dict()
    para_dict = dict()

    for key, para in node.__dict__.items():
        if key in needed:
            para_dict[key] = para

    para_dict['shape'] = list(para_dict['shape'][1:])

    if para_dict['dec_target']:
        para_dict['dec_target'] = para_dict['dec_target'].name

    import torch
    # if isinstance(para_dict['shape'], torch.Size):
    #     if list(para_dict['shape']) == [1]:
    #         para_dict['shape'] = (1,)
    #     else:
    #         para_dict['shape'] = tuple(list(para_dict['shape']))

    para_dict['_class_label'] = '<nod>'
    result_dict[node.name] = para_dict
    return result_dict


def trans_layer(layer: NeuronGroup):
    '''
        Transform the structure of the layer for saving and extract the
            parameters.

        Args:
            layer (NeuronGroup): target layer

        return:
            result_dict (dictionary): the result diction with necessary
                parameters of the layer.

    '''
    result_dict = dict()
    para_dict = dict()


    unneeded = ['id', 'hided', '_backend', '_connections', '_supers', '_input_connections',
                '_output_connections', '_var_names', 'model_class', '_operations', 'model',
                '_groups']
    needed = ['model_name', 'id', 'name', 'num', 'position', 'shape', 'type', 'parameters']
    # Needed parameters: neuron_number, neuron_shape, neuron_type,
    # neuron_position, neuron_model, name, parameters.

    for key, para in layer.__dict__.items():
        if key in needed:
            para_dict[key] = para

    if para_dict['position'] != ('x, y, z' or 'x, y'):
        para_dict.pop('position')
    para_dict['_class_label'] = '<neg>'

    para_dict['neuron_parameters'] = layer.model.neuron_parameters

    result_dict[layer.name] = para_dict
    return result_dict


def trans_projection(projection: Projection):
    '''
        Transform the structure of the projection for saving and extract the
            parameters.

        Args:
            projection (Projection): target projection

        return:
            result_dict (dictionary): the result diction with necessary
                parameters of the projection.

    '''
    result_dict = dict()
    para_dict = dict()
    name_needed = ['pre_assembly', 'post_assembly']
    needed = ['name', '_policies', 'link_type', 'ConnectionParameters']

    for key, para in projection.__dict__.items():
        if key in name_needed:
            para_dict[key] = para.name
        elif key in needed:
            para_dict[key] = para

    para_dict['_class_label'] = '<prj>'
    result_dict[projection.name] = para_dict

    return result_dict


def trans_connection(connection: Connection, combine: bool):
    '''
        Transform the structure of the connection for saving and extract the
            parameters.

        Args:
            connection (Connection): target connection
            combine (bool): whether combine weights.

        return:
            result_dict (dictionary): the result diction with necessary
                parameters of the connection.

    '''
    result_dict = dict()
    para_dict = dict()

    name_needed = ['pre_assembly', 'post_assembly']
    needed = ['name', 'link_type', 'max_delay', 'sparse_with_mask',
              'pre_var_name', 'post_var_name', 'parameters']
    unneeded = ['id', 'hided', 'pre_groups', 'post_groups', 'pre_assemblies', 'post_assemblies',
                'unit_connections', '_var_names', '_supers', '_policies', '_backend']
    # **link_parameters

    for key, para in connection.__dict__.items():
        if key in name_needed:
            para_dict[key] = para.name
        elif key in needed:
            para_dict[key] = para
    if combine:     # 是否需要在文件中存储weight
        para_dict['weight'] = dict()
        t = 0
        for conn in connection.unit_connections:
            # preg = conn.pre_assembly
            # posg = conn.post_assembly
            # weight_name = connection.get_weight_name(preg, posg)
            weight_name = conn[3][0]
            weight = connection._backend._variables[weight_name].detach().cpu().numpy().tolist()

            para_dict['weight'][weight_name] = weight
            t += 1

    para_dict['_class_label'] = '<con>'
    result_dict[connection.name] = para_dict

    return result_dict


def trans_backend(backend: Backend, path: str, save: bool):
    '''
    Transform the data of backend for saving.

    Args:
        backend: target backend.
        path(string): Target path for saving net data.

    Returns:
        result(dict): Contain the parameters of backend to be saved.
    '''

    # Needed parameters: _variables, _parameters_dict, _InitVariables_dict,
    # dt, time, time_step, _graph_var_dicts,

    # key_parameters_dict = ['_variables', '_parameters_dict', '_InitVariables_dict']
    key_parameters_dict = ['_parameters_dict']
    key_parameters_list = ['dt', 'time', 'n_time_step']


    if backend._variables is None:
        import warnings
        warnings.warn('Backend end don\'t have variables. Have not built Backend. Weight not exists.')
        return
    else:
        if 'parameters' not in os.listdir(os.getcwd() + path[1:]):
            os.mkdir(os.getcwd() + path[1:]+'/parameters')
    sim_path = path + '/parameters'

    import torch

    result_dict = dict()
    for key in key_parameters_dict:
        if save:
            save_path = sim_path + '/' + key + '.pt'
            data = backend.__dict__[key]
            torch.save(data, save_path)
            result_dict[key] = './parameters/' + key + '.pt'
        else:
            raise ValueError("Wrong save choosen, since parameters can be get from network"
                             "unneeded to use network_save function.")

    for key in key_parameters_list:
        result_dict[key] = backend.__dict__[key]


    # for key in key_parameters_dict:
    #     if key not in os.listdir(sim_path):
    #         os.mkdir(os.getcwd()+path[1:]+'/backend/'+key)
    #     result_dict[key] = dict()
    #     num = 0
    #     for k in backend.__dict__[key].keys():
    #         result_dict[key][k] = './backend/' + key + '/' + \
    #                                            str(num) + '.pt'
    #         num += 1
    #         data = backend.__dict__[key][k]
    #         if save:
    #             torch.save(data, path+result_dict[key][k][1:])
    # torch.save(backend._variables, path+'total.pt')
    # result = {'backend': result_dict}

    return result_dict


def trans_learner(learner, learn_name):
    """
    Transform learner parameters to dict.
    Args:
        learner: Target learner with needed parameters.

    Returns:
        result(dict): Contain the parameters of learner to be saved.
    """
    import torch
    # result_dict = dict()
    para_dict = dict()
    trainables = ['trainable_connections', 'trainable_groups', 'trainable_nodes']
    para_dict['trainable'] = []
    needed = ['name', 'parameters', 'optim_name', 'optim_lr', 'optim_para', 'lr_schedule_name', 'lr_schedule_para']
    para_dict['_class_label'] = '<learner>'
    for key in needed:
        if key in learner.__dict__.keys():
            para = learner.__dict__.get(key)
            para_dict[key] = para if type(para) != torch.Tensor \
                    else para.detach().cpu().numpy().tolist()

    for train_name in trainables:
        for key, train in learner.__dict__[train_name].items():
            para_dict['trainable'].append(train.name)

    para_dict['algorithm'] = para_dict['name']
    para_dict['name'] = learn_name
    if 'algorithm' in para_dict['parameters'].keys():
        del para_dict['parameters']['algorithm']

    return para_dict


def trans_monitor(monitor: Monitor):
    from spaic.Monitor.Monitor import StateMonitor, SpikeMonitor
    needed = ['var_name', 'index', 'dt', 'get_grad', 'nbatch']
    name, mon = monitor
    result_dict = dict()
    for i in needed:
        result_dict[i] = mon.__dict__[i]
    result_dict['target'] = mon.target.name if mon.target else None
    result_dict['monitor_type'] = 'StateMonitor' if type(monitor[1]) == StateMonitor else 'SpikeMonitor'

    return {name: result_dict}



# def






