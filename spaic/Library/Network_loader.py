# -*- coding: utf-8 -*-
"""
Created on 2020/8/17
@project: SPAIC
@filename: Network_loader
@author: Mengxiao Zhang
@contact: mxzhangice@gmail.com

@description:
对已按格式储存网络的加载和重建
"""

import yaml
import json

from ..Network.Network import Network
from ..Network.Assembly import Assembly
from ..Neuron.Neuron import NeuronGroup
from ..Neuron.Node import Node, Decoder, Encoder, Action, Generator, Reward
from ..Network.Topology import Connection
from ..Backend.Backend import Backend
from ..Backend.Torch_Backend import Torch_Backend
from ..Learning.Learner import Learner
from ..Network.Topology import Projection
from ..Monitor.Monitor import Monitor, StateMonitor, SpikeMonitor
from ..IO.Initializer import BaseInitializer
from ..IO import Initializer as Initer

import torch


def network_load(filename=None, path=None, device='cpu', load_weight=True):
    '''
        The main function for getting the target filename and reloading the
            network.

        Args:
            filename(str) : The filename of the target network, given by user.
            dataloader(dataloader) : The dataloader for input node layer,
                should be given or crash.
            encoding(str) : The encoding model chosen by user.
            device(str) : The device type we choose to run our network.

        Return:
            net(Assembly) :The network that reloaded from the file.

        Example:
            Net = network_load('TestNetwork', dataloader, 'poisson')

    '''
    import os
    if path:
        filedir = path + '/' + filename
    else:
        path = './'
        # filedir = path + filename
    file = filename.split('.')[0]
    origin_path = os.getcwd()
    os.chdir(path + '/' + file)
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = f.read()
            if data.startswith('{'):
                data = json.loads(data)
            else:
                data = yaml.load(data, Loader=yaml.FullLoader)

    else:
        if os.path.exists(filename + '.yml'):
            with open(filename + '.yml', 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)

        elif os.path.exists(filename + '.json'):
            with open(filename + '.json', 'r') as f:
                data = json.load(f)

        elif os.path.exists(filename + '.txt'):
            with open(filename + '.txt', 'r') as f:
                data = f.read()
                if data.startswith('{'):
                    data = json.loads(data)
                else:
                    data = yaml.load(data, Loader=yaml.FullLoader)

        else:
            raise ValueError("file %s doesn't exist, please check the "
                             "filename" % filename)

    net = ReloadedNetwork(net_data=data, device=device, load_weight=load_weight)

    os.chdir(origin_path)
    return net


class ReloadedNetwork(Network):
    '''
        The network rebuild from the yaml file.

        Args:
            net_data(dict) : The network information reloaded from yaml files.
            dataloader(dataloader) : The dataloader of input layer, since the
                large scale of data, we will not save the data.
            encoding(str) : The encoding model, default as poisson, will change
                in the future.
            backend(backend) : Backend that user want to use.
            learner(str) : The learning model of this network, will change in
                the future.
            learner_alpha(int) : The parameter alpha for learning model, will
                change in the future.
            device(str) : The type of device that run our model.

        Methods:
            load_net(self, data: dict) : The function for load the whole
                network, main function of this class.
            load_layer(self, layer: dict) : The function for load layer.
            load_connection(self, con: dict) : The function for load
                connection.
            load_node(self, node: dict) : The function for load node like input
                or output.
            load_backend(self, path: str): The function for load backend.

        Example:
            Net = ReloadNetwork(net_data, dataloader, 'poisson', backend,
                'STCA', 0.5)

    '''

    def __init__(self, net_data: dict, backend=None, device='cpu', load_weight=True, data_type=None):
        super(ReloadedNetwork, self).__init__()

        self.device = device
        self.name = list(net_data)[0]
        self._backend_info = dict()
        self._diff_para = dict()

        self.load_net(net_data)

        self.load_backend(backend, device=device, load_weight=load_weight, data_type=data_type)
        self._backend.initial_step()

        # self._learner = Learner(algorithm='STCA', lr=0.5, trainable=self)

        del self._backend_info

    def load_net(self, data):
        '''
            The function for load the whole network, main function of this class.

            Args:
                data(dict) : The data should contains the network structure and
                    parameter from yaml.

        '''
        setid = 0
        data = data[list(data)[0]]
        for g1 in data:
            if list(g1)[0] == 'backend':
                self._backend_info = g1[list(g1)[0]]
                break
        if 'diff_para_dict' in self._backend_info.keys():
            self._diff_para = torch.load(self._backend_info['diff_para_dict'])
        for g in data:
            if list(g)[0] == 'monitor':
                monitors = g.get('monitor')
                for monitor in monitors:
                    self.load_monitor(monitor)
                continue
            if list(g)[0] == 'backend':
                # self._backend_info = g[list(g)[0]]
                continue
            para = g[list(g)[0]]
            if type(para) is dict:
                if para.get('_class_label') == '<neg>':
                    lay_name = para.get('name')
                    self.add_assembly(name=lay_name,
                                      assembly=self.load_layer(para))
                elif para.get('_class_label') == '<nod>':
                    nod_name = para.get('name')
                    self.add_assembly(name=nod_name,
                                      assembly=self.load_node(para))
                elif para.get('_class_label') == '<con>':
                    con_name = para.get('name')
                    self.add_connection(name=con_name,
                                        connection=self.load_connection(pnet=self, con=para))
                elif para.get('_class_label') == '<prj>':
                    prj_name = para.get('name')
                    self.add_projection(name=prj_name,
                                        projection=self.load_projection(prj=para))
                elif para.get('_class_label') == '<learner>':
                    learner = self.load_learner(para)
                    self.add_learner(para.get('name'), learner)
                    # self._learners[para.get('name')] = learner
                else:
                    print('Unknown class label %d' % para['_class_label'])

                    break
            else:
                self.load_assembly(p_net=self, name=list(g)[0], assembly=para)
                # self.add_assembly(name=list(g)[0], assembly=self.load_assembly(list(g)[0], para))

        del self._diff_para

    def load_assembly(self, p_net, name, assembly: list):
        target = Assembly(name=name)
        p_net.add_assembly(name=name, assembly=target)
        for g in assembly:
            para = g[list(g)[0]]
            if type(para) is dict:
                if para.get('_class_label') == '<neg>':
                    lay_name = para.get('name')
                    target.add_assembly(name=lay_name,
                                        assembly=self.load_layer(para))
                elif para.get('_class_label') == '<nod>':
                    nod_name = para.get('name')
                    target.add_assembly(name=nod_name,
                                        assembly=self.load_node(para))
                elif para.get('_class_label') == '<con>':
                    con_name = para.get('name')
                    target.add_connection(name=con_name,
                                          connection=self.load_connection(pnet=target, con=para))
                elif para.get('_class_label') == '<prj>':
                    prj_name = para.get('name')
                    target.add_projection(name=prj_name,
                                          projection=self.load_projection(prj=para))
            else:
                self.load_assembly(p_net=target, name=list(g)[0], assembly=para)
                # target.add_assembly(name=list(g)[0], assembly=self.load_assembly(list(g)[0], para))
        return target

    def load_layer(self, layer: dict):
        '''
            The function for load layer.

            Args:
                layer(dict): Data contains the parameters of layers.

            Return：
                NeuronGroup with need parameters.

        '''
        # layer.pop('_class_label')
        # parameters = self.trans_para(layer.get('parameters'))
        parameters = layer.get('parameters')
        for key, value in parameters.items():
            if isinstance(value, str):
                if value in self._diff_para.keys():
                    parameters[key] = self._diff_para[value]
        return_neuron = NeuronGroup(
            num=layer.get('num', 100),
            shape=layer.get('shape', [100]),
            neuron_type=layer.get('type', 'non_type'),
            neuron_position=layer.get('position', 'x, y, z'),
            model=layer.get('model_name', 'clif'),
            name=layer.get('name'),
            **parameters
        )
        return_neuron.id = layer.get('id', None)
        return return_neuron

    def load_connection(self, pnet, con: dict):
        '''
            The function for load connections,

            Args:
                con(dict): Data contains the parameters of connections.

            Return:
                Connection with needed parameters.

        '''
        if pnet._class_label == '<prj>':
            for pretarget in pnet.pre.get_groups():
                if con['pre'] == pretarget.id:
                    con['pre'] = pretarget
            for posttarget in pnet.post.get_groups():
                if con['post'] == posttarget.id:
                    con['post'] = posttarget
        else:
            for target in pnet.get_groups():
                if con['pre'] == target.id:
                    con['pre'] = target
                if con['post'] == target.id:
                    con['post'] = target

        for con_tar in ['pre', 'post']:
            if isinstance(con[con_tar], str):
                con[con_tar] = self.get_elements()[con[con_tar]] if con[con_tar] in self.get_elements().keys() else None

        assert (not isinstance(con['pre'], str) and con['pre'])
        assert (not isinstance(con['post'], str) and con['post'])

        if 'bias' in con['parameters'].keys():
            bias = con['parameters']['bias']
            if isinstance(con['parameters']['bias'], dict):
                if 'method' in con['parameters']['bias'].keys():
                    method = bias.get('method')
                    con['parameters']['bias'] = Initer.__dict__[method](**bias.get('para'))

        # con.pop('weight_path')
        return_conn = Connection(
            pre=con.get('pre'),
            post=con.get('post'),
            name=con.get('name'),
            link_type=con.get('link_type', 'full'),
            syn_type=con.get('synapse_type', ['basic_synapse']),
            max_delay=con.get('max_delay', 0),
            sparse_with_mask=con.get('sparse_with_mask', False),
            pre_var_name=con.get('pre_var_name', 'O'),
            post_var_name=con.get('post_var_name', 'WgtSum'),
            **con.get('parameters')
        )
        return_conn.id = con.get('id', None)
        return return_conn

    def load_projection(self, prj: dict):
        '''
            The function for load projection,

            Args:
                prj(dict): Data contains the parameters of projection.

            Return:
                Projection with needed parameters.

        '''
        if prj['pre'] in self._groups.keys() and \
                prj['post'] in self._groups.keys():
            prj['pre'] = self._groups[prj['pre']]
            prj['post'] = self._groups[prj['post']]
        else:
            print("Trans_error")
            print(self._groups.keys())

        assert not isinstance(prj['pre'], str)
        assert not isinstance(prj['post'], str)

        this_prj = Projection(
            pre=prj.get('pre'),
            post=prj.get('post'),
            name=prj.get('name'),
            link_type=prj.get('link_type', 'full'),
            # policies             = prj.get('policies', []),
            ConnectionParameters=prj.get('ConnectionParameters'),
        )

        for conn in prj['conns']:
            for key, value in conn.items():
                this_prj.add_connection(
                    con=self.load_connection(pnet=this_prj, con=value),
                    name=key,
                )

        # prj['policies'] = []
        # from spaic.Network.ConnectPolicy import IndexConnectPolicy, ExcludedTypePolicy, IncludedTypePolicy
        # policy_dict = {'Included_policy': IncludedTypePolicy,
        #                'Excluded_policy': ExcludedTypePolicy}
        #
        # for ply in prj['_policies']:
        #     if ply['name'] == 'Index_policy':
        #         prj['policies'].append(IndexConnectPolicy(pre_indexs=ply['pre_indexs'],
        #                                                   post_indexs=ply['post_indexs'],
        #                                                   level=ply['level']))
        #     else:
        #         prj['policies'].append(policy_dict[ply['name']](pre_types=ply['pre_types'],
        #                                                   post_types=ply['post_types'],
        #                                                   level=ply['level']))

        # con.pop('weight_path')
        return this_prj

    def load_node(self, node: dict):
        '''
            The function for load node like input or output.

            Args:
                node(dict): Data contains the parameters of nodes.

            Return:
                Node of input or output layer, contains needed parameters.

        '''

        Node_dict = {'<decoder>': Decoder, '<action>': Action, '<reward>': Reward,
                     '<generator>': Generator, '<encoder>': Encoder}

        if node.get('kind') == '<decoder>':
            return_node = Node_dict[node.get('kind')](
                num=node.get('num'),
                dec_target=self._groups.get(node.get('dec_target', None), None),
                dt=node.get('dt', 0.1),
                # time            = node.get('time'),
                coding_method=node.get('coding_method', 'poisson'),
                coding_var_name=node.get('coding_var_name', 'O'),
                node_type=node.get('type', None),
                **node.get('coding_param')
            )
        else:
            return_node = Node_dict[node.get('kind')](
                shape=node.get('shape', None),
                num=node.get('num'),
                dec_target=self._groups.get(node.get('dec_target', None), None),
                dt=node.get('dt', 0.1),
                # time            = node.get('time'),
                coding_method=node.get('coding_method', 'poisson'),
                coding_var_name=node.get('coding_var_name', 'O'),
                node_type=node.get('type', None),
                **node.get('coding_param')
            )
        return_node.id = node.get('id', None)
        return return_node

    def load_backend(self, backend=None, device=None, load_weight=False, data_type=None):
        '''
            The function for load backend parameters.

        '''

        key_parameters_list = ['dt', 'runtime', 'time', 'n_time_step']
        key_parameters_dict = ['_variables', '_parameters_dict']
        typical = ['_graph_var_dicts']

        import torch

        if backend is None:
            backend = Torch_Backend(device)
        self.set_backend(backend)
        self.set_backend_data_type(data_type=data_type)
        if self._backend_info:
            for key in key_parameters_list:
                self._backend.__dict__[key] = self._backend_info[key]
        self.build()

        if load_weight:
            for para_key in key_parameters_dict:
                path = self._backend_info[para_key]
                data = torch.load(path, map_location=self._backend.device0)
                for key, value in data.items():
                    # print(key, 'value:', value)
                    if key in self._backend.__dict__[para_key].keys():
                        if isinstance(self._backend.__dict__[para_key][key], torch.Tensor):
                            if 'device' in self._backend.__dict__[para_key][key].__dict__.keys():
                                target_device = self._backend.__dict__[para_key][key].device
                            else:
                                target_device = self._backend.device0
                            self._backend.__dict__[para_key][key] = value.to(target_device)
                        else:
                            self._backend.__dict__[para_key][key] = value

        return

    def set_backend_data_type(self, data_type=None):
        import torch
        supported_data_type = {'torch.float64': torch.float64,
                               'torch.float32': torch.float32,
                               'torch.float16': torch.float16,
                               'torch.bfloat16': torch.bfloat16,
                               'torch.int64': torch.int64,
                               'torch.int32': torch.int32,
                               'torch.int16': torch.int16,
                               'torch.bool': torch.bool,
                               'torch.uint8': torch.uint8}
        if data_type:
            self._backend.data_type = data_type
        else:
            if self._backend_info:
                self._backend.data_type = supported_data_type[self._backend_info['data_type']]
            else:
                self._backend.data_type = supported_data_type['torch.float32']

    def load_learner(self, learner: dict):
        '''
            The function for load learners' parameters.

        '''
        if '<net>' in learner['trainable']:  ## If self in net, use the whole net as the trainable target.
            learner.pop('trainable')
            builded_learner = Learner(
                algorithm=learner.get('algorithm'),
                trainable=self,
                **learner.get('parameters')
            )
        else:
            trainable_list = []
            for trains in learner['trainable']:
                if trains in self._groups:
                    trainable_list.append(self._groups[trains])
                elif trains in self._connections:
                    trainable_list.append(self._connections[trains])
            learner.pop('trainable')
            if learner.get('parameters'):
                if learner.get('parameters').get('pathway'):
                    pathway_target_list = []
                    for target_id in learner['parameters']['pathway']:
                        if target_id in self.get_elements():
                            pathway_target_list.append(self.get_elements()[target_id])
                        else:
                            for ctarget_key, ctarget in self._groups.items():
                                if ctarget.id == target_id:
                                    pathway_target_list.append(ctarget)
                                    break
                            for conn_key, conn in self._connections.items():
                                if conn.id == target_id:
                                    pathway_target_list.append(conn)
                                    break
                    learner['parameters']['pathway'] = pathway_target_list
                builded_learner = Learner(
                    trainable=trainable_list,
                    algorithm=learner.get('algorithm'),
                    **learner.get('parameters')
                )
            else:
                builded_learner = Learner(
                    trainable=trainable_list,
                    algorithm=learner.get('algorithm')
                )
        if learner.get('optim_name', None):
            builded_learner.set_optimizer(optim_name=learner.get('optim_name'),
                                          optim_lr=learner.get('optim_lr'),
                                          **learner.get('optim_para'))
        if learner.get('lr_schedule_name', None):
            builded_learner.set_schedule(lr_schedule_name=learner.get('lr_schedule_name'),
                                         **learner.get('lr_schedule_para'))

        return builded_learner

    def load_monitor(self, monitor: dict):
        '''
        Used to add monitors to the model according to the

        Args:
            monitor: a dict that contains monitors' information.


        '''
        monitor_dict = {'StateMonitor': StateMonitor,
                        'SpikeMonitor': SpikeMonitor}

        for name, mon in monitor.items():
            for target in self.get_groups():
                if mon['target'] == target.id:
                    mon['target'] = target
                    break
            for target in self.get_connections():
                if mon['target'] == target.id:
                    mon['target'] = target
                    break

            self.add_monitor(name=name,
                             monitor=monitor_dict[mon.get('monitor_type', 'StateMonitor')](
                                 target=mon['target'],
                                 var_name=mon['var_name'],
                                 dt=mon['dt'],
                                 get_grad=mon['get_grad'],
                                 nbatch=mon['nbatch'],
                                 index=mon['index']))

    def trans_para(self, para):
        if isinstance(para, dict):
            for key, value in para.items():
                para[key] = self.trans_para(value)
        else:
            para = torch.tensor(para, dtype=torch.float32, device=self.device[0])
        return para
