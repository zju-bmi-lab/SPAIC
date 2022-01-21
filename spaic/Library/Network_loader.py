# -*- coding: utf-8 -*-
"""
Created on 2020/8/17
@project: SNNFlow
@filename: Network_loader
@author: Mengxiao Zhang
@contact: mxzhangice@gmail.com

@description:
对已按格式储存网络的加载和重建
"""

import yaml
import json
from ..Network import Network, Connection
from ..Neuron.Neuron import NeuronGroup
from ..Neuron.Node import Node, Encoder, Decoder
from ..Learning.Learner import Learner
from ..Monitor.Monitor import StateMonitor
from ..Simulation.Torch_Backend import Torch_Backend

import torch


def network_load(filename=None, device='cuda:0'):
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
    path = filename.split('.')[0]
    origin_path = os.getcwd()
    os.chdir(os.getcwd()+'/NetData/'+path)
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = f.read()
            if data.startswith('{'):
                data = json.loads(data)
            else:
                data = yaml.load(data, Loader=yaml.FullLoader)

    else:
        if os.path.exists(filename+'.yml'):
            with open(filename+'.yml', 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)

        elif os.path.exists(filename+'.json'):
            with open(filename+'.json', 'r') as f:
                data = json.load(f)

        elif os.path.exists(filename+'.txt'):
            with open(filename+'.txt', 'r') as f:
                data = f.read()
                if data.startswith('{'):
                    data = json.loads(data)
                else:
                    data = yaml.load(data, Loader=yaml.FullLoader)

        else:
            raise ValueError("file %s doesn't exist, please check the "
                             "filename" % filename)

    net = ReloadedNetwork(net_data=data, device=device)

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
            simulator(simulator) : Backend that user want to use.
            learner(str) : The learning model of this network, will change in
                the future.
            learner_alpha(int) : The parameter alpha for learning model, will
                change in the future.
            device(str) : The type of device that run our model.

        Methods:
            load_net(self, data: dict) : The function for load the whole
                network, main function of this class.
            load_layer(self, layer: dict) : The function for load layer.
            load_connections(self, con: dict) : The function for load
                connections.
            load_node(self, node: dict) : The function for load node like input
                or output.
            load_simulator(self, path: str): The function for load simulator.

        Example:
            Net = ReloadNetwork(net_data, dataloader, 'poisson', simulator,
                'STCA', 0.5)

    '''
    def __init__(self, net_data: dict, simulator=None, device='cpu', sub_net=False):
        super(ReloadedNetwork, self).__init__()

        self.device = device
        self.name = list(net_data)[0]
        if simulator is None:
            simulator = Torch_Backend(device)
        self.load_net(net_data)

        if not sub_net:
            self.set_backend(simulator)
            # self._learner = Learner(algorithm='STCA', lr=0.5, trainable=self)
            self.build()

            self.load_simulator(device)

    def load_net(self, data: dict):
        '''
            The function for load the whole network, main function of this class.

            Args:
                data(dict) : The data should contains the network structure and
                    parameter from yaml.

        '''
        data = data[list(data)[0]]
        for g in data:
            para = g[list(g)[0]]
            if type(para) is dict:
                if para.get('_class_label') == '<neg>':
                    lay_name = para.get('name')
                    self.add_assembly(name=lay_name,
                                      assembly=self.load_layer(para))
                elif para.get('_class_label') == '<con>':
                    con_name = para.get('name')
                    self.add_connection(name=con_name,
                                        connection=self.load_connections(para))
                elif para.get('_class_label') == '<nod>':
                    nod_name = para.get('name')
                    self.add_assembly(name=nod_name,
                                      assembly=self.load_node(para))
                elif para.get('_class_label') == '<learner>':
                    learner = self.load_learner(para)
                    self._learners[para.get('name')] = learner
                else:
                    print('Unknown class label %d' % para['_class_label'])

                    break
            else:

                self.add_assembly(name=list(g)[0], assembly=ReloadedNetwork(
                                                    net_data=g, sub_net=True))

    def load_layer(self, layer: dict):
        '''
            The function for load layer.

            Args:
                layer(dict): Data contains the parameters of layers.

            Return：
                NeuronGroup with need parameters.

        '''
        # layer.pop('_class_label')
        return NeuronGroup(
            neuron_number   = layer.get('num', 100),
            neuron_shape    = layer.get('shape', [100]),
            neuron_type     = layer.get('type', None),
            neuron_position = layer.get('position', 'x, y, z'),
            neuron_model    = layer.get('model_name', 'clif'),
            name            = layer.get('name'),
            **layer.get('parameters')
        )

    def load_connections(self, con: dict):
        '''
            The function for load connections,

            Args:
                con(dict): Data contains the parameters of connections.

            Return:
                Connection with needed parameters.

        '''
        # con.pop('_class_label')
        if con['pre_assembly'] in self._groups.keys() and \
                con['post_assembly'] in self._groups.keys():
            con['pre_assembly']  = self._groups[con['pre_assembly']]
            con['post_assembly'] = self._groups[con['post_assembly']]
        else:
            print("Trans_error")
            print(self._groups.keys())

        # con.pop('weight_path')
        return Connection(
            pre_assembly    = con.get('pre_assembly'),
            post_assembly   = con.get('post_assembly'),
            name            = con.get('name'),
            link_type       = con.get('link_type', 'full'),
            policies        = con.get('_policies', []),
            max_delay       = con.get('max_delay', 0),
            sparse_with_mask= con.get('sparse_with_mask', False),
            pre_var_name    = con.get('pre_var_name', 'O'),
            post_var_name   = con.get('post_var_name', 'WgtSum'),
            **con.get('parameters')
        )

    def load_node(self, node: dict):
        '''
            The function for load node like input or output.

            Args:
                node(dict): Data contains the parameters of nodes.

            Return:
                Node of input or output layer, contains needed parameters.

        '''

        if node.get('dec_target'):  # output
            return Decoder(
                shape         = node.get('shape', None),
                num           = node.get('num'),
                dec_target    = self._groups.get(node.get('dec_target', None), None),
                # coding_time   = node.get('_time', 200.0),
                dt            = node.get('_dt', 0.1),
                coding_method = node.get('coding_method', 'poisson'),
                coding_var_name = node.get('coding_var_name', 'O'),
                node_type     = node.get('type', None),
            )
        else:  # input
            return Encoder(
                shape           = node.get('shape', None),
                num             = node.get('num'),
                dec_target      = self._groups.get(node.get('dec_target', None), None),
                # coding_time     = node.get('_time', 200.0),
                dt              = node.get('_dt', 0.1),
                coding_method   = node.get('coding_method', 'poisson'),
                coding_var_name = node.get('coding_var_name', 'O'),
                node_type       = node.get('type', None)
            )


    def load_simulator(self, device):
        '''
            The function for load simulator parameters.

        '''

        key_parameters_dict = ['_variables', '_parameters_dict', '_InitVariables_dict']
        key_parameters_list = ['dt', 'time', 'n_time_step']
        typical = ['_graph_var_dicts']

        import torch
        import os
        if os.path.exists('simulator.yml'):
            with open('simulator.yml', 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)

        elif os.path.exists('simulator.json'):
            with open('simulator.json', 'r') as f:
                data = json.load(f)
        else:
            import warnings
            warnings.warn('Backend data files not exists, does data with combined '
                          'net structure and weights?')
            return

        simulator = data[list(data)[0]]

        for key in key_parameters_list:
            self._simulator.__dict__[key] = simulator[key]

        for key in key_parameters_dict:
            for k, path in simulator[key].items():
                self._simulator.__dict__[key][k] = torch.load(path)

        self._simulator.__dict__[typical[0]]['variables_dict'] = \
            self._simulator.__dict__['_variables']

        for key, value in self._simulator.__dict__['_InitVariables_dict'].items():
            self._simulator.__dict__['_variables'][key] = value  # 这些变量的 requires_grad应该都是False
        for key, value in self._simulator.__dict__['_parameters_dict'].items():
            self._simulator.__dict__['_variables'][key] = value  # 这些变量的 requires_grad应该都是True

    def load_learner(self, learner:dict):
        '''
            The function for load learners' parameters.

        '''
        if '<net>' in learner['trainable']:
            learner.pop('trainable')
            builded_learner = Learner(
                algorithm = learner.pop('algorithm'),
                trainable = self,
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
            builded_learner = Learner(
                algorithm = learner.pop('algorithm'),
                trainable = trainable_list,
                **learner.get('parameters')
                )
        if learner.get('optim_name', None):
            builded_learner.set_optimizer(optim_name=learner.get('optim_name'),
                                      optim_lr=learner.get('optim_lr'),
                                      **learner.get('optim_para'))
        if learner.get('lr_schedule_name', None):
            builded_learner.set_schedule(lr_schedule_name=learner.get('lr_schedule_name'),
                                         **learner.get('lr_schedule_para'))

        return builded_learner