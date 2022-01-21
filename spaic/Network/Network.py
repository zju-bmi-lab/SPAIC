# -*- coding: utf-8 -*-
"""
Created on 2020/8/5
@project: SNNFlow
@filename: Network
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
定义网络以及子网络，网络包含所有的神经网络元素、如神经集群、连接以及学习算法、仿真器等，实现最终的网络仿真与学习。
执行过程：网络定义->网络生成->网络仿真与学习
"""
from .Assembly import Assembly
from .Connection import Connection
# from ..Neuron.Node import Encoder, Decoder
from collections import OrderedDict
import spaic

class Network(Assembly):

    _class_label = '<net>'
    def __init__(self, name=None):

        super(Network, self).__init__(name=name)
        self._monitors = OrderedDict()
        self._learners = OrderedDict()
        self._pipline = None
        self._simulator = None
        pass

    # --------- Frontend code ----------
    def set_backend(self, simulator=None, device='cpu'):
        if simulator is None:
            self._simulator = spaic.Torch_Backend(device)
        elif isinstance(simulator, spaic.Backend):
            self._simulator = simulator
        elif isinstance(simulator, str):
            if simulator in ['torch', 'pytorch']:
                self._simulator = spaic.Torch_Backend(device)
            elif simulator == 'tensorflow':
                self._simulator = spaic.Tensorflow_Backend(device)
            else:
                raise ValueError("Not such backend named %s"%simulator)

    def set_random_seed(self, seed):
        if isinstance(self._simulator, spaic.Torch_Backend):
            import torch
            torch.random.manual_seed(int(seed))
            if self._simulator.device == 'cuda':
                torch.cuda.manual_seed(int(seed))

    def get_testparams(self):
        self.all_Wparams = list()
        for key, value in self._simulator._parameters_dict.items():
            self.all_Wparams.append(value)
        return self.all_Wparams

    def __setattr__(self, name, value):
        from ..Monitor.Monitor import Monitor
        from ..Learning.Learner import Learner
        super(Network, self).__setattr__(name, value)
        if isinstance(value, Monitor):
            self._monitors[name] = value

        elif isinstance(value, Learner):
            self._learners[name] = value

    # ---------  backend code  ----------

    def build(self, simulator=None):
        if self._simulator is None:
            if simulator is not None:
                self.set_backend(simulator)
            else:
                self.set_backend()

        # build 试运行时，假设一个runtime
        if self._simulator.runtime is None:
            self._simulator.runtime = 10.0

        all_groups = self.get_groups()
        all_connections = self.get_connections()
        for asb in all_groups:
            asb.set_id()

        for con in all_connections:
            con.set_id()

        # if forward == 1:
        #     self.forward_build(all_groups, all_connections)
        # elif forward == 2:
        #     self.strategy_build(all_groups, all_connections)
        # else:
        for connection in all_connections:
            connection.build(self._simulator)

        for group in all_groups:
            group.build(self._simulator)

        # self.strategy_build(all_groups)

        for monitor in self._monitors.values():
            monitor.build(self._simulator)

        for learner in self._learners.values():
            learner.build(self._simulator)

        self._simulator.build_graph()
        self._simulator.builded = True
        # self._simulator.build()
        pass

    def forward_build(self, all_groups=None, all_connections=None):
        builded_groups = []
        builded_connections = []
        for group in all_groups:
            if (group._class_label == '<nod>') and ('predict' not in dir(group)):
                group.build(self._simulator)
                builded_groups.append(group)
                all_groups.remove(group)
        while all_groups or all_connections:
            for conn in all_connections:
                if conn.pre_assembly in builded_groups: # 如果连接的突触前神经元已经build，则可以build
                    conn.build(self._simulator)
                    builded_connections.append(conn)
                    all_connections.remove(conn)
            for group in all_groups:
                can_build = 1
                if not all_connections:
                    group.build(self._simulator)
                    builded_groups.append(group)
                    all_groups.remove(group)
                else:
                    for conn in all_connections:
                        if group == conn.post_assembly:
                            can_build = 0
                            break
                    if can_build:
                        group.build(self._simulator)
                        builded_groups.append(group)
                        all_groups.remove(group)

    def strategy_build(self, all_groups=None):
        builded_groups = []
        unbuild_groups = {}
        output_groups = []
        level = 0
        from ..Neuron.Node import Encoder, Decoder, Generator
        # ===================从input开始按深度构建计算图==============
        for group in all_groups:
            if isinstance(group, Encoder) or isinstance(group, Generator):
                group.build(self._simulator)
                builded_groups.append(group)
                # all_groups.remove(group)
                for conn in group._output_connections:
                    builded_groups, unbuild_groups = self.deep_build_conn(conn, builded_groups,
                                                                          unbuild_groups, level)
            elif isinstance(group, Decoder):
                output_groups.append(group)
            else:
                if (not group._input_connections) and (not group._output_connections):
                    import warnings
                    warnings.warn('Isolated group occurs, please check the network.')
                    group.build(self._simulator)
            # if group._class_label == '<nod>':
            #     if 'predict' not in dir(group):
            #         group.build(self._simulator)
            #         builded_groups.append(group)
            #         # all_groups.remove(group)
            #         for conn in group._output_connections:
            #             builded_groups, unbuild_groups = self.deep_build_conn(conn, builded_groups,
            #                                                                   unbuild_groups, level)
            #     else:
            #         output_groups.append(group)
            # else:
            #     if (not group._input_connections) and (not group._output_connections):
            #         import warnings
            #         warnings.warn('Isolated group occurs, please check the network.')
            #         group.build(self._simulator)
        # print('builded_groups: ', builded_groups, '\nunbuilded_groups: ', unbuild_groups)
        if unbuild_groups:
            import warnings
            warnings.warn('Loop occurs')
        # ====================开始构建环路==================
        for key in unbuild_groups.keys():
            for i in unbuild_groups[key]:
                if i in builded_groups:
                    continue
                else:
                    builded_groups = self.deep_build_neurongroup_with_delay(i, builded_groups)

        # print('builded_groups: ', builded_groups, '\nunbuilded_groups: ', unbuild_groups)

        # ====================构建output节点===============
        for group in output_groups:
            group.build(self._simulator)

    def deep_build_neurongroup(self, neuron=None, builded_groups=None, unbuild_groups=None, level=0):
        conns = [i for i in neuron._input_connections if i not in builded_groups]
        if conns: #==========如果存在conns说明有input_connections还没有被build===========
            if str(level) in unbuild_groups.keys():
                unbuild_groups[str(level)].append(neuron)
            else:
                unbuild_groups[str(level)] = [neuron]
            return builded_groups, unbuild_groups
        else:
            if neuron not in builded_groups:
                neuron.build(self._simulator)
                builded_groups.append(neuron)
                for conn in neuron._output_connections:
                    builded_groups, unbuild_groups = self.deep_build_conn(conn, builded_groups, unbuild_groups, level)
            return builded_groups, unbuild_groups

    def deep_build_conn(self, conn=None, builded_groups=None, unbuild_groups=None, level=0):
        conn.build(self._simulator)
        builded_groups.append(conn)
        level += 1
        builded_groups, unbuild_groups = self.deep_build_neurongroup(conn.post_assembly, builded_groups, unbuild_groups, level)
        return builded_groups, unbuild_groups

    def deep_build_conn_with_delay(self, conn, builded_groups):
        conn.build(self._simulator)
        builded_groups.append(conn)
        if conn.post_assembly not in builded_groups:
            builded_groups = self.deep_build_neurongroup_with_delay(conn.post_assembly, builded_groups)
        return builded_groups

    def deep_build_neurongroup_with_delay(self, neuron, builded_groups):
        conns = [i for i in neuron._input_connections if i not in builded_groups]
        if conns:
            for conn in conns:
                conn.build(self._simulator)
                builded_groups.append(conn)
            neuron.build(self._simulator)
        else:
            neuron.build(self._simulator)
        builded_groups.append(neuron)
        for conn in neuron._output_connections:
            # print("\nBuilding:", conn.name)
            if conn not in builded_groups:
                builded_groups = self.deep_build_conn_with_delay(conn, builded_groups)
        return builded_groups

    def run(self, simulation_time):
        self._simulator.set_runtime(simulation_time)
        if self._simulator.builded is False:
            self.build()

        self._simulator.initial_step()
        self._simulator.update_time_steps()
        
    def save_state(self, dict=None, mode=False):
        """
        Save weights in memory or on hard disk.

        Args:
            dict: Target direction for saving state.
            mode: Determines whether saved in hard disk, default set false, it means will not save on disk.

        Returns:
            state: Connections' weight of the network.

        """
        state = self._simulator._parameters_dict
        if not mode:
            return state
        path = '/NetData/' + dict + '/simulator/_parameters_dict/'
        import os
        import torch
        origin_path = os.getcwd()
        if 'NetData' not in os.listdir(os.getcwd()):
            os.mkdir('NetData')
        if dict not in os.listdir(os.getcwd() + '/NetData'):
            os.mkdir('./NetData/' + dict)
        if 'simulator' not in os.listdir(os.getcwd() + '/NetData/' + dict):
            os.mkdir('./NetData/' + dict + '/simulator')
            os.mkdir('./NetData/' + dict + '/simulator/_parameters_dict')
        os.chdir(os.getcwd() + path)

        import h5py
        dict = dict if dict.endswith('.hdf5') else dict + '.hdf5'
        with h5py.File(dict, "w") as f:
            for i, item in enumerate(state):
                f.create_dataset(item, data=self._simulator._parameters_dict[item].cpu().detach().numpy())
                # torch.save(self._simulator._parameters_dict[item], os.getcwd()+'/'+str(i)+'.pt')
                print(i, item, ': saved')
        os.chdir(origin_path)
        return dict

    def state_from_dict(self, dict=None, device='cpu'):
        path = '/NetData/' + dict + '/simulator/_parameters_dict/'
        import os
        import torch
        origin_path = os.getcwd()
        os.chdir(os.getcwd()+path)
        # state = self._simulator._parameters_dict
        import h5py
        dict = dict if dict.endswith('.hdf5') else dict+'.hdf5'
        with h5py.File(dict, 'r') as f:
            for i, item in enumerate(self._simulator._parameters_dict):
                self._simulator._parameters_dict[item] = torch.tensor(f[item].value, device=device)
                # self._simulator._parameters_dict[item] = torch.load(os.getcwd()+'/'+str(i)+'.pt')
                print('load:', i, item)
        os.chdir(origin_path)

    def train(self):
        pass

    def test(self):
        pass
