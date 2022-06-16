# -*- coding: utf-8 -*-
"""
Created on 2020/8/5
@project: SPAIC
@filename: Network
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
定义网络以及子网络，网络包含所有的神经网络元素、如神经集群、连接以及学习算法、仿真器等，实现最终的网络仿真与学习。
执行过程：网络定义->网络生成->网络仿真与学习
"""
from spaic.Network.Assembly import Assembly
# from spaic.Network.Topology import Connection
from collections import OrderedDict
from warnings import warn
import spaic

class Network(Assembly):

    _class_label = '<net>'
    def __init__(self, name=None):

        super(Network, self).__init__(name=name)
        self._monitors = OrderedDict()
        self._learners = OrderedDict()
        self._pipline = None
        self._backend = None
        pass

    # --------- Frontend code ----------
    def set_backend(self, backend=None, device='cpu'):
        if backend is None:
            self._backend = spaic.Torch_Backend(device)
        elif isinstance(backend, spaic.Backend):
            self._backend = backend
        elif isinstance(backend, str):
            if backend == 'torch' or backend =='pytorch':
                self._backend = spaic.Torch_Backend(device)
            elif backend == 'tensorflow':
                self._backend = spaic.Tensorflow_Backend(device)

    def set_backend_dt(self, dt=0.1):
        if self._backend is None:
            warn("have not set backend, default pytorch backend is set automatically")
            self._backend = spaic.Torch_Backend('cpu')
            self._backend.dt = dt
        else:
            self._backend.dt = dt

    def set_random_seed(self, seed):
        if isinstance(self._backend, spaic.Torch_Backend):
            import torch
            torch.random.manual_seed(int(seed))
            if self._backend.device == 'cuda':
                torch.cuda.manual_seed(int(seed))

    def get_testparams(self):
        self.all_Wparams = list()
        for key, value in self._backend._parameters_dict.items():
            self.all_Wparams.append(value)
        return self.all_Wparams

    # TODO: 这里的setattr是否有必要要，是否可以全部放到Assembly里？
    def __setattr__(self, name, value):
        from ..Monitor.Monitor import Monitor
        from ..Learning.Learner import Learner
        super(Network, self).__setattr__(name, value)
        if isinstance(value, Monitor):
            self._monitors[name] = value

        elif isinstance(value, Learner):
            self._learners[name] = value

    # ---------  backend code  ----------

    def build(self, backend=None, strategy=0):
        if self._backend is None:
            if backend is not None:
                self.set_backend(backend)
            else:
                self.set_backend()

        self._backend.clear_step()

        # build 试运行时，假设一个runtime
        if self._backend.runtime is None:
            self._backend.runtime = 1.0

        all_groups = self.get_groups()
        for asb in all_groups:
            asb.set_id()

        self.build_projections(self._backend)

        all_connections = self.get_connections()


        for con in all_connections:
            con.set_id()

            #----根据连接，对每个神经元建立input_connection和output_connection
            con.pre_assembly.register_connection(con, True)
            con.post_assembly.register_connection(con, False)


        # if strategy == 1:
        #     # 采取单纯的从头递归地build，一旦出现环路会陷入死循环，可以避开固有延迟的问题
        #     self.forward_build(all_groups, all_connections)
        if strategy == 1:
            # 采取策略性构建，但是目前存在两个问题：
            #   1. 网络中存在Assembly块时会出现bug，尚未修复
            #   2. Connection所使用的input_spike为上一步的，需要添加[updated]，目前暂不使用所以未添加
            #   该构建方式可以较大程度上避开固有延迟的问题
            self.strategy_build(self.get_groups(False))
        else:
            # 原本的构建方式，首先构建连接，每个连接都是用上一轮神经元的输出脉冲，从而存在固有延迟的问题
            # 但是可以避开环路的问题。
            for connection in all_connections:
                connection.build(self._backend)

            for group in all_groups:
                group.build(self._backend)

        for monitor in self._monitors.values():
            monitor.build(self._backend)

        for learner in self._learners.values():
            learner.build(self._backend)

        self._backend.build_graph()
        # self._backend.build()
        self._backend.builded = True


        # for group in all_groups:
        #     if hasattr(group, 'index'):
        #         group.index = 0

        pass

    # def forward_build(self, all_groups=None, all_connections=None):
    #     builded_groups = []
    #     builded_connections = []
    #     for group in all_groups:
    #         if (group._class_label == '<nod>') and ('predict' not in dir(group)):
    #             group.build(self._backend)
    #             builded_groups.append(group)
    #             all_groups.remove(group)
    #     while all_groups or all_connections:
    #         for conn in all_connections:
    #             if conn.pre_assembly in builded_groups: # 如果连接的突触前神经元已经build，则可以build
    #                 conn.build(self._backend)
    #                 builded_connections.append(conn)
    #                 all_connections.remove(conn)
    #         for group in all_groups:
    #             can_build = 1
    #             if not all_connections:
    #                 group.build(self._backend)
    #                 builded_groups.append(group)
    #                 all_groups.remove(group)
    #             else:
    #                 for conn in all_connections:
    #                     if group == conn.post_assembly:
    #                         can_build = 0
    #                         break
    #                 if can_build:
    #                     group.build(self._backend)
    #                     builded_groups.append(group)
    #                     all_groups.remove(group)

    def strategy_build(self, all_groups=None):
        builded_groups = []
        unbuild_groups = {}
        output_groups = []
        level = 0
        from ..Neuron.Node import Encoder, Decoder, Generator
        # ===================从input开始按深度构建计算图==============
        for group in all_groups:
            if isinstance(group, Encoder) or isinstance(group, Generator):
                # 如果是input节点，则开始深度构建计算图
                group.build(self._backend)
                builded_groups.append(group)
                # all_groups.remove(group)
                for conn in group._output_connections:
                    builded_groups, unbuild_groups = self.deep_build_conn(conn, builded_groups,
                                                                          unbuild_groups, level)
            elif isinstance(group, Decoder):
                # 如果节点是output节点，则放入output组在最后进行构建
                output_groups.append(group)
            else:
                if (not group._input_connections) and (not group._output_connections):
                    # 孤立点的情况
                    import warnings
                    warnings.warn('Isolated group occurs, please check the network.')
                    group.build(self._backend)

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

        # ====================构建output节点===============
        for group in output_groups:
            group.build(self._backend)

    def deep_build_neurongroup(self, neuron=None, builded_groups=None, unbuild_groups=None, level=0):
        conns = [i for i in neuron._input_connections if i not in builded_groups]
        # conns表示神经元还没有被建立的依赖连接
        if conns: #==========如果存在conns说明有input_connections还没有被build===========
            if str(level) in unbuild_groups.keys():
                unbuild_groups[str(level)].append(neuron)
            else:
                unbuild_groups[str(level)] = [neuron]
            return builded_groups, unbuild_groups
        else:

            if neuron not in builded_groups:
                if neuron._class_label == '<asb>':
                    neuron.build(self._backend, strategy=2)
                else:
                    neuron.build(self._backend)
                builded_groups.append(neuron)
                for conn in neuron._output_connections:
                    builded_groups, unbuild_groups = self.deep_build_conn(conn, builded_groups,
                                                                          unbuild_groups, level)
            return builded_groups, unbuild_groups

    def deep_build_conn(self, conn=None, builded_groups=None, unbuild_groups=None, level=0):
        conn.build(self._backend)
        builded_groups.append(conn)
        level += 1
        builded_groups, unbuild_groups = self.deep_build_neurongroup(conn.post_assembly, builded_groups, unbuild_groups, level)
        return builded_groups, unbuild_groups

    def deep_build_conn_with_delay(self, conn, builded_groups):
        conn.build(self._backend)
        builded_groups.append(conn)
        if conn.post_assembly not in builded_groups:
            builded_groups = self.deep_build_neurongroup_with_delay(conn.post_assembly, builded_groups)
        return builded_groups

    def deep_build_neurongroup_with_delay(self, neuron, builded_groups):
        conns = [i for i in neuron._input_connections if i not in builded_groups]
        if conns:
            for conn in conns:
                conn.build(self._backend)
                builded_groups.append(conn)
            neuron.build(self._backend)
        else:
            neuron.build(self._backend)
        builded_groups.append(neuron)
        for conn in neuron._output_connections:
            if conn not in builded_groups:
                builded_groups = self.deep_build_conn_with_delay(conn, builded_groups)
        return builded_groups

    def run(self, backend_time):
        self._backend.set_runtime(backend_time)
        if self._backend.builded is False:
            self.build()
        self._backend.initial_step()
        self._backend.update_time_steps()

    def run_continue(self, backend_time):
        self._backend.set_runtime(backend_time)
        if self._backend.builded is False:
            self.build()
            self._backend.initial_step()
        self._backend.initial_continue_step()
        self._backend.update_time_steps()

    def reset(self, ):
        if self._backend.builded is True:
            self._backend.initial_step()

    def init_run(self):
        self._backend.initial_step()

    def add_monitor(self, name, monitor):
        from spaic.Monitor.Monitor import Monitor
        assert isinstance(monitor, Monitor), "Type Error, it is not monitor"
        assert monitor not in self._monitors.values(), "monitor %s is already added" % (name)
        assert name not in self._monitors.keys(), "monitor with name: %s have the same name with an already exists monitor" % (
            name)

        self.__setattr__(name, monitor)

    def save_state(self, filename=None, direct=None, save=True, hdf5=False):
        """
        Save weights in memory or on hard disk.

        Args:
            direct: Target direction for saving state.
            mode: Determines whether saved in hard disk, default set false, it means will not save on disk.

        Returns:
            state: Connections' weight of the network.

        """
        state = self._backend._parameters_dict
        if not save:
            return state
        if not filename:
            filename = self.name if self.name else 'autoname'
        if not direct:
            direct = './'
        file = filename.split('.')[0]
        path = direct + file + '/parameters/'
        import os
        import torch
        origin_path = os.getcwd()
        os.chdir(direct)
        if file not in os.listdir():
            os.mkdir(file)
        if 'parameters' not in os.listdir('./' + file):
            os.mkdir('./' + file + '/parameters')
            # os.mkdir('./NetData/' + dict + '/backend/_parameters_dict')
        os.chdir('./' + file + '/parameters')

        if hdf5:
            import h5py
            filename = filename if direct.endswith('.hdf5') else direct + '.hdf5'
            with h5py.File(direct, "w") as f:
                for i, item in enumerate(state):
                    f.create_dataset(item, data=self._backend._parameters_dict[item].cpu().detach().numpy())
                    # torch.save(self._backend._parameters_dict[item], os.getcwd()+'/'+str(i)+'.pt')
                    print(i, item, ': saved')
        else:
            torch.save(self._backend._parameters_dict, './_parameters_dict.pt')
        os.chdir(origin_path)
        return

    def state_from_dict(self, state=False, filename=None, direct=None, device=None):
        """
        Reload states from memory or disk.

        Args:
            state: contains backend._parameters_dict .
            direct: Target direction for reloading state.
            mode: Determines whether saved in hard disk, default set false, it means will not save on disk.

        Returns:
            state: Connections' weight of the network.

        """
        if not self._backend:
            self.set_backend('torch', device=device)
        if self._backend.builded is False:
            self.build()
        if self._backend.device != device:
            import warnings
            warnings.warn('Backend device setting is '+self._backend.device+'. Backend device selection is priority.')
            device = self._backend.device
        if state:
            import torch
            if isinstance(state, dict) or isinstance(state, torch.Tensor):
                for key, para in state.items():
                    backend_key = self._backend.check_key(key, self._backend._parameters_dict)
                    if key:
                        self._backend._parameters_dict[backend_key] = para.to(device)

                # if self._backend.device
                return
            else:
                raise ValueError("Given state has wrong type")

        if direct:
            if filename:
                path = direct + '/' + filename + '/parameters/'
            else:
                path = direct + '/parameters/'
        else:
            if filename:
                path = './' + filename + '/parameters/'
            else:
                path = './parameters/'

        import os
        import torch
        origin_path = os.getcwd()
        try:
            os.chdir(path)
        except:
            raise ValueError('Wrong Path.')

        if '_parameters_dict.pt' in os.listdir('./'):
            data = torch.load('./_parameters_dict.pt')
            for key, para in data.items():
                backend_key = self._backend.check_key(key, self._backend._parameters_dict)
                if backend_key:
                    self._backend._parameters_dict[backend_key] = para.to(device)
        else:
            for file in os.listdir('./'):
                if file.endswith('.hdf5'):
                    import h5py
                    with h5py.File(direct, 'r') as f:
                        for key, para in f.items():
                            backend_key = self._backend.check_key(key, self._backend._parameters_dict)
                            if key:
                                self._backend._parameters_dict[backend_key] = para.to(device)
        os.chdir(origin_path)
        return

    def train(self):
        pass

    def test(self):
        pass


