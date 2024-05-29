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
from .Assembly import Assembly
from collections import OrderedDict
from warnings import warn
from ..Backend.Backend import Backend
from ..Backend.Torch_Backend import Torch_Backend

try:
    import torch
except:
    pass


class Network(Assembly):
    _class_label = '<net>'

    def __init__(self, name=None):

        super(Network, self).__init__(name=name)
        self._monitors = OrderedDict()
        self._learners = OrderedDict()
        self._pipline = None
        self._backend: Backend = None
        self._forward_build = False
        pass

    # --------- Frontend code ----------
    def set_backend(self, backend=None, device='cpu', partition=False):
        if isinstance(device, str):
            device = [device]
        if backend is None:
            self._backend = Torch_Backend(device)
            self._backend.partition = partition
        elif isinstance(backend, Backend):
            self._backend = backend
        elif isinstance(backend, str):
            if backend == 'torch' or backend == 'pytorch':
                self._backend = Torch_Backend(device)
                self._backend.partition = partition
            # elif backend == 'tensorflow':
            #     self._backend = spaic.Tensorflow_Backend(device)

    def set_backend_dt(self, dt=0.1, partition=False):
        if self._backend is None:
            warn("have not set backend, default pytorch backend is set automatically")
            self._backend = Torch_Backend('cpu')
            self._backend.dt = dt
        else:
            self._backend.dt = dt
            self._backend.partition = partition

    def set_random_seed(self, seed):
        if isinstance(self._backend, Torch_Backend):
            import torch
            torch.random.manual_seed(int(seed))
            if self._backend.device == 'cuda':
                torch.cuda.manual_seed(int(seed))

    def get_testparams(self):
        self.all_Wparams = list()
        for key, value in self._backend._parameters_dict.items():
            self.all_Wparams.append(value)
        return self.all_Wparams

    def add_learner(self, name, learner):
        from ..Learning.Learner import Learner
        assert isinstance(learner, Learner)
        self.__setattr__(name, learner)

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

    def build(self, backend=None, strategy=0, full_enable_grad=None, device=None):
        if full_enable_grad is not None:
            self.enable_full_grad(full_enable_grad)
        if self._backend is None:
            if backend is not None:
                if device is not None:
                    self.set_backend(backend, device)
                else:
                    self.set_backend(backend)
            else:
                if device is not None:
                    self.set_backend(device=device)
                else:
                    self.set_backend()

        self._backend.clear_step()

        # build 试运行时，假设一个runtime
        if self._backend.runtime is None:
            self._backend.runtime = 10 * self._backend.dt

        all_groups = self.get_groups()
        for asb in all_groups:
            asb.set_id()

        self.build_projections(self._backend)

        all_connections = self.get_connections()

        # for debug
        con_debug = False
        con_syn_count = 0

        for con in all_connections:
            con.set_id()

            # ----根据连接，对每个神经元建立input_connection和output_connection
            con.pre.register_connection(con, True)
            con.post.register_connection(con, False)

        if strategy == 1:
            # 采取单纯的从头递归地build，一旦出现环路会陷入死循环，可以避开固有延迟的问题,
            # Use directly build strategy to avoid inherent delay. But cannot be used on models with loop, will fall in an endless loop.
            # Unfortunately,
            self._backend.forward_build = True
            self.forward_build(all_groups, all_connections)
        # elif strategy == 2:
        #     # 采取策略性构建，但是目前存在两个问题：
        #     #   1. 网络中存在Assembly块时会出现bug，尚未修复
        #     #   2. Connection所使用的input_spike为上一步的，需要添加[updated]，目前暂不使用所以未添加
        #     #   该构建方式可以较大程度上避开固有延迟的问题
        #     self.strategy_build(self.get_groups(False))
        else:
            # 原本的构建方式，首先构建连接，每个连接都是用上一轮神经元的输出脉冲，从而存在固有延迟的问题
            # 但是可以避开环路的问题。
            from multiprocessing.pool import ThreadPool as Pool
            self._backend.forward_build = False

            def build_fn(module):
                # if con_debug:
                #     con_syn_count += torch.count_nonzero(connection.weight.value).item()
                module.build(self._backend)

            # for connection in all_connections:
            #     connection.build(self._backend)
            #     if con_debug:
            #         import torch
            #         con_syn_count += torch.count_nonzero(connection.weight.value).item()

            # for group in all_groups:
            #     group.build(self._backend)    
            pool = Pool(4)
            pool.map(build_fn, all_connections)
            pool.close()
            pool.join()
            pool = Pool(4)
            pool.map(build_fn, all_groups)
            pool.close()
            pool.join()

        for learner in self._learners.values():
            learner.set_id()
            learner.build(self._backend)

        for monitor in self._monitors.values():
            monitor.build(self._backend)

        self._backend.build_graph()
        # self._backend.build()
        self._backend.builded = True
        # if con_debug:
        #     print("Connection synapses count:%d"%con_syn_count)

        # for group in all_groups:
        #     if hasattr(group, 'index'):
        #         group.index = 0

        pass

    def forward_build(self, all_groups=None, all_connections=None):
        builded_groups = []
        builded_connections = []
        nod_groups = []
        for group in all_groups.copy():
            if group._class_label == '<nod>':
                if (group._node_sub_class == '<encoder>') or (group._node_sub_class == '<generator>'):
                    group.build(self._backend)
                    builded_groups.append(group)
                    all_groups.remove(group)
                    for conn in group._output_connections:
                        self.deep_forward_build(conn, all_groups, all_connections, builded_groups, builded_connections)
                    for module in group._output_modules:
                        self.deep_forward_build(module, all_groups, all_connections, builded_groups,
                                                builded_connections)
                else:
                    all_groups.remove(group)
                    nod_groups.append(group)

        while all_groups or all_connections:
            for group in all_groups:
                self.deep_forward_build(group, all_groups, all_connections, builded_groups, builded_connections)
            for conn in all_connections:
                self.deep_forward_build(conn, all_groups, all_connections, builded_groups, builded_connections)

        for group in nod_groups:
            group.build(self._backend)
            builded_groups.append(group)

    def deep_forward_build(self, target, all_groups, all_connections, builded_groups, builded_connections):
        if (target in builded_groups) or (target in builded_connections):
            return
        if target._class_label == '<con>':
            pre = [target.pre]
            post = [target.post]
        elif target._class_label == '<neg>':
            pre = target._input_connections + target._input_modules
            post = target._output_connections + target._output_modules
        elif target._class_label == '<mod>':
            pre = target.input_targets.copy()
            post = target.output_targets.copy()
        else:
            raise ValueError("Deep forward build Error, unsupported class label.")

        for pr in pre:
            if (pr in all_groups) or (pr in all_connections):
                return

        target.build(self._backend)
        if target._class_label == '<con>':
            builded_connections.append(target)
            all_connections.remove(target)
        elif (target._class_label == '<neg>') or (target._class_label == '<mod>'):
            builded_groups.append(target)
            all_groups.remove(target)

        for po in post:
            self.deep_forward_build(po, all_groups, all_connections, builded_groups, builded_connections)

        return

    # def strategy_build(self, all_groups=None):
    #     builded_groups = []
    #     unbuild_groups = {}
    #     output_groups = []
    #     level = 0
    #     from ..Neuron.Node import Encoder, Decoder, Generator
    #     # ===================从input开始按深度构建计算图==============
    #     for group in all_groups:
    #         if isinstance(group, Encoder) or isinstance(group, Generator):
    #             # 如果是input节点，则开始深度构建计算图
    #             group.build(self._backend)
    #             builded_groups.append(group)
    #             # all_groups.remove(group)
    #             for conn in group._output_connections:
    #                 builded_groups, unbuild_groups = self.deep_build_conn(conn, builded_groups,
    #                                                                       unbuild_groups, level)
    #         elif isinstance(group, Decoder):
    #             # 如果节点是output节点，则放入output组在最后进行构建
    #             output_groups.append(group)
    #         else:
    #             if (not group._input_connections) and (not group._output_connections):
    #                 # 孤立点的情况
    #                 import warnings
    #                 warnings.warn('Isolated group occurs, please check the network.')
    #                 group.build(self._backend)
    #
    #     if unbuild_groups:
    #         import warnings
    #         warnings.warn('Loop occurs')
    #     # ====================开始构建环路==================
    #     for key in unbuild_groups.keys():
    #         for i in unbuild_groups[key]:
    #             if i in builded_groups:
    #                 continue
    #             else:
    #                 builded_groups = self.deep_build_neurongroup_with_delay(i, builded_groups)
    #
    #     # ====================构建output节点===============
    #     for group in output_groups:
    #         group.build(self._backend)
    #
    # def deep_build_neurongroup(self, neuron=None, builded_groups=None, unbuild_groups=None, level=0):
    #     conns = [i for i in neuron._input_connections if i not in builded_groups]
    #     # conns表示神经元还没有被建立的依赖连接
    #     if conns: #==========如果存在conns说明有input_connections还没有被build===========
    #         if str(level) in unbuild_groups.keys():
    #             unbuild_groups[str(level)].append(neuron)
    #         else:
    #             unbuild_groups[str(level)] = [neuron]
    #         return builded_groups, unbuild_groups
    #     else:
    #
    #         if neuron not in builded_groups:
    #             if neuron._class_label == '<asb>':
    #                 neuron.build(self._backend, strategy=2)
    #             else:
    #                 neuron.build(self._backend)
    #             builded_groups.append(neuron)
    #             for conn in neuron._output_connections:
    #                 builded_groups, unbuild_groups = self.deep_build_conn(conn, builded_groups,
    #                                                                       unbuild_groups, level)
    #         return builded_groups, unbuild_groups
    #
    # def deep_build_conn(self, conn=None, builded_groups=None, unbuild_groups=None, level=0):
    #     conn.build(self._backend)
    #     builded_groups.append(conn)
    #     level += 1
    #     builded_groups, unbuild_groups = self.deep_build_neurongroup(conn.post_assembly, builded_groups, unbuild_groups, level)
    #     return builded_groups, unbuild_groups
    #
    # def deep_build_conn_with_delay(self, conn, builded_groups):
    #     conn.build(self._backend)
    #     builded_groups.append(conn)
    #     if conn.post_assembly not in builded_groups:
    #         builded_groups = self.deep_build_neurongroup_with_delay(conn.post_assembly, builded_groups)
    #     return builded_groups
    #
    # def deep_build_neurongroup_with_delay(self, neuron, builded_groups):
    #     conns = [i for i in neuron._input_connections if i not in builded_groups]
    #     if conns:
    #         for conn in conns:
    #             conn.build(self._backend)
    #             builded_groups.append(conn)
    #         neuron.build(self._backend)
    #     else:
    #         neuron.build(self._backend)
    #     builded_groups.append(neuron)
    #     for conn in neuron._output_connections:
    #         if conn not in builded_groups:
    #             builded_groups = self.deep_build_conn_with_delay(conn, builded_groups)
    #     return builded_groups

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

    def enable_full_grad(self, requires_grad=True):
        self._backend.full_enable_grad = requires_grad

    def init_run(self):
        self._backend.initial_step()

    def add_monitor(self, name, monitor):
        from ..Monitor.Monitor import Monitor
        assert isinstance(monitor, Monitor), "Type Error, it is not monitor"
        assert monitor not in self._monitors.values(), "monitor %s is already added" % (name)
        assert name not in self._monitors.keys(), "monitor with name: %s have the same name with an already exists monitor" % (
            name)

        self.__setattr__(name, monitor)
        # self._monitors[name] = monitor

    def get_elements(self):
        element_dict = dict()
        for element in self.get_groups():
            element_dict[element.id] = element
        return element_dict

    def save_state(self, filename=None, direct=None, save=True, hdf5=False):
        """
        Save weights in memory or on hard disk.

        Args:
            filename: The name of saved file.
            direct: Target direction for saving state.
            mode: Determines whether saved in hard disk, default set false, it means will not save on disk.

        Returns:
            state: Connections' weight of the network.

        """
        from ..Neuron.Module import Module
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
            module_dict = {}
            module_exist = False
            for group in self.get_groups():
                if isinstance(group, Module):
                    module_dict[group.id] = group.state_dict
                    module_exist = True
            if module_exist:
                torch.save(module_dict, './module_dict.pt')
        os.chdir(origin_path)
        return

    def state_from_dict(self, state=False, filename=None, direct=None, device=None):
        """
        Reload states from memory or disk.

        Args:
            state: contains backend._parameters_dict .
            filename: The name of saved file.
            direct: Target direction for reloading state.
            mode: Determines whether saved in hard disk, default set false, it means will not save on disk.

        Returns:
            state: Connections' weight of the network.

        """
        from ..Neuron.Module import Module
        if not self._backend:
            if device:
                self.set_backend('torch', device=device)
            else:
                self.set_backend('torch')
        if self._backend.builded is False:
            self.build()
        if self._backend.device != device:
            import warnings
            warnings.warn(
                'Backend device setting is ' + str(self._backend.device) + '. Backend device selection is priority.')
            # device = self._backend.device
        if state:
            import torch
            if isinstance(state, dict) or isinstance(state, torch.Tensor):
                for key, para in state.items():
                    backend_key = self._backend.check_key(key, self._backend._parameters_dict)
                    if backend_key:
                        target_device = self._backend._parameters_dict[backend_key].device
                        self._backend._parameters_dict[backend_key] = para.to(target_device)

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
            data = torch.load('./_parameters_dict.pt', map_location=self._backend.device0)
            for key, para in data.items():
                backend_key = self._backend.check_key(key, self._backend._parameters_dict)
                if backend_key:
                    target_device = self._backend._parameters_dict[backend_key].device
                    self._backend._parameters_dict[backend_key] = para.to(target_device)
            if 'module_dict.pt' in os.listdir('./'):
                module_data = torch.load('./module_dict.pt', map_location=self._backend.device0)
                for group in self.get_groups():
                    if isinstance(group, Module):
                        target_key = self._backend.check_key(group.id, module_data)
                        group.load_state_dict(module_data[target_key])
        else:
            for file in os.listdir('./'):
                if file.endswith('.hdf5'):
                    import h5py
                    with h5py.File(direct, 'r') as f:
                        for key, para in f.items():
                            backend_key = self._backend.check_key(key, self._backend._parameters_dict)
                            if key:
                                target_device = self._backend._parameters_dict[backend_key].device
                                self._backend._parameters_dict[backend_key] = para.to(target_device)
        os.chdir(origin_path)
        return
