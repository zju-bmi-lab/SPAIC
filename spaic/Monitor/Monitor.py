# -*- coding: utf-8 -*-
"""
Created on 2020/8/12
@project: SPAIC
@filename: Monitor
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
定义神经集群放电以及神经元状态量、连接状态量的仿真记录模块
"""
from ..Network.Assembly import BaseModule, Assembly
from ..Network.Connection import Connection
from ..Backend.Backend import Backend
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Monitor(BaseModule):

    def __init__(self, target, var_name, index='full', dt=None, get_grad=False, nbatch=True):
        super().__init__()
        if isinstance(target, Assembly):
            self.target = target
            self.target_type = 'Assembly'
        elif isinstance(target, Connection):
            self.target = target
            self.target_type = 'Connection'
        elif target == None:
            self.target = None
            self.target_type = None
        else:
            raise ValueError("The target does not belong to types that can be watched (Assembly, Connection).")

        self.var_name = '{'+var_name+'}'
        self.index = index
        self.var_container = None
        self.get_grad = get_grad
        self.nbatch = nbatch
        self._nbatch_records = []      # all time window's record
        self._nbatch_times = []
        self._records = []   # single time window's record
        self._times = []
        self.dt = dt
        self.is_recording = True
        self.new_record = True

    def check_var_name(self, var_name):
        '''
        Check if variable is in the traget model, and add the target id label to the variable name.

        Parameters
        ----------
        var_name : original variable name

        Returns : modified variable name
        -------

        '''
        tar_var_name = None
        if var_name[1:-1] in self.backend._variables.keys():
            tar_var_name = var_name[1:-1]
        else:
            for tar_name in self.target.get_var_names():    # 没有中间变量
                if var_name in tar_name:
                    tar_var_name = tar_name
                    break

        if tar_var_name is not None:
            return tar_var_name
        else:
            raise ValueError(" Variable %s is not in the target model"%var_name)

    def get_str(self, level):
        pass
    def monitor_on(self):
        self.is_recording = True

    def monitor_off(self):
        self.is_recording = False

    def clear(self):
        NotImplementedError()

    def build(self, backend: Backend):
        NotImplementedError()


    def init_record(self):
        NotImplementedError()

    def update_step(self):
        NotImplementedError()

    def push_data(self, data, time):
        "push data to monitor by backend"
        self._records.append(data)
        self._times.append(time)




class SpikeMonitor(Monitor):
    def __init__(self, target, var_name='O', index='full', dt=None, get_grad=False, nbatch=False):
        super().__init__(target=target, var_name=var_name, index=index, dt=dt, get_grad=get_grad, nbatch=nbatch)
        self._transform_len = 0
        self._nbatch_index = []      # all time window's record
        self._nbatch_times = []
        self._spk_index = []
        self._spk_times = []
        self._records = []   # single time window's record
        self._times = []


    def build(self, backend: Backend):
        self.backend = backend
        self.backend._monitors.append(self)
        self.var_name = self.check_var_name(self.var_name)
        self.shape = self.backend._variables[self.var_name].shape
        if self.dt is None:
            self.dt = self.backend.dt

    def clear(self):
        self._transform_len = -1
        self._nbatch_index = []      # all time window's record
        self._nbatch_times = []
        self._spk_index = []
        self._spk_times = []
        self._records = []   # single time window's record
        self._times = []


    def init_record(self):
        self.new_record = True
        if len(self._spk_index) > 0:
            if self.nbatch is True:
                if isinstance(self._spk_index[0], torch.Tensor):
                    self._nbatch_index.append(torch.stack(self._spk_index[1:], dim=-1).cpu().detach().numpy())
                else:
                    self._nbatch_index.append(np.stack(self._spk_index[1:], axis=-1))
                self._nbatch_times.append(self._times[1:])
            elif self.nbatch > 0:
                if isinstance(self._spk_index[0], torch.Tensor):
                    self._nbatch_index.append(torch.stack(self._spk_index[1:], dim=-1).cpu().detach().numpy())
                else:
                    self._nbatch_index.append(np.stack(self._spk_index[1:], axis=-1))
                self._nbatch_times.append(self._times[1:])
                if len(self._nbatch_times) > self.nbatch:
                    self._nbatch_index = self._nbatch_index[-self.nbatch:]
                    self._nbatch_times = self._nbatch_times[-self.nbatch:]

        self._records = []   # single time window's record
        self._times = []
        self._transform_len = -1

    def push_spike_train(self, spk_times, spk_index, batch_index=0):
        if len(self._spk_index) < batch_index+1:
            add_num = batch_index + 1 - len(self._spk_index)
            for _ in range(add_num):
                self._spk_index.append([])
                self._spk_times.append([])
        if isinstance(spk_times, list) or isinstance(spk_times, tuple):
            self._spk_times[batch_index].extend(spk_times)
            self._spk_index[batch_index].extend(spk_index)
        else:
            self._spk_times[batch_index].append(spk_times)
            self._spk_index[batch_index].append(spk_index)

        #to override the _spike_transform function when getting spk_times and spk_index
        self._transform_len = 1


    def update_step(self, variables):
        '''
        Recoding the variable values of the current step.

        Returns
        -------

        '''
        if self.is_recording is False:
            return

        if int(10000 * self.backend.time / self.dt) % 10000 == 0:
            record_value = variables[self.var_name]
            if self.get_grad:
                variables[self.var_name].retain_grad()
            if self.index == 'full':
                self._records.append(record_value)
                self._times.append(self.backend.time)
            else:
                if len(self.index) == record_value.ndim:
                    self._records.append(record_value[self.index])
                    self._times.append(self.backend.time)
                else:
                    assert len(self.index) == record_value.ndim -1
                    if self.backend.backend_name == 'pytorch':
                        record_value = torch.movedim(record_value, 0, -1)
                        indexed_value = record_value[tuple(self.index)]
                        indexed_value = torch.movedim(indexed_value, -1, 0)
                    else:
                        record_value = np.array(record_value)
                        record_value = np.moveaxis(record_value, 0, -1)
                        indexed_value = record_value[tuple(self.index)]
                        indexed_value = np.moveaxis(indexed_value, -1, 0)
                    self._records.append(indexed_value)
                    self._times.append(self.backend.time)

    def _spike_transform(self):
        batch_size = self.backend.get_batch_size()
        if len(self._records) > self._transform_len:
            self._transform_len = len(self._records)
            self._spk_index = []
            self._spk_times = []
            if isinstance(self._records[0], torch.Tensor):
                step = len(self._records)
                rec_spikes = torch.stack(self._records, dim=-1).cpu().detach()
                if '{[2]' in self.var_name:
                    for ii in range(batch_size):
                        rec_spikes_i = rec_spikes[ii,0,...].bool().reshape(-1)
                        rec_spikes_t = rec_spikes[ii,1,...].reshape(-1)
                        num = int(rec_spikes_i.size(0)/step)
                        time_seq = torch.tensor(self._times).unsqueeze(dim=0).expand(num, -1).reshape(-1)
                        indx_seq = torch.arange(0, num).unsqueeze(dim=1).expand(-1, step).reshape(-1)
                        time_seq = (torch.masked_select(time_seq - rec_spikes_t, rec_spikes_i) ).numpy()
                        indx_seq = torch.masked_select(indx_seq, rec_spikes_i).numpy()
                        self._spk_index.append(indx_seq)
                        self._spk_times.append(time_seq)
                else:
                    for ii in range(batch_size):

                        rec_spikes_i = rec_spikes[ii,...].bool().reshape(-1)

                        num = int(rec_spikes_i.size(0)/step)
                        time_seq = torch.tensor(self._times).unsqueeze(dim=0).expand(num, -1).reshape(-1)
                        indx_seq = torch.arange(0, num).unsqueeze(dim=1).expand(-1, step).reshape(-1)
                        time_seq = torch.masked_select(time_seq, rec_spikes_i).numpy()
                        indx_seq = torch.masked_select(indx_seq, rec_spikes_i).numpy()
                        self._spk_index.append(indx_seq)
                        self._spk_times.append(time_seq)


    @property
    def spk_times(self):
        self._spike_transform()
        return self._spk_times

    @property
    def spk_index(self):
        self._spike_transform()
        return self._spk_index

    @property
    def spk_grad(self):
        pass
        return None

    @property
    def time_spk_rate(self):
        if isinstance(self._records[0], torch.Tensor):
            if '{[2]' in self.var_name:
                spike = torch.stack(self._records, dim=-1).cpu().detach()[:,0,...]
            else:
                spike = torch.stack(self._records, dim=-1).cpu().detach()
            return torch.mean(spike, dim=0).numpy()
        else:
            if '{[2]' in self.var_name:
                spike = np.stack(self._records, axis=-1)[:,0,...]
            else:
                spike = np.stack(self._records, axis=-1)
            return np.mean(spike, axis=0).numpy()


    @property
    def time(self):
        return np.stack(self._times, axis=-1)









class StateMonitor(Monitor):

    def __init__(self, target, var_name, index='full', dt=None, get_grad=False, nbatch=False):
        # TODO: 初始化有点繁琐，需要知道record的变量，考虑采用更直接的监控函数
        super().__init__(target=target, var_name=var_name, index=index, dt=dt, get_grad=get_grad, nbatch=nbatch)

        self._nbatch_records = []      # all time window's record
        self._nbatch_times = []
        self._records = []   # single time window's record
        self._times = []



    def build(self, backend: Backend):
        self.backend = backend
        self.backend._monitors.append(self)
        self.var_name = self.check_var_name(self.var_name)
        if self.index != 'full':
            self.index = tuple(self.index)
        if self.dt is None:
            self.dt = self.backend.dt


    def clear(self):
        self._nbatch_records = []      # all time window's record
        self._nbatch_times = []
        self._records = []   # single time window's record
        self._times = []


    def init_record(self):
        '''
        Inite record of new trial
        Returns:

        '''
        self.new_record = True
        if len(self._records) > 0:
            if self.nbatch is True:
                if isinstance(self._records[0], torch.Tensor):
                    self._nbatch_records.append(torch.stack(self._records, dim=-1).cpu().detach().numpy())
                else:
                    self._nbatch_records.append(np.stack(self._records, axis=-1))
                self._nbatch_times.append(self._times)
            elif self.nbatch > 0:
                if isinstance(self._records[0], torch.Tensor):
                    self._nbatch_records.append(torch.stack(self._records, dim=-1).cpu().detach().numpy())
                else:
                    self._nbatch_records.append(np.stack(self._records, axis=-1))
                self._nbatch_times.append(self._times)
                if len(self._nbatch_times) > self.nbatch:
                    self._nbatch_records = self._nbatch_records[-self.nbatch:]
                    self._nbatch_times = self._nbatch_times[-self.nbatch:]


            self._records = []
            self._times = []


    def update_step(self, variables):
        '''
        Recoding the variable values of the current step.

        Returns
        -------

        '''
        if self.is_recording is False:
            return

        # only data in variable_dict can be recorded now
        if int(10000 * self.backend.time / self.dt) % 10000 == 0:
            record_value = variables[self.var_name]
            if self.get_grad:
                var = variables[self.var_name]
                if var.requires_grad is True:
                    var.retain_grad()
            if self.index == 'full':
                self._records.append(record_value)
                self._times.append(self.backend.time)
            else:
                if len(self.index) == record_value.ndim:
                    self._records.append(record_value[self.index])
                    self._times.append(self.backend.time)
                else:
                    assert len(self.index) == record_value.ndim -1
                    if self.backend.backend_name == 'pytorch':
                        record_value = torch.movedim(record_value, 0, -1)
                        indexed_value = record_value[tuple(self.index)]
                        indexed_value = torch.movedim(indexed_value, -1, 0)
                    else:
                        record_value = np.array(record_value)
                        record_value = np.moveaxis(record_value, 0, -1)
                        indexed_value = record_value[tuple(self.index)]
                        indexed_value = np.moveaxis(indexed_value, -1, 0)
                    self._records.append(indexed_value)
                    self._times.append(self.backend.time)


    @property
    def nbatch_values(self):
        if self.new_record:
            self._nbatch_records_ = self._nbatch_records + [torch.stack(self._records, dim=-1).cpu().detach().numpy()]
            self._nbatch_times_ = self._nbatch_times + [self._times]
            self.new_record = False
        return np.array([np.stack(records, axis=-1) for records in self._nbatch_records_])

    @property
    def nbatch_times(self):
        if self.new_record:
            self._nbatch_records_ = self._nbatch_records + [torch.stack(self._records, dim=-1).cpu().detach().numpy()]
            self._nbatch_times_ = self._nbatch_times + [self._times]
            self.new_record = False
        return np.array([np.stack(times, axis=-1) for times in self._nbatch_times_])

    @property
    def values(self):
        # return np.concatenate(self._records)
        if isinstance(self._records[0], torch.Tensor):
            return torch.stack(self._records, dim=-1).cpu().detach().numpy()
        else:
            return np.stack(self._records, axis=-1)

    @property
    def grads(self):
        if self.get_grad:
            grads = []
            for v in self._records:
                if v.grad is not None:
                    grads.append(v.grad.cpu().numpy())
                else:
                    grads.append(torch.zeros_like(v).cpu().numpy())
            grads = np.stack(grads[1:], axis=-1)
            return grads
        else:
            return None


    @property
    def times(self):
        if isinstance(self._times[0], torch.Tensor):
            return torch.stack(self._times, dim=-1).cpu().detach().numpy()
        else:
            return np.stack(self._times, axis=-1)


    def plot_weight(self, **kwargs):
        neuron_id = kwargs.get('neuron_id')
        time_id = kwargs.get('time_id')
        batch_id = kwargs.get('batch_id')
        new_shape = kwargs.get('new_shape')
        reshape = kwargs.get('reshape')
        axes = kwargs.get('Axes', None)
        ims = kwargs.get('AxesImage', None)
        n_sqrt = kwargs.get('n_sqrt', None)
        side = kwargs.get('side', None)
        figsize = kwargs.get('figsize', (5, 5))
        cmap = kwargs.get('camp', 'hot_r')
        wmin = kwargs.get('wmin', 0)
        wmax = kwargs.get('wmax', 128)
        im = kwargs.get('im', None)

        if batch_id == None:
            value = self.values[:, :, time_id]
            # value = self.simulator._variables[
            #     'autoname1<net>_connection1<con>:autoname1<net>_layer1<neg><-autoname1<net>_input<nod>:{weight}']
            # value = value.cpu().detach().numpy()

            if reshape:

                value = value.reshape(n_sqrt, n_sqrt, side, side)

                value = value.transpose(0, 2, 1, 3)
                value = value.reshape(n_sqrt*side, n_sqrt*side)
                square_weights = value

            else:
                square_weights = value

        else:
            value = self.nbatch_values[batch_id, :, time_id, :]
            if reshape:
                value = value.reshape(n_sqrt, n_sqrt, side, side)

                value = value.transpose(0, 2, 1, 3)
                value = value.reshape(n_sqrt * side, n_sqrt * side)
                square_weights = value
            else:
                square_weights = value
        if not im:
            fig, ax = plt.subplots(figsize=figsize)

            im = ax.imshow(square_weights, cmap=cmap, vmin=wmin, vmax=wmax)
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="5%", pad=0.05)

            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_aspect("auto")

            plt.colorbar(im, cax=cax)
            fig.tight_layout()
        else:
            im.set_data(square_weights)

        plt.pause(0.1)
        return im

