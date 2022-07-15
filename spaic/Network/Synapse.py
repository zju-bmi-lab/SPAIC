# encoding: utf-8
"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: Synapse.py
@time:2022/5/26 9:19
@description:
"""
from ..Network.Topology import SynapseModel
from ..Neuron.Neuron import NeuronGroup
import numpy as np

class Basic_synapse(SynapseModel):
    """
    Basic synapse
    Compute Isyn
    """

    def __init__(self, conn, **kwargs):
        super(Basic_synapse, self).__init__(conn)
        if conn.link_type == 'conv':
            # if conn.pool is not None:
            #     if conn.pool == 'avg_pool':
            #         self._syn_operations.append(
            #             [conn.post_var_name + '[post]', 'conv_avg_pool2d', conn.pre_var_name + '[input]',
            #              'weight[link]', 'pool_kernel_size[pre]', 'stride[pre]', 'pool_stride[pre]', 'padding[pre]',
            #              'pool_padding[pre]', 'dilation[pre]', 'groups[pre]'])
            #     elif conn.pool == 'max_pool':
            #         self._syn_operations.append(
            #             [conn.post_var_name+'[post]', 'conv_max_pool2d', conn.pre_var_name+'[input]',
            #              'weight[link]', 'pool_kernel_size[pre]', 'stride[pre]', 'pool_stride[pre]', 'padding[pre]',
            #              'pool_padding[pre]', 'dilation[pre]', 'groups[pre]'])
            #     else:
            #         raise ValueError()
            # else:
            self._syn_operations.append(
                [conn.post_var_name + '[post]', 'conv_2d', self.input_name, 'weight[link]',
                 'stride[pre]', 'padding[pre]', 'dilation[pre]', 'groups[pre]'])

            if conn.bias:
                self._syn_operations.append(
                    [conn.post_var_name + '[post]', 'conv_add_bias', conn.post_var_name + '[post]', 'bias[link]'])
        else:
            if conn.is_sparse:
                self._syn_operations.append([conn.post_var_name + '[post]', 'sparse_mat_mult_weight', 'weight[link]',
                                             self.input_name])
            elif conn.post_assembly.model_name=='complex':
                self._syn_operations.append(
                    [conn.post_var_name + '[post]', 'mat_mult_weight_complex', conn.pre_var_name + '[input][updated]',
                     'weight[link]', 'complex_beta[post]'])
            elif conn.max_delay > 0:
                self._syn_operations.append(
                    [conn.post_var_name + '[post]', 'mult_sum_weight', self.input_name, 'weight[link]'])
            else:
                self._syn_operations.append(
                    [conn.post_var_name + '[post]', 'mat_mult_weight', self.input_name,
                     'weight[link]'])

            if conn.bias:
                self._syn_operations.append([conn.post_var_name + '[post]', 'add', conn.post_var_name + '[post]', 'bias[link]'])

SynapseModel.register('basic_synapse', Basic_synapse)
SynapseModel.register('basic', Basic_synapse)

class DirectPass_synapse(SynapseModel):
    """
    DirectPass synapse
    target_name = input_name
    """

    def __init__(self, conn, **kwargs):
        super(DirectPass_synapse, self).__init__(conn)
        self._syn_operations.append([conn.post_var_name + '[post]', 'equal', self.input_name])

SynapseModel.register('directpass_synapse', DirectPass_synapse)
SynapseModel.register('directpass', DirectPass_synapse)

class Dropout_synapse(SynapseModel):
    """
    During training, randomly zeroes some of the elements of the input tensor with probability :
    attr:`p` using samples from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call.
    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """

    def __init__(self, conn, **kwargs):
        super(Dropout_synapse, self).__init__(conn)
        self._syn_constant_variables['p'] = conn.parameters.get('p', 0.5)
        self._syn_constant_variables['inplace'] = conn.parameters.get('inplace', False)
        self._syn_operations.append([conn.pre_var_name + '[input]', 'dropout', self.input_name, 'p',
                                     'inplace'])

SynapseModel.register('dropout_synapse', Dropout_synapse)
SynapseModel.register('dropout', Dropout_synapse)

class AvgPool_synapse(SynapseModel):

    def __init__(self, conn, **kwargs):
        super(AvgPool_synapse, self).__init__(conn)
        self._syn_operations.append([conn.pre_var_name + '[input]', 'avg_pool2d', self.input_name,
                                     'pool_kernel_size[pre]', 'pool_stride[pre]', 'pool_padding[pre]'])

SynapseModel.register('avgpool_synapse', AvgPool_synapse)
SynapseModel.register('avgpool', AvgPool_synapse)

class Flatten_synapse(SynapseModel):

    def __init__(self, conn, **kwargs):
        super(Flatten_synapse, self).__init__(conn)
        self._syn_constant_variables['view_dim'] = [-1, conn.pre_num]
        self._syn_operations.append([conn.pre_var_name + '[input]', 'view', self.input_name,
                                     'view_dim'])

SynapseModel.register('flatten_synapse', Flatten_synapse)
SynapseModel.register('flatten', Flatten_synapse)

class MaxPool_synapse(SynapseModel):

    def __init__(self, conn, **kwargs):
        super(MaxPool_synapse, self).__init__(conn)
        self._syn_operations.append([conn.pre_var_name + '[input]', 'max_pool2d', self.input_name,
                                     'pool_kernel_size[pre]', 'pool_stride[pre]', 'pool_padding[pre]'])

SynapseModel.register('maxpool_synapse', MaxPool_synapse)
SynapseModel.register('maxpool', MaxPool_synapse)


class Electrical_synapse(SynapseModel):
    """
    Electrical synapse
    Iele = weight *（V(l-1) - V(l)）
    """

    def __init__(self, conn, **kwargs):
        super(Electrical_synapse, self).__init__(conn)
        # V_post = conn.get_post_name(conn.post_assembly, 'V')
        # V_pre = conn.get_pre_name(conn.pre_assembly, 'V')
        # Vtemp_post = conn.get_link_name(conn.pre_assembly, conn.post_assembly, 'Vtemp')
        # I_post = conn.get_post_name(conn.post_assembly, 'I_ele')
        # weight = conn.get_link_name(conn.pre_assembly, conn.post_assembly, 'weight')
        # Vtemp_pre = conn.get_link_name(conn.post_assembly, conn.pre_assembly, 'Vtemp')
        # I_pre = conn.get_pre_name(conn.pre_assembly, 'I_ele')
        #
        # self._syn_variables[Vtemp_post] = 0.0
        # self._syn_variables[I_post] = 0.0
        # self._syn_variables[Vtemp_pre] = 0.0
        # self._syn_variables[I_pre] = 0.0
        # self._syn_operations.append([Vtemp_post, 'minus', V_pre, V_post])
        # self._syn_operations.append([I_post, 'var_mult', weight, Vtemp_post + '[updated]'])
        # self._syn_operations.append([Vtemp_pre, 'minus', V_post, V_pre])
        # self._syn_operations.append([I_pre, 'var_mult', weight, Vtemp_pre + '[updated]'])

        # self._syn_variables['Vprepost'] = np.zeros([1, conn.pre_num, conn.post_num])
        assert isinstance(conn.pre_assembly, NeuronGroup) and isinstance(conn.post_assembly,
                                                                         NeuronGroup), f"Electrical synapses exist in connections in which the presynaptic and postsynaptic objects are neurongroups"

        self._syn_variables['Isyn[post]'] = np.zeros([1, conn.post_num])
        self._syn_variables['Isyn[pre]'] = np.zeros([1, conn.pre_num])
        self._syn_constant_variables['unsequence_dim'] = 0
        self._syn_constant_variables['permute_dim'] = [1, 2, 0]
        self._syn_constant_variables['Vpre_permute_dim'] = [2, 1, 0]
        self._syn_constant_variables['post_sum_dim'] = 2
        self._syn_constant_variables['pre_sum_dim'] = 1

        # unsequence_dim_name =
        self._syn_operations.append(['Vpre', 'unsqueeze', 'V[pre]', 'unsequence_dim'])
        self._syn_operations.append(['Vpre_temp', 'permute', 'Vpre', 'Vpre_permute_dim'])
        self._syn_operations.append(['Vpost_temp', 'unsqueeze', 'V[post]', 'unsequence_dim'])
        # [pre_num, batch_size, post_num] [pre_num, batch_size, 1] [1, batch_size, post_num]
        self._syn_operations.append(['Vprepost', 'minus', 'Vpre_temp', 'Vpost_temp'])
        # [batch_size, post_num, pre_num]
        self._syn_operations.append(['Vprepost_temp', 'permute', 'Vprepost', 'permute_dim'])
        self._syn_operations.append(['I_post_temp', 'var_mult', 'Vprepost_temp', 'weight[link]'])
        # [batch_size, post_num]
        self._syn_operations.append(['Isyn[post]', 'reduce_sum', 'I_post_temp', 'post_sum_dim'])

        # [pre_num, batch_size, post_num]  [1, batch_size, post_num] [pre_num, batch_size, 1]
        self._syn_operations.append(['Vpostpre', 'minus', 'Vpost_temp', 'Vpre_temp'])
        # [batch_size, post_num, pre_num]
        self._syn_operations.append(['Vpostpre_temp', 'permute', 'Vpostpre', 'permute_dim'])
        self._syn_operations.append(['I_pre_temp', 'var_mult', 'Vpostpre_temp', 'weight[link]'])
        # [batch_size, pre_num]
        self._syn_operations.append(['Isyn[pre]', 'reduce_sum', 'I_pre_temp', 'pre_sum_dim'])


SynapseModel.register('electrical_synapse', Electrical_synapse)
SynapseModel.register('electrical', Electrical_synapse)

class First_order_chemical_synapse(SynapseModel):
    """
    .. math:: Isyn(t) = weight * e^{-t/tau}
    """
    def __init__(self, conn, **kwargs):
        super(First_order_chemical_synapse, self).__init__()
        from ..Network.Connections import FullConnection
        assert isinstance(conn, FullConnection)
        self._syn_tau_variables['tau[link]'] = kwargs.get('tau', 5.0)
        self._syn_variables['R[link]'] = np.zeros([1, conn.post_num])
        self._syn_variables['WgtSum[link]'] = np.zeros([1, conn.post_num])
        self._syn_operations.append(['WgtSum[link]', 'mat_mult_weight', '[input]', 'weight[link]'])
        self._syn_operations.append(['R[link]', 'var_linear', 'tau[link]', 'R[link]', 'WgtSum[link][updated]'])
        self._syn_operations.append([conn.post_var_name + '[post]', 'assign', 'R[link][updated]'])


SynapseModel.register('1_order_synapse', First_order_chemical_synapse)


class Second_order_chemical_synapse(SynapseModel):
    """
    .. math:: Isyn(t) = weight*( e^{-t/tau_r}} - e^{-t/tau_d} )
    """

    def __init__(self, conn, **kwargs):
        super(Second_order_chemical_synapse, self).__init__()
        from ..Network.Connections import FullConnection
        assert isinstance(conn, FullConnection)
        self._syn_tau_variables['tau_r[link]'] = kwargs.get('tau_r', 12.0)
        self._syn_tau_variables['tau_d[link]'] = kwargs.get('tau_d', 4.0)
        self._syn_variables['R[link]'] = np.zeros([1, conn.post_num])
        self._syn_variables['D[link]'] = np.zeros([1, conn.post_num])
        self._syn_variables['WgtSum[link]'] = np.zeros([1, conn.post_num])
        self._syn_operations.append(['WgtSum[link]', 'mat_mult_weight', '[input]', 'weight[link]'])
        self._syn_operations.append(['R[link]', 'var_linear', 'tau_r[link]', 'R[link]', 'WgtSum[link][updated]'])
        self._syn_operations.append(['D[link]', 'var_linear', 'tau_d[link]', 'D[link]', 'WgtSum[link][updated]'])
        self._syn_operations.append([conn.post_var_name + '[post]', 'minus', 'R[link][updated]', 'D[link][updated]'])


SynapseModel.register('2_order_synapse', Second_order_chemical_synapse)

import torch
class Delayed_complex_synapse(SynapseModel):
    def __init__(self, conn, **kwargs):
        super(Delayed_complex_synapse, self).__init__()
        self._backend = conn._backend
        self.delay = kwargs.get('delay', None)
        self.syn_tau = kwargs.get('tau', 10.0)
        self.syn_rotk = kwargs.get('rot_k', 6.0)
        self.delay_len = kwargs.get('delay_len', 1)
        self.max_delay_len = kwargs.get('max_delay_len', None)
        self.max_delay = kwargs.get('max_delay', 5.0)
        self.min_delay = kwargs.get('min_delay', 0.0)
        self.real_output = kwargs.get('real_output', True)
        self.weight_shape = list(conn.get_value('weight').shape)
        self.delay_shape = [self.delay_len] + self.weight_shape
        self.delay_buffer = torch.zeros(self.delay_shape, dtype=torch.cfloat).unsqueeze(0)
        self.Isyn = torch.zeros(self.weight_shape[0], dtype=torch.cfloat).unsqueeze(0)

        beta_s = np.exp(-self.dt/ self.syn_tau)
        rot_s = -self.dt*np.pi / (self.syn_rotk * self.syn_tau)
        # tmax = np.arctan(-self.syn_tau*rot_s)/-rot_s
        # vmax = np.exp(-tmax/self.syn_tau)*np.sin(-rot_s*tmax)
        self.v0 = 1.0/self.syn_tau
        self.complex_decay = torch.view_as_complex(torch.tensor([beta_s * np.cos(rot_s), beta_s * np.sin(rot_s)]))

        if self.delay is None:
            self.delay = self.min_delay + (self.max_delay-self.min_delay)*torch.rand(self.weight_shape, dtype=torch.float)
        elif self.max_delay is None:
            self.delay = torch.tensor(self.delay)
            self.max_delay = torch.amax(self.delay).item()
        if self.max_delay > self.syn_tau*self.syn_rotk:
            rot_d = -self.dt *np.pi / self.max_delay
        else:
            rot_d = rot_s
        self.complex_delay = torch.view_as_complex(torch.tensor([np.cos(rot_d), np.sin(rot_d)]))

        init_rot = -self.delay * rot_d / self.dt
        self.init_roting_complex = torch.view_as_complex(torch.stack([torch.cos(init_rot), torch.sin(init_rot)], dim=-1))

        self._syn_variables['delay_buffer[link]'] = self.delay_buffer
        self._syn_variables['Isyn[link]'] = self.Isyn
        self._syn_variables['init_roting_complex[link]'] = self.init_roting_complex
        self._syn_variables['complex_decay[link]'] = self.complex_decay
        self._syn_variables['complex_delay[link]'] = self.complex_delay

        if self.real_output:
            self._syn_operations.append([[conn.post_var_name+'[post]', 'delay_buffer[link]', 'Isyn[link]'], self.update_real_out,
                                         ['[input]','delay_buffer[link]', 'Isyn[link]', 'weight[link]', 'init_roting_complex[link]',
                                          'complex_delay[link]', 'complex_decay[link]']])
        else:
            self._syn_operations.append([[conn.post_var_name+'[post]', 'delay_buffer[link]', 'Isyn[link]'], self.update_complex_out,
                                         ['[input]','delay_buffer[link]', 'Isyn[link]', 'weight[link]', 'init_roting_complex[link]',
                                          'complex_delay[link]', 'complex_decay[link]']])


    @property
    def dt(self):
        return self._backend.dt


    def push_spike(self, spike, delay_buffer, init_roting_complex):
        if spike.dtype is torch.cfloat:
            vspk = spike.abs()
        else:
            vspk = spike

        vspk = torch.sum(vspk)
        if vspk > 0:
            if spike.dtype is not torch.cfloat:
                zero_real = torch.zeros_like(spike)
                spike = torch.view_as_complex(torch.stack((zero_real, spike), dim=-1))
            spike = spike.unsqueeze(-2)
            spike = self.v0*spike * init_roting_complex

            values, indices = torch.min(delay_buffer.abs(), dim=1)
            indices = indices.unsqueeze(1)
            spike = spike.unsqueeze(1)
            if torch.sum(values) > 0:
                delay_buffer = torch.cat([spike, delay_buffer], dim=1)
            else:
                delay_buffer.scatter_(dim=1, index=indices, src=spike)
        return delay_buffer

    def update_real_out(self, spike, delay_buffer, Isyn, weight, init_roting_complex, complex_delay, complex_decay):
        # push input spikes
        if spike.dtype is torch.cfloat:
            vspk = spike.abs()
        else:
            vspk = spike

        vspk = torch.sum(vspk)
        if vspk > 0:
            if spike.dtype is not torch.cfloat:
                zero_real = torch.zeros_like(spike)
                spike = torch.view_as_complex(torch.stack((zero_real, spike), dim=-1))
            spike = spike.unsqueeze(-2)
            spike = self.v0*spike * init_roting_complex

            values, indices = torch.min(delay_buffer.abs(), dim=1)
            indices = indices.unsqueeze(1)
            spike = spike.unsqueeze(1)
            if torch.sum(values) > 0:
                delay_buffer = torch.cat([spike, delay_buffer], dim=1)
            else:
                delay_buffer.scatter_(dim=1, index=indices, src=spike)

        # update and output
        delay_buffer = delay_buffer*complex_delay
        spike_mask = delay_buffer.real.gt(0.0)
        Isyn = Isyn*complex_decay + torch.sum(torch.sum(spike_mask*delay_buffer, 1)*weight, dim=-1)
        delay_buffer[spike_mask] = 0.0
        out = Isyn.real
        return out, delay_buffer, Isyn

    def update_complex_out(self, spike, delay_buffer, Isyn, weight, init_roting_complex, complex_delay, complex_decay):
        # push input spikes
        if spike.dtype is torch.cfloat:
            vspk = spike.abs()
        else:
            vspk = spike

        vspk = torch.sum(vspk)
        if vspk > 0:
            if spike.dtype is not torch.cfloat:
                zero_real = torch.zeros_like(spike)
                spike = torch.view_as_complex(torch.stack((zero_real, spike), dim=-1))
            spike = spike.unsqueeze(-2)
            spike = self.v0*spike * init_roting_complex

            values, indices = torch.min(delay_buffer.abs(), dim=1)
            indices = indices.unsqueeze(1)
            spike = spike.unsqueeze(1)
            if torch.sum(values) > 0:
                delay_buffer = torch.cat([spike, delay_buffer], dim=1)
            else:
                delay_buffer.scatter_(dim=1, index=indices, src=spike)

        # update and output
        delay_buffer = delay_buffer*complex_delay
        spike_mask = delay_buffer.real.gt(0.0)
        Isyn = Isyn*complex_decay + torch.sum(torch.sum(spike_mask*delay_buffer, 1)*weight, dim=-1)
        delay_buffer[spike_mask] = 0.0
        return Isyn, delay_buffer, Isyn

SynapseModel.register('delay_complex_synapse', Delayed_complex_synapse)