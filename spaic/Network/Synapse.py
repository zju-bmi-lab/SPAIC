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

        if conn.is_sparse:
            self._syn_operations.append([conn.post_var_name + '[post]', 'sparse_mat_mult_weight', 'weight[link]',
                                         self.input_name])
        elif conn.post.model_name == 'complex':
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

        if conn.bias_flag:
            self._syn_operations.append([conn.post_var_name + '[post]', 'add', conn.post_var_name + '[post]', 'bias[link]'])

# SynapseModel.register('basic_synapse', Basic_synapse)
SynapseModel.register('basic', Basic_synapse)

class conv_synapse(SynapseModel):
    """
    conv synapse
    Compute Isyn
    """

    def __init__(self, conn, **kwargs):
        super(conv_synapse, self).__init__(conn)
        if conn.post.model_name == 'complex':
            self._syn_operations.append(
                [conn.post_var_name + '[post]', 'conv_2d_complex', self.input_name, 'weight[link]',
                 'stride[pre]', 'padding[pre]', 'dilation[pre]', 'groups[pre]', 'complex_beta[post]'])
        else:
            self._syn_operations.append(
                [conn.post_var_name + '[post]', 'conv_2d', self.input_name, 'weight[link]',
                 'stride[pre]', 'padding[pre]', 'dilation[pre]', 'groups[pre]'])

        if conn.bias_flag:
            self._syn_operations.append(
                [conn.post_var_name + '[post]', 'conv_add_bias', conn.post_var_name + '[post]', 'bias[link]'])

# SynapseModel.register('conv_synapse', conv_synapse)
SynapseModel.register('conv', conv_synapse)

class DirectPass_synapse(SynapseModel):
    """
    DirectPass synapse
    target_name = input_name
    """

    def __init__(self, conn, **kwargs):
        super(DirectPass_synapse, self).__init__(conn)
        self._syn_operations.append([conn.post_var_name + '[post]', 'assign', self.input_name])

# SynapseModel.register('directpass_synapse', DirectPass_synapse)
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

# SynapseModel.register('dropout_synapse', Dropout_synapse)
SynapseModel.register('dropout', Dropout_synapse)

class AvgPool_synapse(SynapseModel):

    def __init__(self, conn, **kwargs):
        super(AvgPool_synapse, self).__init__(conn)
        if conn.pool_only:
            self._syn_operations.append([conn.post_var_name + '[post]', 'avg_pool2d', self.input_name,
                                         'pool_kernel_size[pre]', 'pool_stride[pre]', 'pool_padding[pre]'])
        else:
            if conn.pool_before:
                # when pooling before, the return operator name should be pre_var_name
                self._syn_operations.append([conn.pre_var_name + '[input]', 'avg_pool2d', self.input_name,
                                             'pool_kernel_size[pre]', 'pool_stride[pre]', 'pool_padding[pre]'])
            else:
                self._syn_operations.append([conn.post_var_name + '[post]', 'avg_pool2d', conn.post_var_name + '[post]',
                                             'pool_kernel_size[pre]', 'pool_stride[pre]', 'pool_padding[pre]'])

# SynapseModel.register('avgpool_synapse', AvgPool_synapse)
SynapseModel.register('avgpool', AvgPool_synapse)

class MaxPool_synapse(SynapseModel):

    def __init__(self, conn, **kwargs):
        super(MaxPool_synapse, self).__init__(conn)
        if conn.pool_only:
            self._syn_operations.append([conn.post_var_name + '[post]', 'avg_pool2d', self.input_name,
                                         'pool_kernel_size[pre]', 'pool_stride[pre]', 'pool_padding[pre]'])
        else:
            if conn.pool_before:
                # when pooling before, the return operator name should be pre_var_name
                self._syn_operations.append([conn.pre_var_name + '[input]', 'max_pool2d', self.input_name,
                                             'pool_kernel_size[pre]', 'pool_stride[pre]', 'pool_padding[pre]'])
            else:
                if conn.post.model_name == 'complex':
                    self._syn_operations.append(
                        [conn.post_var_name + '[post]', 'post_max_pool2d_complex', conn.post_var_name + '[post]',
                         'pool_kernel_size[pre]', 'pool_stride[pre]', 'pool_padding[pre]'])
                else:
                    self._syn_operations.append(
                        [conn.post_var_name + '[post]', 'max_pool2d', conn.post_var_name + '[post]',
                         'pool_kernel_size[pre]', 'pool_stride[pre]', 'pool_padding[pre]'])

# SynapseModel.register('maxpool_synapse', MaxPool_synapse)
SynapseModel.register('maxpool', MaxPool_synapse)

class BatchNorm2d_synapse(SynapseModel):

    def __init__(self, conn, **kwargs):
        super(BatchNorm2d_synapse, self).__init__(conn)
        if 'num_features' in conn.syn_kwargs.keys():
            num_features = conn.syn_kwargs['num_features']
        else:
            raise ValueError('The parameter num_features is not given.')

        if 'num_features' in conn.syn_kwargs.keys():
            num_features = conn.syn_kwargs['num_features']
        else:
            raise ValueError('The parameter num_features is not given. Set num_features in the syn_kwargs dict '
                             'initialization parameter of the Connecion class.')

        self._syn_constant_variables['num_features'] = num_features
        self._syn_operations.append([conn.post_var_name + '[post]', 'batchnorm2d', conn.post_var_name + '[post]',
                                     'num_features'])

# SynapseModel.register('BatchNorm2d_synapse', BatchNorm2d_synapse)
SynapseModel.register('batchnorm2d', BatchNorm2d_synapse)

class Flatten_synapse(SynapseModel):

    def __init__(self, conn, **kwargs):
        super(Flatten_synapse, self).__init__(conn)
        self._syn_constant_variables['view_dim'] = [-1, conn.pre_num]
        self._syn_operations.append([conn.pre_var_name + '[input]', 'view', self.input_name,
                                     'view_dim'])

# SynapseModel.register('flatten_synapse', Flatten_synapse)
SynapseModel.register('flatten', Flatten_synapse)


class Electrical_synapse(SynapseModel):
    """
    Electrical synapse
    Iele = weight *（V(l-1) - V(l)）
    """

    def __init__(self, conn, **kwargs):
        super(Electrical_synapse, self).__init__(conn)
        # V_post = conn.get_post_name(conn.post, 'V')
        # V_pre = conn.get_pre_name(conn.pre, 'V')
        # Vtemp_post = conn.get_link_name(conn.pre, conn.post, 'Vtemp')
        # I_post = conn.get_post_name(conn.post, 'I_ele')
        # weight = conn.get_link_name(conn.pre, conn.post, 'weight')
        # Vtemp_pre = conn.get_link_name(conn.post, conn.pre, 'Vtemp')
        # I_pre = conn.get_pre_name(conn.pre, 'I_ele')
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
        assert isinstance(conn.pre, NeuronGroup) and isinstance(conn.post,
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


# SynapseModel.register('electrical_synapse', Electrical_synapse)
SynapseModel.register('electrical', Electrical_synapse)

class First_order_chemical_synapse(SynapseModel):
    """
    .. math:: Isyn(t) = weight * e^{-t/tau}
    """
    def __init__(self, conn, **kwargs):
        super(First_order_chemical_synapse, self).__init__(conn)
        from ..Network.Connections import FullConnection
        assert isinstance(conn, FullConnection)
        self._syn_tau_variables['tau[link]'] = kwargs.get('tau', 2.0)
        self._syn_variables['R[link]'] = np.zeros([1, conn.post_num])
        self._syn_variables['WgtSum[link]'] = np.zeros([1, conn.post_num])

        if conn.post.model_name == 'complex':
            self._syn_operations.append(
                ['WgtSum[link]', 'mat_mult_weight_complex', conn.pre_var_name + '[input][updated]',
                 'weight[link]', 'complex_beta[post]'])
        else:
            self._syn_operations.append(['WgtSum[link]', 'mat_mult_weight', '[input]', 'weight[link]'])

        self._syn_operations.append(['R[link]', 'var_linear', 'tau[link]', 'R[link]', 'WgtSum[link][updated]'])
        self._syn_operations.append([conn.post_var_name + '[post]', 'assign', 'R[link][updated]'])


SynapseModel.register('1_order_synapse', First_order_chemical_synapse)


class Second_order_chemical_synapse(SynapseModel):
    """
    .. math:: Isyn(t) = weight*( e^{-t/tau_r}} - e^{-t/tau_d} )
    """

    def __init__(self, conn, **kwargs):
        super(Second_order_chemical_synapse, self).__init__(conn)
        from ..Network.Connections import FullConnection
        assert isinstance(conn, FullConnection)
        self._syn_tau_variables['tau_r[link]'] = kwargs.get('tau_r', 9.0)
        self._syn_tau_variables['tau_d[link]'] = kwargs.get('tau_d', 2.0)
        self._syn_variables['R[link]'] = np.zeros([1, conn.post_num])
        self._syn_variables['D[link]'] = np.zeros([1, conn.post_num])
        self._syn_variables['WgtSum[link]'] = np.zeros([1, conn.post_num])
        if conn.post.model_name == 'complex':
            self._syn_operations.append(
                ['WgtSum[link]', 'mat_mult_weight_complex', conn.pre_var_name + '[input][updated]',
                 'weight[link]', 'complex_beta[post]'])
        else:
            self._syn_operations.append(['WgtSum[link]', 'mat_mult_weight', '[input]', 'weight[link]'])
        self._syn_operations.append(['R[link]', 'var_linear', 'tau_r[link]', 'R[link]', 'WgtSum[link][updated]'])
        self._syn_operations.append(['D[link]', 'var_linear', 'tau_d[link]', 'D[link]', 'WgtSum[link][updated]'])
        self._syn_operations.append([conn.post_var_name + '[post]', 'minus', 'R[link][updated]', 'D[link][updated]'])

SynapseModel.register('2_order_synapse', Second_order_chemical_synapse)

import torch

class Mix_order_chemical_synapse(SynapseModel):

    def __init__(self, conn=None, **kwargs):
        super(Mix_order_chemical_synapse, self).__init__(conn)
        if conn is not None:
            from ..Network.Connections import FullConnection
            assert isinstance(conn, FullConnection)
            assert conn.post.model_name == 'complex'
            tau_r = kwargs.get('tau_r', 2.0)
            tau_s = kwargs.get('tau_s', 25.0)
            assert tau_r > 0 and tau_s > 0
            self._syn_tau_variables['tau_r[link]'] = tau_r
            self._syn_tau_variables['tau_s[link]'] = tau_s
            self._syn_constant_variables['alpha_r[link]'] = 1.0 / tau_r
            self._syn_constant_variables['alpha_s[link]'] = 1.0 / tau_s
            self._syn_variables['R[link]'] = np.zeros([1, conn.post_num])
            self._syn_variables['S[link]'] = np.zeros([1, conn.post_num])
            self._syn_operations.append([[conn.post_var_name + '[post]', 'R[link]', 'S[link]'], self.update,
                                        [conn.pre_var_name + '[input][updated]', 'weight[link]', 'complex_beta[post]',
                                         'R[link]', 'S[link]', 'tau_r[link]','tau_s[link]','alpha_r[link]',
                                         'alpha_s[link]','[dt]']])


    def update(self, inp, weight, complex_beta, R, S, beta_r, beta_s, alpha_r, alpha_s, dt):
        if inp.dtype.is_complex:
            x = inp.unsqueeze(-2)
            complex_beta = complex_beta.unsqueeze(-1)
            rate = x.real
            time = x.imag
            O = complex_beta ** time * (rate * (0 + 1.0j))
            ratio = rate*rate
            WgtSumR = torch.sum(ratio * O * weight, dim=-1)
            WgtSumS = torch.sum((1-ratio) * O * weight, dim=-1)
        else:
            weight = weight.permute(1, 0)
            WgtSumR = torch.matmul(inp, weight)
            WgtSumR = WgtSumR*(0.0+1.0j)
            WgtSumS = 0.0

        S = beta_s * S + alpha_s*WgtSumS
        R = beta_r*R + alpha_r*WgtSumR + alpha_r*S*dt
        return R, R, S
SynapseModel.register('mix_order_synapse', Mix_order_chemical_synapse)





