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