# -*- coding:  utf-8 -*-
"""
Created on 2020/8/5
@project: SPAIC
@filename: Connection
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
定义神经集群间的连接，包括记录神经元集群、连接的突触前、突触后神经元编号、连接形式（全连接、稀疏连接、卷积）、权值、延迟 以及连接产生函数、重连接函数等。
"""
from .Topology import Connection
from .Assembly import Assembly
import numpy as np
import scipy.sparse as sp
import torch
from ..IO.Initializer import BaseInitializer
import collections
import random

from matplotlib.pyplot import *
import math


def _pair(x):
    if x is None:
        return None
    elif isinstance(x, collections.abc.Iterable):
        return tuple(x)
    else:
        assert isinstance(x, int)
        return (x, x)


class FullConnection(Connection):
    '''
    each neuron in the first layer is connected to each neuron in the second layer.

    Args:
        pre(Assembly): The assembly which needs to be connected
        post(Assembly): The assembly which needs to connect the pre
        link_type(str): full
    '''

    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=['basic'], max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn',
                 syn_kwargs=None, **kwargs):
        if syn_type is None: syn_type = ['basic']
        super(FullConnection, self).__init__(pre=pre, post=post, name=name,
                                             link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                             sparse_with_mask=sparse_with_mask,
                                             pre_var_name=pre_var_name, post_var_name=post_var_name,
                                             syn_kwargs=syn_kwargs, **kwargs)
        weight = kwargs.get('weight', None)
        self.w_std = kwargs.get('w_std', 0.05)
        self.w_mean = kwargs.get('w_mean', 0.005)
        self.w_max = kwargs.get('w_max', None)
        self.w_min = kwargs.get('w_min', None)
        self.is_parameter = kwargs.get('is_parameter', True)
        self.is_sparse = kwargs.get('is_sparse', False)

        bias = kwargs.get('bias', None)
        if bias is None:
            self.bias_flag = False
        else:
            self.bias_flag = True
            if isinstance(bias, BaseInitializer):
                self.b_init, self.b_init_param = self.decode_initializer(bias)
                self.bias_value = np.empty(self.post_num)

            else:
                assert (bias.size == self.post_num), f"The size of the given bias {bias.shape} does not correspond" \
                                                     f" to the size of post_num {self.post_num} "
                self.bias_value = bias

            self._variables['bias[link]'] = self.bias_value

        if weight is None:
            # Connection weight
            if self.post.model_name == 'double_complex':
                self.shape = list(self.shape) + [12]  # 3*2 dimensional attention
                self.weight = self.w_std * np.random.randn(*self.shape) + self.w_mean
            else:
                self.weight = self.w_std * np.random.randn(*self.shape) + self.w_mean

        else:
            if isinstance(weight, BaseInitializer):
                self.w_init, self.w_init_param = self.decode_initializer(weight)
                self.weight = np.empty(self.shape)
            else:
                assert (weight.shape == self.shape), f"The size of the given weight {weight.shape} does not correspond" \
                                                     f" to the size of synaptic matrix {self.shape} "
                self.weight = weight

        self._variables['weight[link]'] = self.weight

        self.Isyn_bn = kwargs.get('Isyn_bn', False)
        if self.Isyn_bn is True:
            self.Is = []
            self.beta_ave = kwargs.get('beta_ave', 0.99)
            self.beta_std = kwargs.get('beta_std', 0.999)
            self.beta_bn = kwargs.get('beta_bn', 0.001)
            self.targ_ave = kwargs.get('targ_ave', 1.5)
            self.targ_std = kwargs.get('targ_std', 2.0)
            self.running_ave = self.targ_ave
            self.running_std = self.targ_std
            self.running_bias = 0.0
            self.running_scale = 1.0

            self._operations.append([None, self.update_ave, 'V' + '[post]'])
            self._init_operations.append(['weight[link]', self.update_bn, [post_var_name + '[post]', 'weight[link]']])

        # construct unit connection information by policies,
        # construct in __init__ is potentially bad, as network structure may change before build. should add new function
        # self.connection_inforamtion = ConnectInformation(self.pre, self.post)
        # self.connection_inforamtion.expand_connection()
        # for p in self._policies:
        #     self.connection_inforamtion = self.connection_inforamtion & p.generate_connection(self.pre, self.post)

    def update_ave(self, Isyn):
        with torch.no_grad():
            if Isyn.dtype.is_complex:
                self.Is.append(Isyn.imag.detach())
            else:
                self.Is.append(Isyn.detach())

    def update_bn(self, Isyn, weight):
        if self.training:
            weight.register_hook(self.norm_hook)
            with torch.no_grad():
                if len(self.Is) > 1:
                    Is = torch.cat(self.Is, dim=0)
                    mean_rate = torch.mean(Is, dim=0)
                    std = torch.std(Is, dim=0)
                    self.running_ave = self.beta_ave * self.running_ave + (1 - self.beta_ave) * mean_rate
                    self.running_std = self.beta_std * self.running_std + (1 - self.beta_std) * std

                    d_bias = (self.targ_ave - self.running_ave) * torch.sigmoid(
                        20.0 * (self.targ_ave - self.running_ave) * (self.running_ave - mean_rate))
                    self.running_bias = self.running_bias + self.beta_bn * d_bias
                    d_scale = torch.exp((self.targ_std - self.running_std)) - 1.0
                    self.running_scale = self.running_scale + self.beta_bn * d_scale
                    self.Is = []
                    self.weight_change = True
                else:
                    self.weight_change = False
                    self.running_scale = torch.ones_like(torch.mean(Isyn.real, dim=0))
                    self.running_bias = torch.zeros_like(torch.mean(Isyn.real, dim=0))
        else:
            self.Is = []

        if self.weight_change:
            weight = self.running_scale.unsqueeze(-1) * weight + self.running_bias.unsqueeze(-1)
            if weight.requires_grad:
                weight.retain_grad()

        return weight

    def norm_hook(self, grad):
        mean_grad = torch.mean(grad, dim=1, keepdim=True)
        return grad - mean_grad

    def condition_check(self, pre_group, post_group):
        flag = False
        pre_type = pre_group.type
        post_type = post_group.type
        if pre_type == post_type:
            flag = True
        return flag
        pass


Connection.register('full', FullConnection)
Connection.register('full_connection', FullConnection)


class one_to_one_sparse(Connection):
    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=['basic'], max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn',
                 syn_kwargs=None, **kwargs):
        if syn_type is None: syn_type = ['basic']
        super(one_to_one_sparse, self).__init__(pre=pre, post=post, name=name,
                                                link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                                sparse_with_mask=sparse_with_mask,
                                                pre_var_name=pre_var_name, post_var_name=post_var_name,
                                                syn_kwargs=syn_kwargs, **kwargs)
        try:
            assert self.pre_num == self.post_num
        except AssertionError:
            raise ValueError(
                'One to One connection must be defined in two groups with the same size, but the pre_num %s is not equal to the post_num %s.' % (
                    self.pre_num, self.post_num))
        weight = kwargs.get('weight', None)
        self.w_mean = kwargs.get('w_mean', 0.05)
        self.w_max = kwargs.get('w_max', None)
        self.w_min = kwargs.get('w_min', None)
        self.is_parameter = kwargs.get('is_parameter', False)
        self.is_sparse = kwargs.get('is_sparse', True)

        if weight is None:
            # Connection weight
            self.weight = self.w_mean * np.eye(*self.shape)
        else:
            if isinstance(weight, BaseInitializer):
                raise ValueError('Sparse implementation of one_to_one_sparse connection do not support the %s '
                                 'initialization method. You can pass a numpy matrix whose diagonal is not zero.',
                                 self.w_init)
            else:
                assert (weight.shape == self.shape), f"The size of the given weight {weight.shape} does not correspond" \
                                                     f" to the size of synaptic matrix {self.shape} "
                self.weight = weight
        self._variables['weight[link]'] = self.weight

        bias = kwargs.get('bias', None)
        if bias is None:
            self.bias_flag = False
        else:
            self.bias_flag = True
            if isinstance(bias, BaseInitializer):
                self.b_init, self.b_init_param = self.decode_initializer(bias)
                self.bias_value = np.empty(self.post_num)

            else:
                assert (bias.size == self.post_num), f"The size of the given bias {bias.shape} does not correspond" \
                                                     f" to the size of post_num {self.post_num} "
                self.bias_value = bias

            self._variables['bias[link]'] = self.bias_value

    def condition_check(self, pre_group, post_group):
        flag = False
        pre_type = pre_group.type
        post_type = post_group.type
        if pre_type == post_type:
            flag = True
        return flag
        pass


Connection.register('one_to_one_sparse', one_to_one_sparse)


class one_to_one_mask(Connection):
    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=['basic'], max_delay=0, sparse_with_mask=True, pre_var_name='O', post_var_name='Isyn',
                 syn_kwargs=None, **kwargs):
        if syn_type is None: syn_type = ['basic']
        super(one_to_one_mask, self).__init__(pre=pre, post=post, name=name,
                                              link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                              sparse_with_mask=sparse_with_mask, pre_var_name=pre_var_name,
                                              post_var_name=post_var_name, syn_kwargs=syn_kwargs, **kwargs)
        try:
            assert self.pre_num == self.post_num
        except AssertionError:
            raise ValueError('One to One connection must be defined in two groups with the same size, but '
                             'the pre_num %s is not equal to the post_num %s.' % (self.pre_num, self.post_num))
        weight = kwargs.get('weight', None)
        self.w_mean = kwargs.get('w_mean', 0.05)
        self.w_max = kwargs.get('w_max', None)
        self.w_min = kwargs.get('w_min', None)
        # self.init = kwargs.get('init', None)
        self.is_parameter = kwargs.get('is_parameter', True)
        self.is_sparse = kwargs.get('is_sparse', False)
        # self._variables = dict()

        if weight is None:
            # Connection weight
            self.weight = self.w_mean * np.eye(*self.shape)
        else:
            if isinstance(weight, BaseInitializer):
                raise ValueError('Dense implementation of one_to_one_mask connection do not support the %s '
                                 'initialization method. You can pass a numpy matrix whose diagonal is not zero.',
                                 self.w_init)
            else:
                assert (weight.shape == self.shape), f"The size of the given weight {weight.shape} does not correspond" \
                                                     f" to the size of synaptic matrix {self.shape} "
                self.weight = weight

        self._variables['weight[link]'] = self.weight

        bias = kwargs.get('bias', None)
        if bias is None:
            self.bias_flag = False
        else:
            self.bias_flag = True
            if isinstance(bias, BaseInitializer):
                self.b_init, self.b_init_param = self.decode_initializer(bias)
                self.bias_value = np.empty(self.post_num)

            else:
                assert (bias.size == self.post_num), f"The size of the given bias {bias.shape} does not correspond" \
                                                     f" to the size of post_num {self.post_num} "
                self.bias_value = bias

            self._variables['bias[link]'] = self.bias_value

    def condition_check(self, pre_group, post_group):
        flag = False
        pre_type = pre_group.type
        post_type = post_group.type
        if pre_type == post_type:
            flag = True
        return flag
        pass


Connection.register('one_to_one', one_to_one_mask)


class conv_connect(Connection):
    '''
    do the convolution connection.

    Args:
        pre(Assembly): the assembly which needs to be connected
        post(Assembly): the assembly which needs to connect the pre
        link_type(str): Conv
    Methods:
        unit_connect: define the basic connection information and add them to the connection_information.
        condition_check: check whether the pre_group.type is equal to the post_group.type, only if they are equal, return flag=Ture.

    '''

    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=['conv'], max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn',
                 syn_kwargs=None, **kwargs):
        if syn_type is None: syn_type = ['conv']
        super(conv_connect, self).__init__(pre=pre, post=post, name=name,
                                           link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                           sparse_with_mask=sparse_with_mask,
                                           pre_var_name=pre_var_name, post_var_name=post_var_name,
                                           syn_kwargs=syn_kwargs, **kwargs)
        self.out_channels = kwargs.get('out_channels', None)
        self.in_channels = kwargs.get('in_channels', None)
        self.kernel_size = kwargs.get('kernel_size', [3, 3])
        self.w_std = kwargs.get('w_std', 0.05)
        self.w_mean = kwargs.get('w_mean', 0.05)
        weight = kwargs.get('weight', None)

        self.is_parameter = kwargs.get('is_parameter', True)
        self.is_sparse = kwargs.get('is_sparse', False)
        self.mask = kwargs.get('mask', None)
        self.stride = kwargs.get('stride', 1)
        self.padding = kwargs.get('padding', 0)
        self.dilation = kwargs.get('dilation', 1)
        self.groups = kwargs.get('groups', 1)
        self.upscale = kwargs.get('upscale', None)

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        if isinstance(self.padding, int):
            self.padding = [self.padding] * 2
        # self._variables = dict()
        self._constant_variables['stride[link]'] = self.stride
        self._constant_variables['padding[link]'] = self.padding
        self._constant_variables['dilation[link]'] = self.dilation
        self._constant_variables['groups[link]'] = self.groups
        if self.upscale is not None:
            self._constant_variables['upscale[link]'] = self.upscale

        if self.in_channels is None:
            from ..Neuron.Encoders import NullEncoder
            if len(self.pre.shape) == 1:
                raise ValueError('The shape %s of pre seems to be a flattened value. '
                                 'Give the correct shape with structure [channel, height, width] in the NeuronGroup '
                                 'class with the neuron_shape parameter' % self.pre.shape)
            elif isinstance(self.pre, NullEncoder):
                self.in_channels = self.pre.shape[2]
            else:
                self.in_channels = self.pre.shape[0]

        if self.out_channels is None:
            if len(self.post.shape) == 1:
                raise ValueError('The shape %s of post seems to be a flattened value.  '
                                 'Give the correct shape with structure [channel, height, width] in the NeuronGroup '
                                 'class with the neuron_shape parameter' % self.post.shape)
            else:
                self.out_channels = self.post.shape[0]

        self.shape = (self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])

        if weight is None:
            # Connection weight
            self.weight = self.w_std * np.random.randn(*self.shape) + self.w_mean
        else:
            if isinstance(weight, BaseInitializer):
                self.w_init, self.w_init_param = self.decode_initializer(weight)
                self.weight = np.empty(self.shape)
            else:
                assert (weight.shape == self.shape), f"The size of the given weight {weight.shape} does not correspond" \
                                                     f" to the size of synaptic matrix {self.shape} "
                self.weight = weight

        self._variables['weight[link]'] = self.weight

        bias = kwargs.get('bias', None)
        if bias is None:
            self.bias_flag = False
        else:
            self.bias_flag = True
            if isinstance(bias, BaseInitializer):
                self.b_init, self.b_init_param = self.decode_initializer(bias)
                # import math
                # bound = math.sqrt(1 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1]))
                # self.bias_value = np.random.uniform(-bound, bound, self.out_channels)
                self.bias_value = np.empty(self.out_channels)

            else:
                # assert (bias.size == self.out_channels), f"The size of the given bias {bias.shape} does not correspond" \
                #                                          f" to the size of output_channels {self.out_channels} "
                self.bias_value = bias

            self._variables['bias[link]'] = self.bias_value

        Hin = self.pre.shape[-2]
        Win = self.pre.shape[-1]
        if self.upscale is not None:
            assert isinstance(self.upscale, int)
            Win = Win * self.upscale
            Hin = Hin * self.upscale

        pool_Flag = False
        if 'avgpool' in self.synapse_name:
            pool_Flag = True
            self.pool_index = self.synapse_name.index('avgpool')

        if 'maxpool' in self.synapse_name:
            pool_Flag = True
            self.pool_index = self.synapse_name.index('maxpool')

        self.conv_index = self.synapse_name.index('conv')

        # if self.pool is not None:  # 池化
        if pool_Flag:
            self.pool_only = False
            self.pool_kernel_size = kwargs.get('pool_kernel_size', [2, 2])
            self.pool_stride = kwargs.get('pool_stride', 2)
            self.pool_padding = kwargs.get('pool_padding', 0)
            if isinstance(self.pool_kernel_size, int):
                self.pool_kernel_size = [self.pool_kernel_size] * 2
            if isinstance(self.pool_stride, int):
                self.pool_stride = [self.pool_stride] * 2
            if isinstance(self.pool_padding, int):
                self.pool_padding = [self.pool_padding] * 2

            self._constant_variables['pool_kernel_size[link]'] = self.pool_kernel_size
            self._constant_variables['pool_stride[link]'] = self.pool_stride
            self._constant_variables['pool_padding[link]'] = self.pool_padding

            if self.pool_index < self.conv_index:
                # pooling before conv
                self.pool_before = True
                Ho = int((Hin + 2 * self.pool_padding[0] - self.pool_kernel_size[0]) / self.pool_stride[0] + 1)
                Wo = int((Win + 2 * self.pool_padding[1] - self.pool_kernel_size[1]) / self.pool_stride[1] + 1)
                Ho = int((Ho + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1)
                Wo = int((Wo + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1] + 1)
            else:
                # conv before pooling
                self.pool_before = False
                Ho = int((Hin + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[
                    0] + 1)  # Ho = (Hin + 2 * padding[0] - kernel_size[0]) / stride[0] + 1
                Wo = int((Win + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[
                    1] + 1)  # Wo = (Win + 2 * padding[0] - kernel_size[1]) / stride[0] + 1
                # Ho = int(Ho / self.pool_kernel_size[0])
                # Wo = int(Wo / self.pool_kernel_size[1])
                Ho = int((Ho + 2 * self.pool_padding[0] - self.pool_kernel_size[0]) / self.pool_stride[
                    0] + 1)  # Ho = (Hin + 2 * padding[0] - kernel_size[0]) / stride[0] + 1
                Wo = int((Wo + 2 * self.pool_padding[1] - self.pool_kernel_size[1]) / self.pool_stride[1] + 1)
        else:
            # without pooling
            Ho = int((Hin + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1)
            Wo = int((Win + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1] + 1)

        post_num = int(Ho * Wo * self.out_channels)

        if self.post.num == None:
            self.post.num = post_num
            self.post.shape = (self.out_channels, Ho, Wo)

        if self.post.num != None:
            if self.post.num != post_num:
                raise ValueError(
                    "The post_group num is not equal to the output num, cannot achieve the conv connection, "
                    "the output num is %d * %d * %d " % (self.out_channels, Ho, Wo))
            else:
                self.post.shape = (self.out_channels, Ho, Wo)

    def condition_check(self, pre_group, post_group):
        '''
        check whether the pre_group.type is equal to the post_group.type, only if they are equal, return flag=Ture.

        Args:
            pre_group(Groups): the neuron group which need to be connected in the pre.
            post_group(Groups): the neuron group which need to connect the pre_group in the post.

        Returns: flag

        '''

        flag = False
        pre_type = pre_group.type
        post_type = post_group.type
        if pre_type == post_type:
            flag = True
        return flag


Connection.register('conv', conv_connect)
Connection.register('conv_connection', conv_connect)


class ConvTranspose_connect(Connection):
    '''
    do the convolution connection.

    Args:
        pre(Assembly): the assembly which needs to be connected
        post(Assembly): the assembly which needs to connect the pre
        link_type(str): Conv
    Methods:
        unit_connect: define the basic connection information and add them to the connection_information.
        condition_check: check whether the pre_group.type is equal to the post_group.type, only if they are equal, return flag=Ture.

    '''

    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=['conv'], max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn',
                 syn_kwargs=None, **kwargs):
        if syn_type is None: syn_type = ['conv_transpose']
        super(ConvTranspose_connect, self).__init__(pre=pre, post=post, name=name,
                                                    link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                                    sparse_with_mask=sparse_with_mask,
                                                    pre_var_name=pre_var_name, post_var_name=post_var_name,
                                                    syn_kwargs=syn_kwargs, **kwargs)
        self.out_channels = kwargs.get('out_channels', None)
        self.in_channels = kwargs.get('in_channels', None)
        self.kernel_size = kwargs.get('kernel_size', [3, 3])
        self.w_std = kwargs.get('w_std', 0.05)
        self.w_mean = kwargs.get('w_mean', 0.05)
        weight = kwargs.get('weight', None)

        self.is_parameter = kwargs.get('is_parameter', True)
        self.is_sparse = kwargs.get('is_sparse', False)
        self.mask = kwargs.get('mask', None)
        self.stride = _pair(kwargs.get('stride', 1))
        self.padding = _pair(kwargs.get('padding', 0))
        self.dilation = _pair(kwargs.get('dilation', 1))
        self.groups = kwargs.get('groups', 1)

        # self._variables = dict()
        self._constant_variables['stride[link]'] = self.stride
        self._constant_variables['padding[link]'] = self.padding
        self._constant_variables['dilation[link]'] = self.dilation
        self._constant_variables['groups[link]'] = self.groups

        if self.in_channels is None:
            from ..Neuron.Encoders import NullEncoder
            if len(self.pre.shape) == 1:
                raise ValueError('The shape %s of pre seems to be a flattened value. '
                                 'Give the correct shape with structure [channel, height, width] in the NeuronGroup '
                                 'class with the neuron_shape parameter' % self.pre.shape)
            elif isinstance(self.pre, NullEncoder):
                self.in_channels = self.pre.shape[2]
            else:
                self.in_channels = self.pre.shape[0]

        if self.out_channels is None:
            if len(self.post.shape) == 1:
                raise ValueError('The shape %s of post seems to be a flattened value.  '
                                 'Give the correct shape with structure [channel, height, width] in the NeuronGroup '
                                 'class with the neuron_shape parameter' % self.post.shape)
            else:
                self.out_channels = self.post.shape[0]

        self.shape = (self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])

        if weight is None:
            # Connection weight
            self.weight = self.w_std * np.random.randn(*self.shape) + self.w_mean
        else:
            if isinstance(weight, BaseInitializer):
                self.w_init, self.w_init_param = self.decode_initializer(weight)
                self.weight = np.empty(self.shape)
            else:
                assert (weight.shape == self.shape), f"The size of the given weight {weight.shape} does not correspond" \
                                                     f" to the size of synaptic matrix {self.shape} "
                self.weight = weight

        self._variables['weight[link]'] = self.weight

        bias = kwargs.get('bias', None)
        if bias is None:
            self.bias_flag = False
        else:
            self.bias_flag = True
            if isinstance(bias, BaseInitializer):
                self.b_init, self.b_init_param = self.decode_initializer(bias)
                # import math
                # bound = math.sqrt(1 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1]))
                # self.bias_value = np.random.uniform(-bound, bound, self.out_channels)
                self.bias_value = np.empty(self.out_channels)

            else:
                assert (bias.size == self.out_channels), f"The size of the given bias {bias.shape} does not correspond" \
                                                         f" to the size of output_channels {self.out_channels} "
                self.bias_value = bias

            self._variables['bias[link]'] = self.bias_value

        Hin = self.pre.shape[-2]
        Win = self.pre.shape[-1]

        pool_Flag = False
        if 'avgpool' in self.synapse_name:
            pool_Flag = True
            self.pool_index = self.synapse_name.index('avgpool')

        if 'maxpool' in self.synapse_name:
            pool_Flag = True
            self.pool_index = self.synapse_name.index('maxpool')

        self.conv_index = self.synapse_name.index('conv')

        # if self.pool is not None:  # 池化
        if pool_Flag:
            raise NotImplementedError()
        else:
            # without pooling
            Ho = int(
                ((Hin - 1) * self.stride[0] - 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1)) + 1)
            Wo = int(
                ((Win - 1) * self.stride[0] - 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1)) + 1)

        post_num = int(Ho * Wo * self.out_channels)

        if self.post.num == None:
            self.post.num = post_num
            self.post.shape = (self.out_channels, Ho, Wo)

        if self.post.num != None:
            if self.post.num != post_num:
                raise ValueError(
                    "The post_group num is not equal to the output num, cannot achieve the conv connection, "
                    "the output num is %d * %d * %d " % (self.out_channels, Ho, Wo))
            else:
                self.post.shape = (self.out_channels, Ho, Wo)

    def condition_check(self, pre_group, post_group):
        '''
        check whether the pre_group.type is equal to the post_group.type, only if they are equal, return flag=Ture.

        Args:
            pre_group(Groups): the neuron group which need to be connected in the pre.
            post_group(Groups): the neuron group which need to connect the pre_group in the post.

        Returns: flag

        '''

        flag = False
        pre_type = pre_group.type
        post_type = post_group.type
        if pre_type == post_type:
            flag = True
        return flag


Connection.register('conv_transpose', conv_connect)


class ContinuousLocalConnection(Connection):
    '''
    Continuous local connection for 2D sheet of neurons that has 1 channel, and nearby neurons have similar
    connections.
    The conv parameter is fixed as stride=1, padding=0, dilation=1
    '''

    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=None, max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn',
                 syn_kwargs=None, **kwargs):
        if syn_type is None: syn_type = []
        super(ContinuousLocalConnection, self).__init__(pre=pre, post=post, name=name,
                                                        link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                                        sparse_with_mask=sparse_with_mask,
                                                        pre_var_name=pre_var_name, post_var_name=post_var_name,
                                                        syn_kwargs=syn_kwargs, **kwargs)
        self.w_std = kwargs.get('w_std', 0.5)
        self.w_mean = kwargs.get('w_mean', 0.05)
        self.w_min = kwargs.get('w_min', 0.0)
        self.w_max = kwargs.get('w_max', 0.1)
        self.kernel_size = _pair(kwargs.get('kernel_size', [3, 3]))
        self.clust_size = _pair(kwargs.get('clust_size', 1))
        self.dilation = _pair(kwargs.get('dilation', 1))
        # self.sigma_k = kwargs.get('sigma_k', 0.5*np.minimum(self.kernel_size[0], self.kernel_size[1]))
        self.sigma_c = kwargs.get('sigma_c', None)
        self.stride = _pair(kwargs.get('stride', 1))
        self.padding = _pair(kwargs.get('padding', 0))
        self.cutting = _pair(kwargs.get('cutting', 0))
        self._constant_variables['kernel_size[link]'] = self.kernel_size
        self._constant_variables['stride[link]'] = self.stride
        self._constant_variables['padding[link]'] = self.padding
        self._constant_variables['dilation[link]'] = self.dilation
        # the shape is fixed to (1, H, W), channel==1
        assert (len(self.pre.shape) == 3 or len(self.pre.shape) == 4) and len(self.post.shape) == 3
        assert self.pre.shape[0] == 1 and self.post.shape[0] == 1
        if len(self.pre.shape) == 3:
            self.Hin = self.pre.shape[1]
            self.Win = self.pre.shape[2]
        else:
            self.Hin = self.pre.shape[2]
            self.Win = self.pre.shape[3]
        self.raw_Ho = int(
            (self.Hin + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1)
        self.raw_Wo = int(
            (self.Win + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1)
        self.Ho = self.clust_size[0] * self.raw_Ho - 2 * self.cutting[0]
        self.Wo = self.clust_size[1] * self.raw_Wo - 2 * self.cutting[1]

        post_num = self.Ho * self.Wo
        if self.post.num == None:
            self.post.num = post_num
            self.post.shape = (self.Ho, self.Wo)
        else:
            if not (self.post.shape[1] == self.Ho and self.post.shape[2] == self.Wo):
                raise ValueError(f"neuron shape {self.post.shape[1:]} != connection out shape {self.Ho, self.Wo}")
            assert post_num == self.post.num
        self.bias_flag = False

        # parameters for continuous local STDP
        self.use_STDP = kwargs.get('use_STDP', False)
        self.use_RSTDP = kwargs.get('use_RSTDP', False)
        self.A_plus = kwargs.get('A_plus', 1.0e-3)
        self.A_minus = kwargs.get('A_minus', 1.0e-4)
        self.tau_pre = kwargs.get('tau_pre', 10.0)
        self.tau_post = kwargs.get('tau_post', 100.0)
        self.homeo_add = kwargs.get('homeo_add', 1.0e-4)
        self.homeo_mult = kwargs.get('homeo_mult', 1.0e-3)
        self.reward_name = kwargs.get('reward_name', 'Output_Reward[updated]')
        self.trace_pre = None
        self.trace_post = None

    def init_weight(self):
        import torch
        from torch.nn.functional import conv2d
        clust_size2d = self.clust_size[0] * self.clust_size[1]
        assert clust_size2d >= 1

        # build the cluster gaussian filter
        # hc_size = 1 + 2 * int(2.0*self.clust_size[0] // 2)
        # wc_size = 1 + 2 * int(2.0*self.clust_size[1] // 2)
        # h_c = torch.arange(-(hc_size // 2), hc_size // 2 + 1).view(-1, 1).expand(-1, wc_size)
        # w_c = torch.arange(-(wc_size // 2), wc_size // 2 + 1).view(1, -1).expand(hc_size, -1)

        # print(torch.sum(torch.abs(clust_gauss)))
        # imshow(clust_gauss[0,0,:,:])
        # show()

        # initialize the weight
        H_input_size = self.Hin + 2 * self.padding[0]
        W_input_szie = self.Win + 2 * self.padding[1]
        H_mesh_size = self.clust_size[0] * (self.kernel_size[0] * self.dilation[0]) // self.stride[0]
        W_mesh_size = self.clust_size[1] * (self.kernel_size[1] * self.dilation[1]) // self.stride[1]
        H_output_size = self.raw_Ho * self.clust_size[0]
        W_output_size = self.raw_Wo * self.clust_size[1]
        if self.sigma_c is not None:
            if (self.kernel_size[0] * self.dilation[0]) % self.stride[0] != 0 or \
                    (self.kernel_size[1] * self.dilation[1]) % self.stride[1] != 0:
                raise ValueError("Kernel_size must be integrate multiple of stride")

            self.sigma_c = 2 * int(self.sigma_c // 2) + 1
            weight = self.w_mean + self.w_std * torch.randn(H_input_size * W_input_szie, 1,
                                                            H_mesh_size + self.sigma_c - 1,
                                                            W_mesh_size + self.sigma_c - 1)

            h_f = torch.arange(-(self.sigma_c // 2), self.sigma_c // 2 + 1).view(-1, 1).expand(-1, self.sigma_c)
            w_f = torch.arange(-(self.sigma_c // 2), self.sigma_c // 2 + 1).view(1, -1).expand(self.sigma_c, -1)
            filt_r1 = (0.35 * self.sigma_c) ** 2.0
            filt_r2 = (0.10 * self.sigma_c) ** 2.0
            filter_DoG = (torch.exp(-(h_f ** 2 + w_f ** 2) / filt_r2) / filt_r2
                          - 1.0 * torch.exp(-(h_f ** 2 + w_f ** 2) / filt_r1) / filt_r1).view(1, 1, self.sigma_c,
                                                                                              self.sigma_c)
            self.clust_gauss = filter_DoG / torch.sum(filter_DoG ** 2) ** 0.5
            weight = conv2d(weight, filter_DoG)
            # print(torch.sum(self.clust_gauss**2)**0.5)
            # imshow(self.clust_gauss[0,0,:,:])
            # show()

            self.hw_indices = torch.zeros(self.kernel_size[0] * self.kernel_size[1], 1, H_output_size, W_output_size,
                                          dtype=torch.long)
            self.x_indices = torch.zeros(1, 1, H_output_size, W_output_size, dtype=torch.long)
            self.y_indices = torch.zeros(1, 1, H_output_size, W_output_size, dtype=torch.long)

            # x_indices, y_indices = torch.meshgrid([torch.arange(clust*raw_size), torch.arange(clust*raw_size)])

            for ii in range(H_output_size):
                for jj in range(W_output_size):
                    clust_stride_i = ii % H_mesh_size
                    kernel_i = ii // H_mesh_size
                    clust_stride_j = jj % W_mesh_size
                    kernel_j = jj // W_mesh_size
                    h_range = kernel_i * self.kernel_size[0] * self.dilation[0] + \
                              self.stride[0] * (clust_stride_i // self.clust_size[0]) + \
                              torch.arange(start=0, end=self.kernel_size[0] * self.dilation[0], step=self.dilation[0])
                    w_range = kernel_j * self.kernel_size[1] * self.dilation[1] + \
                              self.stride[1] * (clust_stride_j // self.clust_size[1]) + \
                              torch.arange(start=0, end=self.kernel_size[1] * self.dilation[1], step=self.dilation[1])
                    h_range, w_range = torch.meshgrid([h_range, w_range])
                    hw_range = h_range * W_input_szie + w_range
                    self.x_indices[0, 0, ii, jj] = clust_stride_i
                    self.y_indices[0, 0, ii, jj] = clust_stride_j
                    self.hw_indices[:, 0, ii, jj] = hw_range.view(-1)

            # imshow(self.x_indices[0, 0, :, :])
            # show()
            # imshow(self.y_indices[0, 0, :, :])
            # show()
            self.x_indices = self.x_indices[:, :, self.cutting[0]:H_output_size - self.cutting[0],
                             self.cutting[1]:W_output_size - self.cutting[1]]
            self.y_indices = self.y_indices[:, :, self.cutting[1]:W_output_size - self.cutting[1],
                             self.cutting[1]:W_output_size - self.cutting[1]]
            self.hw_indices = self.hw_indices[:, :, self.cutting[0]:H_output_size - self.cutting[0],
                              self.cutting[1]:W_output_size - self.cutting[1]]

            effective_weight = weight[self.hw_indices, 0, self.x_indices, self.y_indices]
            weight = torch.zeros_like(weight)
            weight[self.hw_indices, 0, self.x_indices, self.y_indices] = effective_weight

            self.shape = (H_input_size * W_input_szie, 1, H_mesh_size, W_mesh_size)
            self.H_input_size = H_input_size
            self.W_input_size = W_input_szie
            if (torch.max(self.hw_indices) >= self.shape[0]):
                print(f"wrong connection shape in {self.pre} to  {self.post}")

        else:
            weight = self.w_mean + self.w_std * torch.randn(self.kernel_size[0] * self.kernel_size[1], 1,
                                                            self.Ho, self.Wo)
            self.clust_gauss = None

            self.shape = (self.kernel_size[0] * self.kernel_size[1], 1,
                          self.Ho, self.Wo)

        # #normailze weight
        # weight = self.w_max*torch.sigmoid(self.w_mean + self.w_std*(weight - torch.mean(weight))/torch.std(weight))

        return weight

    def inti_conv_weight(self, weight):
        # from torch.nn.functional import conv2d
        if self.clust_gauss is not None:
            if weight.device != self.hw_indices.device:
                self.hw_indices = self.hw_indices.to(weight.device)
                self.x_indices = self.x_indices.to(weight.device)
                self.y_indices = self.y_indices.to(weight.device)
                self.clust_gauss = self.clust_gauss.to(weight.device)

            # con_weight = conv2d(weight, self.clust_gauss, padding=self.sigma_c // 2)
            # con_weight = torch.clamp(con_weight, self.w_min, self.w_max)
            # weight[:,:,:,:] = con_weight

            tmp_weight = weight[self.hw_indices, 0, self.x_indices, self.y_indices]
            tmp_mean = torch.mean(tmp_weight, dim=0, keepdim=True)
            tmp_weight = (tmp_weight + 0.2 * (self.w_mean - tmp_mean)) * (self.w_mean / tmp_mean) ** 0.5
            weight[self.hw_indices, 0, self.x_indices, self.y_indices] = tmp_weight
            weight.clamp_(self.w_min, self.w_max)
        # diff_mean = self.w_mean / (torch.mean(weight, dim=0, keepdim=True) + 1.0e-8)
        # weight.multiply_(diff_mean)

        # normailze weight
        # weight = self.w_max * torch.sigmoid(
        #     self.w_mean + 2*self.w_std * (weight - torch.mean(weight)) / torch.std(weight))
        # imshow(weight[100,0,:,:].detach().cpu())
        # show()

        return weight

    def local_connect_update(self, input, weight):
        import torch
        from torch.nn.functional import unfold

        # shape of input = (-1, 1, Hin, Win)
        # pad_input = pad(input, self.padding*2)
        unfold_input = unfold(input, self.kernel_size, padding=self.padding, stride=self.stride, dilation=self.dilation)
        # shape of unfold_input = (-1, self.kernel_size[0]*self.kernel_size[1], Ho*Wo)
        unfold_input = unfold_input.permute(1, 0, 2).reshape(
            self.kernel_size[0] * self.kernel_size[1], -1, self.raw_Ho, 1, self.raw_Wo, 1).expand(
            -1, -1, -1, self.clust_size[0], -1, self.clust_size[1]).reshape(self.kernel_size[0] * self.kernel_size[1],
                                                                            -1,
                                                                            self.raw_Ho * self.clust_size[0],
                                                                            self.raw_Wo * self.clust_size[1])
        unfold_input = unfold_input[:, :, self.cutting[0]:self.raw_Ho * self.clust_size[0] - self.cutting[0],
                       self.cutting[1]:self.raw_Wo * self.clust_size[1] - self.cutting[1]]
        if self.clust_gauss is not None:
            weight = weight[self.hw_indices, 0, self.x_indices, self.y_indices]
        output = torch.sum(unfold_input * weight, dim=0).unsqueeze(dim=1)
        return output

    def local_STDP_update(self, pre, post, weight):
        import torch
        from torch.nn.functional import unfold, conv2d
        if self.trace_pre is None:
            self.trace_pre = torch.zeros_like(pre)
            self.trace_post = torch.zeros_like(post)
            self.clust_gauss = self.clust_gauss.to(pre.device)

        # weight shape = (self.kernel_size[0]*self.kernel_size[1], 1,
        #                       self.Ho*self.clust_size[0], self.Wo*self.clust_size[1])
        # pre shape = (-1, 1, Hin, Win)
        # post shape = (-1, 1, Ho*clust_size[0], Wo*clust_size[1])
        self.trace_pre = self.beta_pre * self.trace_pre * pre.le(0.0) + pre
        self.trace_post = self.beta_post * self.trace_post * post.le(0.0) + post
        pre_reshape = torch.cat([pre, self.trace_pre], dim=0)  # ( 2*-1, 1, Hin, Win)
        pre_reshape = unfold(pre_reshape, self.kernel_size, padding=self.padding, stride=self.stride,
                             dilation=self.dilation)
        pre_reshape = pre_reshape.permute(1, 0, 2).reshape(
            self.kernel_size[0] * self.kernel_size[1], -1, self.Ho, 1, self.Wo, 1).expand(
            -1, -1, -1, self.clust_size[0], -1, self.clust_size[1]).reshape(self.kernel_size[0] * self.kernel_size[1],
                                                                            2, -1,
                                                                            self.Ho * self.clust_size[0],
                                                                            self.Wo * self.clust_size[1])
        trace_pre_reshape = pre_reshape[:, 1, :, :, :]
        pre_reshape = pre_reshape[:, 0, :, :, :]

        post_reshape = post.permute(1, 0, 2, 3).expand(self.kernel_size[0] * self.kernel_size[1], -1, -1, -1)
        trace_post_reshape = self.trace_post.permute(1, 0, 2, 3).expand(
            self.kernel_size[0] * self.kernel_size[1], -1, -1, -1)

        pre_post = trace_pre_reshape * post_reshape
        post_pre = trace_post_reshape * pre_reshape

        dw = torch.mean(self.A_plus * pre_post - self.A_minus * post_pre, dim=1, keepdim=True)
        if self.clust_gauss is not None:
            dw[self.hw_indices, 0, self.x_indices, self.y_indices] = dw
            dw = conv2d(dw, self.clust_gauss, padding=(self.hc_size // 2, self.wc_size // 2))

        # weight = torch.clamp(weight.add(dw),0,1)
        weight.add_(dw)
        weight.clamp_(0, 0.1)
        diff_mean = self.w_mean - torch.mean(weight, dim=0, keepdim=True)
        weight.multiply_(torch.exp(self.homeo_mult * diff_mean))
        weight.add_(self.homeo_add * diff_mean)
        return weight

    def local_RSTDP_update(self, pre, post, reward, weight):
        import torch
        from torch.nn.functional import unfold, conv2d
        if self.trace_pre is None:
            self.trace_pre = torch.zeros_like(pre)
            self.trace_post = torch.zeros_like(post)
            if self.clust_gauss is not None:
                self.clust_gauss = self.clust_gauss.to(pre.device)

        self.trace_pre = self.beta_pre * self.trace_pre + (1 - self.beta_pre) * pre
        self.trace_post = self.beta_post * self.trace_post + (1 - self.beta_post) * post
        pre_reshape = torch.cat([pre, self.trace_pre], dim=0)  # ( 2*-1, 1, Hin, Win)
        pre_reshape = unfold(pre_reshape, self.kernel_size, padding=self.padding, stride=self.stride,
                             dilation=self.dilation)
        pre_reshape = pre_reshape.permute(1, 0, 2).reshape(
            self.kernel_size[0] * self.kernel_size[1], -1, self.raw_Ho, 1, self.raw_Wo, 1).expand(
            -1, -1, -1, self.clust_size[0], -1, self.clust_size[1]).reshape(self.kernel_size[0] * self.kernel_size[1],
                                                                            2, -1,
                                                                            self.raw_Ho * self.clust_size[0],
                                                                            self.raw_Wo * self.clust_size[1])
        trace_pre_reshape = pre_reshape[:, 1, :, self.cutting[0]:self.raw_Ho * self.clust_size[0] - self.cutting[0],
                            self.cutting[1]:self.raw_Wo * self.clust_size[1] - self.cutting[1]]
        pre_reshape = pre_reshape[:, 0, :, self.cutting[0]:self.raw_Ho * self.clust_size[0] - self.cutting[0],
                      self.cutting[1]:self.raw_Wo * self.clust_size[1] - self.cutting[1]]

        post_reshape = post.permute(1, 0, 2, 3).expand(self.kernel_size[0] * self.kernel_size[1], -1, -1, -1)
        trace_post_reshape = self.trace_post.permute(1, 0, 2, 3).expand(
            self.kernel_size[0] * self.kernel_size[1], -1, -1, -1)

        pre_post = trace_pre_reshape * post_reshape
        post_pre = trace_post_reshape * pre_reshape
        post_dw = self.A_plus * pre_post - self.A_minus * post_pre
        dw = torch.mean(reward.unsqueeze(0).unsqueeze(-1) * post_dw, dim=1, keepdim=True)
        if self.clust_gauss is not None:
            trans_dw = torch.zeros_like(weight)
            trans_dw[self.hw_indices, 0, self.x_indices, self.y_indices] = dw

            dw = conv2d(trans_dw, self.clust_gauss, padding=self.sigma_c // 2)
            # dw = dw * (dw.gt(0)*torch.exp(-0.001*self.w_max/(self.w_max-weight+1.0e-5))
            #            + dw.lt(0)*torch.exp(-0.001*self.w_max/(weight+1.0e-5)))
            dw = dw * (dw.gt(0) * torch.exp(-0.02 * weight / self.w_max)
                       + dw.lt(0) * torch.exp(-0.02 * (self.w_max - weight) / self.w_max))
            weight.add_(dw)

            # weight = conv2d(weight, self.clust_gauss, padding=self.sigma_c // 2)
            # for ii in range(100):
            #     subplot(2,1,1)
            #     imshow(weight[100,0,:,:].detach().cpu().numpy())
            #     weight = conv2d(weight, self.clust_gauss, padding=self.sigma_c // 2)
            # subplot(2, 1, 2)
            # imshow(weight[100, 0, :,:].detach().cpu().numpy())
            # show()
        else:
            dw = dw * (dw.gt(0) * torch.exp(-0.01 * self.w_max / (self.w_max - weight + 1.0e-5))
                       + dw.lt(0) * torch.exp(-0.01 * self.w_max / (weight + 1.0e-5)))
            weight.add_(dw)
            # weight = conv2d(weight, self.clust_gauss, padding=self.sigma_c // 2)

            # weight = torch.clamp(weight.add(dw),0,1)

            diff_mean = self.w_mean - torch.mean(weight, dim=0, keepdim=True)
            # weight.multiply_(diff_mean)
            weight.add_(self.homeo_add * diff_mean)
            weight.clamp_(self.w_min, self.w_max)
        # imshow(weight[100, 0, :,:].detach().cpu().numpy())
        # show()
        return weight

    def build(self, backend):
        from .BaseModule import Op
        self.weight = self.init_weight()
        self._variables['weight[link]'] = self.weight
        super(ContinuousLocalConnection, self).build(backend)
        update_op = Op(owner=self)
        update_op.func_name = self.local_connect_update
        update_op.output = self.add_conn_label(self.post_var_name + '[post]')
        update_op.input = [self.add_conn_label(self.pre_var_name + '[pre]'), self.add_conn_label('weight[link]')]
        backend.add_operation(update_op)
        self._operations.append(update_op)
        weightConv_op = Op(owner=self)
        weightConv_op.func_name = self.inti_conv_weight
        weightConv_op.output = self.add_conn_label('weight[link]')
        weightConv_op.input = self.add_conn_label('weight[link]')
        self._init_operations.append(weightConv_op)
        backend.register_initial(weightConv_op)

        if self.use_STDP:
            self.beta_pre = np.exp(-backend.dt / self.tau_pre)
            self.beta_post = np.exp(-backend.dt / self.tau_post)
            STDP_op = Op(owner=self)
            STDP_op.func_name = self.local_STDP_update
            STDP_op.output = self.add_conn_label('weight[link]')
            STDP_op.input = [self.add_conn_label(self.pre_var_name + '[pre]'),
                             self.add_conn_label('O[post]'),
                             self.add_conn_label('weight[link]')]
            backend.add_operation(STDP_op)
            self._operations.append(STDP_op)
        elif self.use_RSTDP:
            self.beta_pre = np.exp(-backend.dt / self.tau_pre)
            self.beta_post = np.exp(-backend.dt / self.tau_post)
            RSTDP_op = Op(owner=self)
            RSTDP_op.func_name = self.local_RSTDP_update
            RSTDP_op.output = self.add_conn_label('weight[link]')
            RSTDP_op.input = [self.add_conn_label(self.pre_var_name + '[pre]'),
                              self.add_conn_label('O[post]'),
                              self.reward_name,
                              self.add_conn_label('weight[link]')]
            backend.add_operation(RSTDP_op)
            self._operations.append(RSTDP_op)


Connection.register('continuous_local', ContinuousLocalConnection)


class ComplexAttentionConnection(Connection):
    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=None, max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn',
                 syn_kwargs=None, **kwargs):
        if syn_type is None: syn_type = []
        super(ComplexAttentionConnection, self).__init__(pre=pre, post=post, name=name,
                                                         link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                                         sparse_with_mask=sparse_with_mask,
                                                         pre_var_name=pre_var_name, post_var_name=post_var_name,
                                                         syn_kwargs=syn_kwargs, **kwargs)

        self.group_num = kwargs.get('group_num', 1)  # multi_head number
        self.d_k = kwargs.get('d_k', 3)
        self.sqrt_dk = (self.d_k) ** 0.5
        self.w_std = kwargs.get('w_std', 0.5)
        self.w_mean = kwargs.get('w_mean', 0.05)
        assert self.post.num % self.group_num == 0
        self._variables['W_v[link]'] = torch.empty(self.pre_num, self.post_num)  # w_v
        self._variables['W_k[link]'] = torch.empty(self.group_num, self.pre_num, self.d_k)
        self._variables['W_q[link]'] = torch.empty(self.group_num, self.pre_num,
                                                   self.d_k * self.post_num // self.group_num)
        torch.nn.init.normal_(self._variables['W_v[link]'], self.w_mean, self.w_std)
        torch.nn.init.xavier_normal_(self._variables['W_k[link]'])
        torch.nn.init.xavier_normal_(self._variables['W_q[link]'])
        self._variables['Key[link]'] = torch.zeros((1, self.group_num, self.pre_num, self.d_k))
        self._variables['Query[link]'] = torch.zeros((1, self.group_num, self.post_num // self.group_num, self.d_k))
        self._variables['tau_m[link]'] = kwargs.get('tau_m', 10.0)
        self._variables['tau_r[link]'] = kwargs.get('tau_r', 5.0)
        self._variables['beta[link]'] = kwargs.get('beta', 1.0)

    def update(self, input0, Key, Query, beta, W_v, W_k, W_q):
        # input.shape = (batch, pre_neuron_num)
        # Key.shape = (batch, group_num, pre_neuron_num, d_k)
        # W_k.shape = (group_num, pre_neuron_num, d_k)
        # Query.shape = (batch, group_num, post_neuron_num//group_num, d_k)
        # W_q.shape = (group_num, pre_neuron_num, post_neuron_num//group_num * d_k)
        # Value.shape = (batch, group_num, pre_neuron_num, post_neuron_num//group_num)
        # W_v.shape = (group_num, pre_neuron_num, post_neuron_num//group_num)
        if input0.dtype.is_complex:
            input = beta ** input0.imag * (1.0j * input0.real)
        else:
            input = input0 * 1.0j
        Key = beta * Key + input.view(-1, 1, self.pre_num, 1) * W_k.unsqueeze(0)
        Query = beta * Query + torch.complex(torch.matmul(input.view(-1, 1, 1, self.pre.num).real,
                                                          W_q).view(-1, self.group_num, self.post.num // self.group_num,
                                                                    self.d_k),
                                             torch.matmul(input.view(-1, 1, 1, self.pre.num).imag, W_q).
                                             view(-1, self.group_num, self.post.num // self.group_num, self.d_k))

        Z = torch.softmax(torch.matmul(Query.real, Key.transpose(3, 2).real) / self.sqrt_dk, dim=-2).view(-1,
                                                                                                          self.pre_num,
                                                                                                          self.post_num)  # shape=(batch, group_num, 1, pre_neuron_num)
        Value = input.view(-1, self.pre.num, 1) * W_v
        Isyn = torch.sum(Z * Value, dim=-2).view(-1, self.post.num)  # shape=(batch, post_neuron_num)

        return Key, Query, Isyn

    def init_beta(self, tau_m, tau_r, dt):
        beta = torch.exp(-dt / tau_m - 2 * np.pi * dt * 1.0j / tau_r)
        return beta

    def build(self, backend):
        super(ComplexAttentionConnection, self).build(backend)
        from .BaseModule import Op
        Wv_name = self.add_conn_label('W_v[link]')
        Wk_name = self.add_conn_label('W_k[link]')
        Wq_name = self.add_conn_label('W_q[link]')
        beta_name = self.add_conn_label('beta[link]')
        Key_name = self.add_conn_label('Key[link]')
        Query_name = self.add_conn_label('Query[link]')
        update_op = Op(owner=self)
        update_op.func_name = self.update
        update_op.output = [Key_name, Query_name, self.add_conn_label(self.post_var_name + '[post]')]
        update_op.input = [self.add_conn_label(self.pre_var_name + '[pre]'), Key_name, Query_name,
                           beta_name, Wv_name, Wk_name, Wq_name]
        backend.add_operation(update_op)
        self._operations.append(update_op)

        init_op = Op(owner=self)
        init_op.func_name = self.init_beta
        init_op.output = self.add_conn_label('beta[link]')
        init_op.input = [self.add_conn_label('tau_m[link]'), self.add_conn_label('tau_r[link]'), '[dt]']
        self._init_operations.append(init_op)
        backend.register_initial(init_op)


Connection.register('complex_attention', ComplexAttentionConnection)


class pool_connect(Connection):
    '''
    do the pooling operation on spiking trains.

    Args:
        pre(Assembly): the assembly which needs to be connected
        post(Assembly): the assembly which needs to connect the pre
        link_type(str): Conv
    Methods:
        unit_connect: define the basic connection information and add them to the connection_information.
        condition_check: check whether the pre_group.type is equal to the post_group.type, only if they are equal, return flag=Ture.

    '''

    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=['maxpool'], max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn',
                 syn_kwargs=None, **kwargs):
        if syn_type is None: syn_type = ['maxpool']
        super(pool_connect, self).__init__(pre=pre, post=post, name=name,
                                           link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                           sparse_with_mask=sparse_with_mask,
                                           pre_var_name=pre_var_name, post_var_name=post_var_name,
                                           syn_kwargs=syn_kwargs, **kwargs)

        self.pool_kernel_size = kwargs.get('pool_kernel_size', [2, 2])
        self.pool_stride = kwargs.get('pool_stride', 2)
        self.pool_padding = kwargs.get('pool_padding', 0)
        if isinstance(self.pool_kernel_size, int):
            self.pool_kernel_size = [self.pool_kernel_size] * 2
        if isinstance(self.pool_stride, int):
            self.pool_stride = [self.pool_stride] * 2
        if isinstance(self.pool_padding, int):
            self.pool_padding = [self.pool_padding] * 2

        self._constant_variables['pool_kernel_size[link]'] = self.pool_kernel_size
        self._constant_variables['pool_stride[link]'] = self.pool_stride
        self._constant_variables['pool_padding[link]'] = self.pool_padding
        self.pool_only = True

    def condition_check(self, pre_group, post_group):
        '''
        check whether the pre_group.type is equal to the post_group.type, only if they are equal, return flag=Ture.

        Args:
            pre_group(Groups): the neuron group which need to be connected in the pre.
            post_group(Groups): the neuron group which need to connect the pre_group in the post.

        Returns: flag

        '''

        flag = False
        pre_type = pre_group.type
        post_type = post_group.type
        if pre_type == post_type:
            flag = True
        return flag


Connection.register('pool', pool_connect)
Connection.register('pool_connection', pool_connect)


class sparse_connect_sparse(Connection):

    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=['basic'], max_delay=0, sparse_with_mask=False, pre_var_name='O',
                 post_var_name='Isyn', syn_kwargs=None, **kwargs):
        if syn_type is None: syn_type = ['basic']
        super(sparse_connect_sparse, self).__init__(pre=pre, post=post, name=name,
                                                    link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                                    sparse_with_mask=sparse_with_mask,
                                                    pre_var_name=pre_var_name, post_var_name=post_var_name,
                                                    syn_kwargs=syn_kwargs, **kwargs)
        weight = kwargs.get('weight', None)
        self.w_std = kwargs.get('w_std', 0.05)
        self.density = kwargs.get('density', 0.1)
        self.w_max = kwargs.get('w_max', None)
        self.w_min = kwargs.get('w_min', None)
        # self.init = kwargs.get('init', None)
        self.is_parameter = kwargs.get('is_parameter', False)
        self.is_sparse = kwargs.get('is_sparse', True)
        # self._variables = dict()

        if weight is None:
            # Connection weight
            sparse_matrix = self.w_std * sp.rand(*self.shape, density=self.density, format='csr')
            self.weight = sparse_matrix.toarray()
        else:
            if isinstance(weight, BaseInitializer):
                self.w_init, self.w_init_param = self.decode_initializer(weight)
                if self.w_init == 'sparse':
                    self.weight = np.empty(self.shape)
                else:
                    raise ValueError('Sparse implementation of sparse_connection_sparse connection do not support '
                                     'the %s initialization method', self.w_init)
            else:
                assert (weight.shape == self.shape), f"The size of the given weight {weight.shape} does not correspond" \
                                                     f" to the size of synaptic matrix {self.shape} "
                self.weight = weight

        self._variables['weight[link]'] = self.weight

        bias = kwargs.get('bias', None)
        if bias is None:
            self.bias_flag = False
        else:
            self.bias_flag = True
            if isinstance(bias, BaseInitializer):
                self.b_init, self.b_init_param = self.decode_initializer(bias)
                self.bias_value = np.empty(self.post_num)

            else:
                assert (bias.size == self.post_num), f"The size of the given bias {bias.shape} does not correspond" \
                                                     f" to the size of post_num {self.post_num} "
                self.bias_value = bias

            self._variables['bias[link]'] = self.bias_value

    def condition_check(self, pre_group, post_group):
        flag = False
        pre_type = pre_group.type
        post_type = post_group.type
        if pre_type == post_type:
            flag = True
        return flag
        pass


Connection.register('sparse_sparse', sparse_connect_sparse)
Connection.register('sparse_connection_sparse', sparse_connect_sparse)


class sparse_connect_mask(Connection):

    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=['basic'], max_delay=0, sparse_with_mask=False, pre_var_name='O',
                 post_var_name='Isyn', syn_kwargs=None, **kwargs):
        if syn_type is None: syn_type = ['basic']
        super(sparse_connect_mask, self).__init__(pre=pre, post=post, name=name,
                                                  link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                                  sparse_with_mask=sparse_with_mask,
                                                  pre_var_name=pre_var_name, post_var_name=post_var_name,
                                                  syn_kwargs=syn_kwargs, **kwargs)
        weight = kwargs.get('weight', None)
        self.w_std = kwargs.get('w_std', 0.05)
        self.w_mean = kwargs.get('w_mean', 0.005)
        self.density = kwargs.get('density', 0.1)
        self.w_max = kwargs.get('w_max', None)
        self.w_min = kwargs.get('w_min', None)
        # self.init = kwargs.get('init', None)
        self.is_parameter = kwargs.get('is_parameter', True)
        self.is_sparse = kwargs.get('is_sparse', False)
        self.is_int = kwargs.get('is_int', False)
        self.kwargs = kwargs
        # self._variables = dict()

        if weight is None:
            # Connection weight
            # sparse_matrix = self.w_std * sp.rand(*self.shape, density=self.density, format='csr')
            # self.weight = sparse_matrix.toarray()
            # self.weight[self.weight.nonzero()] = self.weight[self.weight.nonzero()] + self.w_mean

            ########################### this takes a lot of time################################
            self.weight = self.w_std * np.random.randn(*self.shape) + self.w_mean
            if self.w_max is not None or self.w_min is not None:
                # self.weight = self.w_std * np.random.randn(*self.shape) + self.w_mean
                self.weight = np.clip(self.w_std * np.random.randn(*self.shape) + self.w_mean, a_min=self.w_min,
                                      a_max=self.w_max)
            else:
                self.weight = self.w_std * np.random.randn(*self.shape) + self.w_mean
            sparse_mask = np.less(np.random.rand(*self.shape), self.density)
            self.weight = np.multiply(self.weight, sparse_mask)
            ###################################################################################
            if self.is_int:
                self.weight = np.round(self.weight)
        else:
            if isinstance(weight, BaseInitializer):
                raise ValueError('Dense implementation of sparse_connection_mask do not support the %s initialization'
                                 ' method', self.w_init)
            else:
                assert (weight.shape == self.shape), f"The size of the given weight {weight.shape} does not correspond" \
                                                     f" to the size of synaptic matrix {self.shape} "
                self.weight = weight

        self._variables['weight[link]'] = self.weight

        bias = kwargs.get('bias', None)
        if bias is None:
            self.bias_flag = False
        else:
            self.bias_flag = True
            if isinstance(bias, BaseInitializer):
                self.b_init, self.b_init_param = self.decode_initializer(bias)
                self.bias_value = np.empty(self.post_num)

            else:
                assert (bias.size == self.post_num), f"The size of the given bias {bias.shape} does not correspond" \
                                                     f" to the size of post_num {self.post_num} "
                self.bias_value = bias

            self._variables['bias[link]'] = self.bias_value

    def condition_check(self, pre_group, post_group):
        flag = False
        pre_type = pre_group.type
        post_type = post_group.type
        if pre_type == post_type:
            flag = True
        return flag
        pass


Connection.register('sparse', sparse_connect_mask)
Connection.register('sparse_connection', sparse_connect_mask)


class random_connect_sparse(Connection):

    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=['basic'], max_delay=0, sparse_with_mask=False, pre_var_name='O',
                 post_var_name='Isyn', syn_kwargs=None, **kwargs):
        if syn_type is None: syn_type = ['basic']
        super(random_connect_sparse, self).__init__(pre=pre, post=post, name=name,
                                                    link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                                    sparse_with_mask=sparse_with_mask,
                                                    pre_var_name=pre_var_name, post_var_name=post_var_name,
                                                    syn_kwargs=syn_kwargs, **kwargs)
        weight = kwargs.get('weight', None)
        self.probability = kwargs.get('probability', 0.1)
        self.w_max = kwargs.get('w_max', None)
        self.w_min = kwargs.get('w_min', None)
        # self.init = kwargs.get('init', None)
        self.is_parameter = kwargs.get('is_parameter', False)
        self.is_sparse = kwargs.get('is_sparse', True)
        # self._variables = dict()

        if weight is None:
            # Link_parameters
            prob_weight = np.random.rand(*self.shape)
            diag_index = np.arange(min([self.pre_num, self.post_num]))
            prob_weight[diag_index, diag_index] = 1
            index = (prob_weight < self.probability)
            # Connection weight
            self.weight = np.zeros(self.shape)
            self.weight[index] = prob_weight[index]
        else:
            if isinstance(weight, BaseInitializer):
                self.w_init, self.w_init_param = self.decode_initializer(weight)
                if self.w_init == 'sparse':
                    self.weight = np.empty(self.shape)
                else:
                    raise ValueError('Sparse implementation of random_connect_sparse connection do not support '
                                     'the %s initialization method', self.w_init)
                self.weight = np.random.randn(*self.shape)
            else:
                assert (weight.shape == self.shape), f"The size of the given weight {weight.shape} does not correspond" \
                                                     f" to the size of synaptic matrix {self.shape} "
                self.weight = weight

        self._variables['weight[link]'] = self.weight

        bias = kwargs.get('bias', None)
        if bias is None:
            self.bias_flag = False
        else:
            self.bias_flag = True
            if isinstance(bias, BaseInitializer):
                self.b_init, self.b_init_param = self.decode_initializer(bias)
                self.bias_value = np.empty(self.post_num)

            else:
                assert (bias.size == self.post_num), f"The size of the given bias {bias.shape} does not correspond" \
                                                     f" to the size of post_num {self.post_num} "
                self.bias_value = bias

            self._variables['bias[link]'] = self.bias_value

    def condition_check(self, pre_group, post_group):
        flag = False
        pre_type = pre_group.type
        post_type = post_group.type
        if pre_type == post_type:
            flag = True
        return flag
        pass


Connection.register('random_connection_sparse', random_connect_sparse)


class random_connect_mask(Connection):

    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=['basic'], max_delay=0, sparse_with_mask=False, pre_var_name='O',
                 post_var_name='Isyn', syn_kwargs=None, **kwargs):
        if syn_type is None: syn_type = ['basic']
        super(random_connect_mask, self).__init__(pre=pre, post=post, name=name,
                                                  link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                                  sparse_with_mask=sparse_with_mask,
                                                  pre_var_name=pre_var_name, post_var_name=post_var_name,
                                                  syn_kwargs=syn_kwargs, **kwargs)
        weight = kwargs.get('weight', None)
        self.probability = kwargs.get('probability', 0.1)
        self.w_max = kwargs.get('w_max', None)
        self.w_min = kwargs.get('w_min', None)
        # self.init = kwargs.get('init', None)
        self.is_parameter = kwargs.get('is_parameter', True)
        self.is_sparse = kwargs.get('is_sparse', False)
        # self._variables = dict()

        if weight is None:
            # Link_parameters
            prob_weight = np.random.rand(*self.shape)
            diag_index = np.arange(min([self.pre_num, self.post_num]))
            prob_weight[diag_index, diag_index] = 1
            index = (prob_weight < self.probability)
            # Connection weight
            self.weight = np.zeros(self.shape)
            self.weight[index] = prob_weight[index]
        else:
            if isinstance(weight, BaseInitializer):
                raise ValueError(
                    'Dense implementation of random_connect_mask do not support the %s initialization method',
                    self.w_init)
            else:
                assert (weight.shape == self.shape), f"The size of the given weight {weight.shape} does not correspond" \
                                                     f" to the size of synaptic matrix {self.shape} "
                self.weight = weight

        self._variables['weight[link]'] = self.weight

        bias = kwargs.get('bias', None)
        if bias is None:
            self.bias_flag = False
        else:
            self.bias_flag = True
            if isinstance(bias, BaseInitializer):
                self.b_init, self.b_init_param = self.decode_initializer(bias)
                self.bias_value = np.empty(self.post_num)

            else:
                assert (bias.size == self.post_num), f"The size of the given bias {bias.shape} does not correspond" \
                                                     f" to the size of post_num {self.post_num} "
                self.bias_value = bias

            self._variables['bias[link]'] = self.bias_value

    def condition_check(self, pre_group, post_group):
        flag = False
        pre_type = pre_group.type
        post_type = post_group.type
        if pre_type == post_type:
            flag = True
        return flag
        pass


Connection.register('random', random_connect_mask)
Connection.register('random_connection', random_connect_mask)


class NullConnection(Connection):
    '''
    each neuron in the first layer is connected to each neuron in the second layer.

    Args:
        pre(Assembly): The assembly which needs to be connected
        post(Assembly): The assembly which needs to connect the pre
        link_type(str): full

    Methods:
        unit_connect: define the basic connection information and add them to the connection_information.
        condition_check: check whether the pre_group.type is equal to the post_group.type, only if they are equal, return flag=Ture.

    '''

    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=['basic'], max_delay=0, sparse_with_mask=False, pre_var_name='O',
                 post_var_name='Isyn', syn_kwargs=None, **kwargs):
        if syn_type is None: syn_type = ['basic']
        super(NullConnection, self).__init__(pre=pre, post=post, name=name,
                                             link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                             sparse_with_mask=sparse_with_mask,
                                             pre_var_name=pre_var_name, post_var_name=post_var_name,
                                             syn_kwargs=syn_kwargs, **kwargs)

        pass


Connection.register('null', NullConnection)


class Spike_to_Mass(Connection):
    '''
    Connect pre of spiking neurons to post of Neural Mass model
    '''

    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=['basic'], max_delay=0, sparse_with_mask=False, pre_var_name='O',
                 post_var_name='Isyn', syn_kwargs=None, **kwargs):
        if syn_type is None: syn_type = ['basic']
        super(Spike_to_Mass, self).__init__(pre=pre, post=post, name=name,
                                            link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                            sparse_with_mask=sparse_with_mask,
                                            pre_var_name=pre_var_name, post_var_name=post_var_name,
                                            syn_kwargs=syn_kwargs, **kwargs)
        weight = kwargs.get('weight', None)
        bias = kwargs.get('bias', None)
        self.tau_syn = kwargs.get('tau_syn', 6.0)
        self.w_std = kwargs.get('w_std', 0.05)
        self.w_mean = kwargs.get('w_mean', 0.005)
        self.w_max = kwargs.get('w_max', None)
        self.w_min = kwargs.get('w_min', None)
        self.is_parameter = kwargs.get('is_parameter', True)
        self.shape = (post.num, 1)
        self.synapse_class = []

        if weight is None:
            # Connection weight
            self.weight = self.w_std * np.random.randn(*self.shape) + self.w_mean
        else:
            if isinstance(weight, BaseInitializer):
                self.w_init, self.w_init_param = self.decode_initializer(weight)
                self.weight = np.empty(self.shape)
            else:
                assert (weight.shape == self.shape), f"The size of the given weight {weight.shape} does not correspond" \
                                                     f" to the size of synaptic matrix {self.shape} "
                self.weight = weight

        self._variables['weight[link]'] = self.weight
        self._variables['beta[link]'] = np.exp(-0.1 / self.tau_syn)
        self._variables['rate[link]'] = 0.0
        self._operations.append(
            ['rate[link]', self.spike_to_rate, [pre_var_name + '[pre]', 'rate[link]', 'beta[link]']])
        self._operations.append(
            [self.post_var_name + '[post]', 'mat_mult_weight', 'rate[link][updated]', 'weight[link]'])

    def spike_to_rate(self, spikes, last_rate, beta):
        rate = beta * last_rate + torch.mean(spikes, dim=1, keepdim=True)
        return rate


Connection.register('spike2mass', Spike_to_Mass)


class Mass_to_Spike(Connection):
    '''
    Connect pre of spiking neurons to post of Neural Mass model
    '''

    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=['basic'], max_delay=0, sparse_with_mask=False, pre_var_name='O',
                 post_var_name='Isyn', syn_kwargs=None, **kwargs):
        if syn_type is None: syn_type = ['basic']
        super(Mass_to_Spike, self).__init__(pre=pre, post=post, name=name,
                                            link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                            sparse_with_mask=sparse_with_mask,
                                            pre_var_name=pre_var_name, post_var_name=post_var_name,
                                            syn_kwargs=syn_kwargs, **kwargs)
        weight = kwargs.get('weight', None)
        bias = kwargs.get('bias', None)
        self.tau_syn = kwargs.get('tau_syn', 6.0)
        self.w_std = kwargs.get('w_std', 0.05)
        self.w_mean = kwargs.get('w_mean', 0.005)
        self.w_max = kwargs.get('w_max', None)
        self.w_min = kwargs.get('w_min', None)
        self.is_parameter = kwargs.get('is_parameter', True)
        self.shape = (post.num, pre.num)
        self.synapse_class = []
        self.unit_conversion = kwargs.get('unit_conversion', 1.0)

        if weight is None:
            # Connection weight
            self.weight = self.w_std * np.random.randn(*self.shape) + self.w_mean
        else:
            if isinstance(weight, BaseInitializer):
                self.w_init, self.w_init_param = self.decode_initializer(weight)
                self.weight = np.empty(self.shape)
            else:
                assert (weight.shape == self.shape), f"The size of the given weight {weight.shape} does not correspond" \
                                                     f" to the size of synaptic matrix {self.shape} "
                self.weight = weight

        self._variables['weight[link]'] = self.weight
        self._variables['beta[link]'] = np.exp(-0.1 / self.tau_syn)
        self._variables['rate[link]'] = 0.0

        self._operations.append([post_var_name + '[post]', self.rate_to_Isyn, [pre_var_name + '[pre]', 'weight[link]']])

    def rate_to_Isyn(self, rate, weight):
        rate = rate.view(rate.shape[0], 1, self.pre.num).expand(rate.shape[0],
                                                                self.post.num, self.pre.num)
        spikes = torch.rand_like(rate).le(rate * self.unit_conversion * self.dt).type(self._backend.data_type)
        Isyn = torch.sum(spikes * weight, dim=-1)
        return Isyn


Connection.register('mass2spike', Mass_to_Spike)


class DistDepd_connect(Connection):

    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=['basic'], max_delay=0, sparse_with_mask=False, pre_var_name='O',
                 post_var_name='Isyn', syn_kwargs=None, **kwargs):
        if syn_type is None: syn_type = ['basic']
        super(DistDepd_connect, self).__init__(pre=pre, post=post, name=name,
                                               link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                               sparse_with_mask=sparse_with_mask,
                                               pre_var_name=pre_var_name, post_var_name=post_var_name,
                                               syn_kwargs=syn_kwargs, **kwargs)
        self.distance_weight_function = kwargs.get('distance_weight_function', None)
        self.zero_self = kwargs.get('zero_self', False)
        if self.distance_weight_function is None:
            self.distance_weight_function = self.default_dist_weight_function
        self.dist_function = kwargs.get('dist_function', 'euclidean')
        if self.dist_function == 'euclidean':
            self.dist_function = self.euclidean_dist_function
        elif self.dist_function == 'circular':
            self.dist_function = self.circular_dist_function

        self.dist_a = kwargs.get('dist_a', 0.2)
        self.dist_b = kwargs.get('dist_b', 0.4)
        self.w_amp = kwargs.get('w_amp', -0.1)
        self.pos_range = kwargs.get('pos_range', 1.0)
        self.p = kwargs.get('p', 1.0)

        # building the weight
        pre_num = pre.num
        post_num = post.num
        assert len(pre.position) > 0
        assert len(post.position) > 0

        post_pos = np.expand_dims(post.position, axis=1)
        pre_pos = np.expand_dims(pre.position, axis=0)
        weight = self.distance_weight_function(self.dist_function(pre_pos, post_pos))
        if self.p < 1.0:
            rand_mask = torch.rand_like(weight).lt(self.p)
            weight = weight * rand_mask
        self._variables['weight[link]'] = weight

    def unit_connect(self, pre_group, post_group):
        # The number of neurons in neuron group
        pre_num = pre_group.num
        post_num = post_group.num
        assert len(pre_group.position) > 0
        assert len(post_group.position) > 0
        link_num = pre_num * post_num
        shape = (post_num, pre_num)
        weight = np.zeros(shape)
        post_pos = np.expand_dims(post_group.position, axis=1)
        pre_pos = np.expand_dims(pre_group.position, axis=0)
        weight = self.distance_weight_function(self.dist_function(pre_pos, post_pos))
        # from matplotlib import pyplot as plt
        # plt.imshow(weight)
        # plt.show()

        # The name for backend variables
        input_name = self.get_input_name(pre_group, post_group)
        weight_name = self.get_weight_name(pre_group, post_group)
        target_name = self.get_target_name(post_group)

        # The backend variable
        var_code = (weight_name, shape, weight, True, False)  # (var_name, shape, value, is_parameter, is_sparse, init)
        op_code = [target_name, 'mat_mult', input_name, weight_name]
        connection_information = (pre_group, post_group, link_num, var_code, op_code)
        self.unit_connections.append(connection_information)

    def circular_dist_function(self, pre_pos, post_pos):
        if not isinstance(pre_pos, torch.Tensor):
            pre_pos = torch.tensor(pre_pos)
        if not isinstance(post_pos, torch.Tensor):
            post_pos = torch.tensor(post_pos)

        z = torch.maximum(pre_pos, post_pos)
        k = torch.minimum(pre_pos, post_pos)
        dist = torch.minimum(z - k, self.pos_range + k - z)
        dist = torch.norm(dist, p=2, dim=-1)
        return dist

    def euclidean_dist_function(self, pre_pos, post_pos):
        if isinstance(pre_pos, torch.Tensor) and isinstance(post_pos, torch.Tensor):
            diff = pre_pos - post_pos
        else:
            diff = torch.tensor(pre_pos - post_pos)
        dist = torch.norm(diff, p=2, dim=-1)
        return dist

    def default_dist_weight_function(self, dist):
        weights = self.w_amp * (
                    torch.exp(-dist / self.dist_a) / self.dist_a - 0.5 * torch.exp(-dist / self.dist_b) / self.dist_b)
        if self.zero_self:
            weights = weights * (dist != 0).type(weights.dtype)
        # import matplotlib.pyplot as plt
        # plt.imshow(weights, aspect='auto')
        # plt.show()
        return weights


Connection.register('dist_depd', DistDepd_connect)

# class reconnect(Connection):
#     def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'), policies=[],
#                  max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn', syn_kwargs=None, **kwargs):
#         super(reconnect, self).__init__(pre=pre, post=post, name=name, link_type=link_type,
#                                              policies=policies, max_delay=max_delay, sparse_with_mask=sparse_with_mask, pre_var_name=pre_var_name, post_var_name=post_var_name,syn_kwargs=syn_kwargs, **kwargs)
#     def unit_connect(self, pre_group, post_group):
#         pass
#
#     def condition_check(self, pre_group, post_group):
#         pass
