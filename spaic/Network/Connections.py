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
from ..Network.Topology import Connection
from ..Network.Assembly import Assembly
import numpy as np
import scipy.sparse as sp
import torch
from ..IO.Initializer import BaseInitializer
import random

class FullConnection(Connection):

    '''
    each neuron in the first layer is connected to each neuron in the second layer.

    Args:
        pre(Assembly): The assembly which needs to be connected
        post(Assembly): The assembly which needs to connect the pre
        link_type(str): full
    '''

    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=['basic'], max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn', syn_kwargs=None, **kwargs):
        super(FullConnection, self).__init__(pre=pre, post=post, name=name,
                                             link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                             sparse_with_mask=sparse_with_mask,
                                             pre_var_name=pre_var_name, post_var_name=post_var_name, syn_kwargs=syn_kwargs, **kwargs)
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
            self.weight = self.w_std*np.random.randn(*self.shape) + self.w_mean
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

            self._operations.append([None, self.update_ave, 'V'+'[post]'])
            self._init_operations.append(['weight[link]', self.update_bn, [post_var_name+'[post]', 'weight[link]']])

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
                    self.running_ave = self.beta_ave*self.running_ave + (1-self.beta_ave)*mean_rate
                    self.running_std = self.beta_std*self.running_std + (1-self.beta_std)*std

                    d_bias = (self.targ_ave - self.running_ave)*torch.sigmoid(20.0*(self.targ_ave - self.running_ave)*(self.running_ave-mean_rate))
                    self.running_bias = self.running_bias + self.beta_bn*d_bias
                    d_scale = torch.exp((self.targ_std - self.running_std)) - 1.0
                    self.running_scale = self.running_scale + self.beta_bn*d_scale
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
                 syn_type=['basic'], max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn', syn_kwargs=None, **kwargs):
        super(one_to_one_sparse, self).__init__(pre=pre, post=post, name=name,
                                                link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                                sparse_with_mask=sparse_with_mask,
                                                pre_var_name=pre_var_name, post_var_name=post_var_name, syn_kwargs=syn_kwargs, **kwargs)
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
                 syn_type=['basic'], max_delay=0, sparse_with_mask=True, pre_var_name='O', post_var_name='Isyn', syn_kwargs=None, **kwargs):
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
                                 'initialization method. You can pass a numpy matrix whose diagonal is not zero.', self.w_init)
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
                 syn_type=['conv'], max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn', syn_kwargs=None, **kwargs):
        super(conv_connect, self).__init__(pre=pre, post=post, name=name,
                                           link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                           sparse_with_mask=sparse_with_mask,
                                           pre_var_name=pre_var_name, post_var_name=post_var_name, syn_kwargs=syn_kwargs, **kwargs)
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

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        if isinstance(self.padding, int):
            self.padding = [self.padding] * 2
        # self._variables = dict()
        self._constant_variables['stride[pre]'] = self.stride
        self._constant_variables['padding[pre]'] = self.padding
        self._constant_variables['dilation[pre]'] = self.dilation
        self._constant_variables['groups[pre]'] = self.groups

        if self.in_channels is None:
            from spaic.Neuron.Encoders import NullEncoder
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

            self._constant_variables['pool_kernel_size[pre]'] = self.pool_kernel_size
            self._constant_variables['pool_stride[pre]'] = self.pool_stride
            self._constant_variables['pool_padding[pre]'] = self.pool_padding

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
                Ho = int((Hin + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1)  # Ho = (Hin + 2 * padding[0] - kernel_size[0]) / stride[0] + 1
                Wo = int((Win + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1] + 1)  # Wo = (Win + 2 * padding[0] - kernel_size[1]) / stride[0] + 1
                # Ho = int(Ho / self.pool_kernel_size[0])
                # Wo = int(Wo / self.pool_kernel_size[1])
                Ho = int((Ho + 2 * self.pool_padding[0] - self.pool_kernel_size[0]) / self.pool_stride[0] + 1)  # Ho = (Hin + 2 * padding[0] - kernel_size[0]) / stride[0] + 1
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
                 syn_type=['maxpool'], max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn', syn_kwargs=None, **kwargs):
        super(pool_connect, self).__init__(pre=pre, post=post, name=name,
                                           link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                           sparse_with_mask=sparse_with_mask,
                                           pre_var_name=pre_var_name, post_var_name=post_var_name, syn_kwargs=syn_kwargs, **kwargs)

        self.pool_kernel_size = kwargs.get('pool_kernel_size', [2, 2])
        self.pool_stride = kwargs.get('pool_stride', 2)
        self.pool_padding = kwargs.get('pool_padding', 0)
        if isinstance(self.pool_kernel_size, int):
            self.pool_kernel_size = [self.pool_kernel_size] * 2
        if isinstance(self.pool_stride, int):
            self.pool_stride = [self.pool_stride] * 2
        if isinstance(self.pool_padding, int):
            self.pool_padding = [self.pool_padding] * 2

        self._constant_variables['pool_kernel_size[pre]'] = self.pool_kernel_size
        self._constant_variables['pool_stride[pre]'] = self.pool_stride
        self._constant_variables['pool_padding[pre]'] = self.pool_padding
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
        # self._variables = dict()

        if weight is None:
            # Connection weight
            # sparse_matrix = self.w_std * sp.rand(*self.shape, density=self.density, format='csr')
            # self.weight = sparse_matrix.toarray()
            # self.weight[self.weight.nonzero()] = self.weight[self.weight.nonzero()] + self.w_mean
            self.weight = self.w_std * np.random.rand(*self.shape) + self.w_mean
            sparse_mask = np.less(np.random.rand(*self.shape), self.density)
            self.weight = np.multiply(self.weight, sparse_mask)
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
        self._variables['beta[link]'] = np.exp(-0.1/self.tau_syn)
        self._variables['rate[link]'] = 0.0
        self._operations.append(['rate[link]', self.spike_to_rate, [pre_var_name+'[pre]', 'rate[link]', 'beta[link]']])
        self._operations.append([self.post_var_name+'[post]', 'mat_mult_weight', 'rate[link][updated]', 'weight[link]'])


    def spike_to_rate(self, spikes, last_rate, beta):
        rate = beta*last_rate + torch.mean(spikes, dim=1, keepdim=True)
        return rate

Connection.register('spike2mass', Spike_to_Mass)

class Mass_to_Spike(Connection):
    '''
    Connect pre of spiking neurons to post of Neural Mass model
    '''
    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=['basic'], max_delay=0, sparse_with_mask=False, pre_var_name='O',
                 post_var_name='Isyn', syn_kwargs=None, **kwargs):
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
        self._variables['beta[link]'] = np.exp(-0.1/self.tau_syn)
        self._variables['rate[link]'] = 0.0

        self._operations.append([post_var_name+'[post]', self.rate_to_Isyn, [pre_var_name+'[pre]', 'weight[link]']])



    def rate_to_Isyn(self, rate, weight):
        rate = rate.view(rate.shape[0], 1, self.pre.num).expand(rate.shape[0],
                                                                self.post.num, self.pre.num)
        spikes = torch.rand_like(rate).le(rate * self.unit_conversion * self.dt).type(self._backend.data_type)
        Isyn = torch.sum(spikes*weight, dim=-1)
        return Isyn
Connection.register('mass2spike', Mass_to_Spike)

class DistDepd_connect(Connection):

    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connection', 'conv', '...'),
                 syn_type=['basic'], max_delay=0, sparse_with_mask=False, pre_var_name='O',
                 post_var_name='Isyn', syn_kwargs=None, **kwargs):
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
            weight = weight*rand_mask
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
        post_pos = np.expand_dims(post_group.position,  axis=1)
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
        var_code = (weight_name, shape, weight, True, False) # (var_name, shape, value, is_parameter, is_sparse, init)
        op_code = [target_name, 'mat_mult', input_name, weight_name]
        connection_information = (pre_group, post_group, link_num, var_code, op_code)
        self.unit_connections.append(connection_information)

    def circular_dist_function(self, pre_pos, post_pos):
        if not isinstance(pre_pos, torch.Tensor):
            pre_pos = torch.tensor(pre_pos)
        if not  isinstance(post_pos, torch.Tensor):
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
            diff = torch.tensor(pre_pos-post_pos)
        dist = torch.norm(diff, p=2, dim=-1)
        return dist

    def default_dist_weight_function(self, dist):
        weights = self.w_amp * (torch.exp(-dist / self.dist_a)/self.dist_a - 0.5*torch.exp(-dist / self.dist_b)/self.dist_b)
        if self.zero_self:
            weights = weights * (dist!=0).type(self._backend.data_type)
        # import matplotlib.pyplot as plt
        # plt.imshow(weights, aspect='auto')
        # plt.show()
        return weights
Connection.register('dist_depd', DistDepd_connect)

# class reconnect(Connection):
#     def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connect', 'conv', '...'), policies=[],
#                  max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn', syn_kwargs=None, **kwargs):
#         super(reconnect, self).__init__(pre=pre, post=post, name=name, link_type=link_type,
#                                              policies=policies, max_delay=max_delay, sparse_with_mask=sparse_with_mask, pre_var_name=pre_var_name, post_var_name=post_var_name,syn_kwargs=syn_kwargs, **kwargs)
#     def unit_connect(self, pre_group, post_group):
#         pass
#
#     def condition_check(self, pre_group, post_group):
#         pass