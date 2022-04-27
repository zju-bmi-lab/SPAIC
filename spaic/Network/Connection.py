# -*- coding: utf-8 -*-
"""
Created on 2020/8/5
@project: SPAIC
@filename: Connection
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
定义神经集群间的连接，包括记录神经元集群、连接的突触前、突触后神经元编号、连接形式（全连接、稀疏连接、卷积）、权值、延迟 以及连接产生函数、重连接函数等。
"""
from ..Network.BaseModule import BaseModule
from ..Network.Topology import Projection, Connection
from ..Network.Assembly import Assembly
# from ..Network.ConnectPolicy import  ConnectPolicy
from collections import OrderedDict
from typing import Dict, List, Tuple
from abc import abstractmethod
import numpy as np
import scipy.sparse as sp
import spaic
import torch
import math

class FullConnection(Connection):

    '''
    each neuron in the first layer is connected to each neuron in the second layer.

    Args:
        pre_assembly(Assembly): The assembly which needs to be connected
        post_assembly(Assembly): The assembly which needs to connect the pre_assembly
        link_type(str): full

    Methods:
        unit_connect: define the basic connection information and add them to the connection_information.
        condition_check: check whether the pre_group.type is equal to the post_group.type, only if they are equal, return flag=Ture.

    '''

    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse_connect', 'conv','...'), policies=[],
                 max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):

        super(FullConnection, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name,
                                             link_type=link_type, policies=policies, max_delay=max_delay,
                                             sparse_with_mask=sparse_with_mask,
                                             pre_var_name=pre_var_name, post_var_name=post_var_name, **kwargs)
        self.weight = kwargs.get('weight', None)
        self.w_std = kwargs.get('w_std', 0.05)
        self.w_mean = kwargs.get('w_mean', 0.005)
        self.w_max = kwargs.get('w_max', None)
        self.w_min = kwargs.get('w_min', None)
        self.param_init = kwargs.get('param_init', None)
        self.is_parameter = kwargs.get('is_parameter', True)
        self.is_sparse = kwargs.get('is_sparse', False)
        # self.flatten_on = kwargs.get('flatten', False)

    def unit_connect(self, pre_group, post_group):
        # The number of neurons in neuron group
        pre_num = pre_group.num
        post_num = post_group.num
        link_num = pre_num * post_num
        shape = (post_num, pre_num)
        if self.weight is None:
            # Connection weight
            weight = self.w_std*np.random.randn(*shape) + self.w_mean
        else:
            assert (self.weight.shape == shape), f"The size of the given weight {self.weight.shape} does not correspond to the size of synaptic matrix {shape} "
            weight = self.weight

        # The name for backend variables
        input_name = self.get_input_name(pre_group, post_group)
        # 如果需要避开固有延迟的问题，采用strategy_build.
        # 取消下面一行的注释，取用突触前神经元当前步的发放情况
        input_name = input_name
        weight_name = self.get_link_name(pre_group, post_group, 'weight')
        target_name = self.get_post_name(post_group, self.post_var_name)

        # The backend variable
        # var_code = (weight_name, shape, weight, True, False, 'uniform')   # (var_name, shape, value, is_parameter, is_sparse, init)
        var_code = (weight_name, shape, weight, self.is_parameter, self.is_sparse, self.param_init)

        # The backend basic operation
        if self.max_delay > 0 :
            op_code = [target_name, 'mult_sum_weight', input_name, weight_name]
        else:
            op_code = [target_name, 'mat_mult_weight', input_name, weight_name]


        # if self.max_delay > 0 and self.flatten_on != True:
        #     op_code = [target_name, 'mult_sum_weight', input_name, weight_name]
        # elif self.max_delay > 0 and self.flatten_on == True:
        #     raise ValueError("Conv_connectoin cannot do delay recently!")
        # elif self.max_delay <= 0 and self.flatten_on == True:
        #     op_code = [target_name, 'reshape_mat_mult', input_name, weight_name]
        # else:
        #     op_code = [target_name, 'mat_mult_weight', input_name, weight_name]

        connection_information = (pre_group, post_group, link_num, var_code, op_code)
        self.unit_connections.append(connection_information)

        pass

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
    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse_connect', 'conv','...'), policies=[],
                 max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):
        super(one_to_one_sparse, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name,
                                                link_type=link_type, policies=policies, max_delay=max_delay,
                                                sparse_with_mask=sparse_with_mask,
                                                pre_var_name=pre_var_name, post_var_name=post_var_name, **kwargs)
        self.weight = kwargs.get('weight', None)
        self.w_mean = kwargs.get('w_mean', 0.05)
        self.w_max = kwargs.get('w_max', None)
        self.w_min = kwargs.get('w_min', None)
        self.param_init = kwargs.get('param_init', None)
        self.is_parameter = kwargs.get('is_parameter', False)
        self.is_sparse = kwargs.get('is_sparse', True)

    def unit_connect(self, pre_group, post_group):
        pre_num = pre_group.num
        post_num = post_group.num
        try:
            assert pre_num == post_num
        except AssertionError:
            raise ValueError('One to One connection must be defined in two groups with the same size, but the pre_num %s is not equal to the post_num %s.'%(pre_num, post_num))
        link_num = pre_num * post_num
        shape = (post_num, pre_num)

        if self.weight is None:
            # Connection weight
            weight = self.w_mean * np.eye(*shape)
        else:
            assert (self.weight.shape == shape), f"The size of the given weight {self.weight.shape} does not correspond to the size of synaptic matrix {shape} "
            weight = self.weight

        # The name for backend variables
        input_name = self.get_input_name(pre_group, post_group)
        weight_name = self.get_link_name(pre_group, post_group, 'weight')
        target_name = self.get_post_name(post_group, self.post_var_name)

        # The backend variable
        var_code = (weight_name, shape, weight, self.is_parameter, self.is_sparse, self.param_init)  # (var_name, shape, value, is_parameter, is_sparse)

        # The backend basic operation
        op_code = [target_name, 'sparse_mat_mult_weight', weight_name, input_name]

        connection_information = (pre_group, post_group, link_num, var_code, op_code)
        self.unit_connections.append(connection_information)

        pass

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
    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse_connect', 'conv','...'), policies=[],
                 max_delay=0, sparse_with_mask=True, pre_var_name='O', post_var_name='WgtSum', **kwargs):
        super(one_to_one_mask, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name,
                                              link_type=link_type, policies=policies, max_delay=max_delay,
                                              sparse_with_mask=sparse_with_mask,
                                              pre_var_name=pre_var_name, post_var_name=post_var_name, **kwargs)
        self.weight = kwargs.get('weight', None)
        self.w_mean = kwargs.get('w_mean', 0.05)
        self.w_max = kwargs.get('w_max', None)
        self.w_min = kwargs.get('w_min', None)
        self.param_init = kwargs.get('param_init', None)
        self.is_parameter = kwargs.get('is_parameter', True)
        self.is_sparse = kwargs.get('is_sparse', False)

    def unit_connect(self, pre_group, post_group):
        pre_num = pre_group.num
        post_num = post_group.num
        try:
            assert pre_num == post_num
        except AssertionError:
            raise ValueError('One to One connection must be defined in two groups with the same size, '
                             'but the pre_num %s is not equal to the post_num %s.'%(pre_num, post_num))

        link_num = pre_num * post_num
        shape = (post_num, pre_num)

        if self.weight is None:
            # Connection weight
            weight = self.w_mean * np.eye(*shape)
        else:
            assert (self.weight.shape == shape), f"The size of the given weight {self.weight.shape} does not correspond to the size of synaptic matrix {shape} "
            weight = self.weight

        # The name for backend variables
        input_name = self.get_input_name(pre_group, post_group)
        weight_name = self.get_link_name(pre_group, post_group, 'weight')
        target_name = self.get_post_name(post_group, self.post_var_name)

        # The backend variable
        var_code = (weight_name, shape, weight, self.is_parameter, self.is_sparse, self.param_init)  # (var_name, shape, value, is_parameter, is_sparse)

        # The backend basic operation
        op_code = [target_name, 'mat_mult_weight', input_name, weight_name]

        connection_information = (pre_group, post_group, link_num, var_code, op_code)
        self.unit_connections.append(connection_information)

        if self.sparse_with_mask:
            mask = (weight != 0)
            mask_name = self.get_link_name(pre_group, post_group, 'mask')
            mask_var_code = (mask_name, shape, mask)
            mask_op = (weight_name, self.mask_operation, [weight_name, mask_name])
            mask_information = (mask_var_code, mask_op)
            self.mask_info.append(mask_information)
        pass

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
    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse_connect', 'conv','...'), policies=[],
                 max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):
        super(conv_connect, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name,
                                           link_type=link_type, policies=policies, max_delay=max_delay,
                                           sparse_with_mask=sparse_with_mask,
                                           pre_var_name=pre_var_name, post_var_name=post_var_name, **kwargs)

    '''
    do the convolution connection.

    Args:
        pre_assembly(Assembly): the assembly which needs to be connected
        post_assembly(Assembly): the assembly which needs to connect the pre_assembly
        link_type(str): Conv
    Methods:
        unit_connect: define the basic connection information and add them to the connection_information.
        condition_check: check whether the pre_group.type is equal to the post_group.type, only if they are equal, return flag=Ture.

    '''
    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse_connect', 'conv','...'), policies=[],
                 max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):
        super(conv_connect, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name,
                                           link_type=link_type, policies=policies, max_delay=max_delay,
                                           sparse_with_mask=sparse_with_mask,
                                           pre_var_name=pre_var_name, post_var_name=post_var_name, **kwargs)
        self.out_channels = kwargs.get('out_channels', 4)
        self.in_channels = kwargs.get('in_channels', 1)
        self.kernel_size = kwargs.get('kernel_size', (3, 3))
        self.maxpool_on = kwargs.get('maxpool_on', True)
        self.maxpool_kernel_size = kwargs.get('maxpool_kernel_size', (2, 2))
        self.w_std = kwargs.get('w_std', 0.05)
        self.w_mean = kwargs.get('w_mean', 0.05)

        self.weight = kwargs.get('weight', None)
        self.mask = kwargs.get('mask', None)
        self.stride = kwargs.get('stride', 1)
        self.padding = kwargs.get('padding', 0)
        self.dilation = kwargs.get('dilation', 1)
        self.groups = kwargs.get('groups', 1)



    def unit_connect(self, pre_group, post_group):
        '''
        set the basic parameters, for example: link_length, connection weight, connection shape, the name for backend variables, the backend variable,the backend basic operation.

        Args:
            pre_group(Groups): the neuron group which need to be connected in the pre_assembly.
            post_group(Groups): the neuron group which need to be connected with the pre_group neuron.

        '''
        shape = (self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        if self.weight is None:
            # Connection weight
            weight = self.w_std * np.random.randn(*shape) + self.w_mean
        else:
            assert (self.weight.shape == shape), f"The size of the given weight {self.weight.shape} does not correspond to the size of synaptic matrix {shape} "
            weight = self.weight


        Hin = pre_group.shape[-2]
        Win = pre_group.shape[-1]

        if self.maxpool_on:  # 池化

            Hin = int(Hin / self.maxpool_kernel_size[0])
            Win = int(Win / self.maxpool_kernel_size[1])

        Ho = int((Hin + 2 * self.padding - self.kernel_size[
            0]) / self.stride + 1)  # Ho = (Hin + 2 * padding[0] - kernel_size[0]) / stride[0] + 1
        Wo = int((Win + 2 * self.padding - self.kernel_size[
            1]) / self.stride + 1)  # Wo = (Win + 2 * padding[0] - kernel_size[1]) / stride[0] + 1


        post_num = int(Ho * Wo * self.out_channels)

        if post_group.num == None:
            post_group.num = post_num
            post_group.shape = (self.out_channels, Ho, Wo)

        if post_group.num != None:
            if post_group.num != post_num:
                raise ValueError(
                    "The post_group num is not equal to the output num, cannot achieve the conv connection, "
                    "the output num is %d * %d * %d " % (self.out_channels, Ho, Wo))

            else:
                post_group.shape = (self.out_channels, Ho, Wo)
        link_num = post_num
        # The name for backend variables
        input_name = self.get_pre_name(pre_group, self.pre_var_name)
        weight_name = self.get_link_name(pre_group, post_group, 'weight')
        target_name = self.get_post_name(post_group, self.post_var_name)

        # The backend variable
        var_code_weight = (weight_name, shape, weight, True)
        stride = self.get_pre_name(pre_group, 'stride')
        # stride = self.stride_name(pre_group)
        stride_value = np.array(self.stride)
        var_code_stride = (stride, stride_value.shape, stride_value)
        padding = self.get_pre_name(pre_group, 'padding')
        # padding = self.padding_name(pre_group)
        padding_value = np.array(self.padding)
        var_code_padding = (padding, padding_value.shape, padding_value)
        dilation = self.get_pre_name(pre_group, 'dilation')
        dilation_value = np.array(self.dilation)
        var_code_dilation = (dilation, dilation_value.shape, dilation_value)
        groups = self.get_pre_name(pre_group, 'groups')
        groups_value = np.array(self.groups)
        var_code_groups = (groups, groups_value.shape, groups_value)

        maxpool_kernel_size_name = self.get_pre_name(pre_group, 'maxpool_kernel_size')
        # maxpool_kernel_size_name = self.maxpool_kernel_size_name(pre_group)
        maxpool_kernel_size_value = np.array(self.maxpool_kernel_size)
        var_code_maxpool = (maxpool_kernel_size_name, maxpool_kernel_size_value.shape, maxpool_kernel_size_value)

        # flatten_value = np.array(self.flatten)
        # flatten = self.get_post_name(post_group, 'flatten')
        # var_code_flatten = (flatten, flatten_value.shape, flatten_value)

        if self.maxpool_on:

            op_code1 = [target_name, 'conv_max_pool2d', input_name, weight_name, maxpool_kernel_size_name,
                        stride, padding, dilation, groups]
            connection_information = (pre_group, post_group, link_num, var_code_weight, var_code_maxpool,
                                      var_code_stride, var_code_padding, var_code_dilation, var_code_groups, op_code1)
        else:
            op_code1 = [target_name, 'conv_2d', input_name, weight_name, stride, padding, dilation, groups]
            connection_information = (pre_group, post_group, link_num, var_code_weight,
                                      var_code_stride, var_code_padding, var_code_dilation, var_code_groups, op_code1)

        self.unit_connections.append(connection_information)


    def condition_check(self, pre_group, post_group):
        '''
        check whether the pre_group.type is equal to the post_group.type, only if they are equal, return flag=Ture.

        Args:
            pre_group(Groups): the neuron group which need to be connected in the pre_assembly.
                   post_group(Groups): the neuron group which need to connect the pre_group in the post_assembly.

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


class sparse_connect_sparse(Connection):

    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse_connect', 'conv','...'), policies=[],
                 max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):
        super(sparse_connect_sparse, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name,
                                                    link_type=link_type, policies=policies, max_delay=max_delay,
                                                    sparse_with_mask=sparse_with_mask,
                                                    pre_var_name=pre_var_name, post_var_name=post_var_name, **kwargs)
        self.weight = kwargs.get('weight', None)
        self.w_std = kwargs.get('w_std', 0.05)
        self.density = kwargs.get('density', 0.1)
        self.w_max = kwargs.get('w_max', None)
        self.w_min = kwargs.get('w_min', None)
        self.param_init = kwargs.get('param_init', None)
        self.is_parameter = kwargs.get('is_parameter', False)
        self.is_sparse = kwargs.get('is_sparse', True)


    def unit_connect(self, pre_group, post_group):
        # The number of neurons in neuron group
        pre_num = pre_group.num
        post_num = post_group.num
        link_num = pre_num * post_num
        shape = (post_num, pre_num)

        if self.weight is None:
            # Connection weight
            sparse_matrix = self.w_std * sp.rand(post_num, pre_num, density=self.density, format='csr')
            weight = sparse_matrix.toarray()
        else:
            assert (self.weight.shape == shape), f"The size of the given weight {self.weight.shape} does not correspond to the size of synaptic matrix {shape} "
            weight = self.weight

        # The name for backend variables
        input_name = self.get_input_name(pre_group, post_group)
        weight_name = self.get_link_name(pre_group, post_group, 'weight')
        target_name = self.get_post_name(post_group, self.post_var_name)

        # The backend variable
        var_code = (weight_name, shape, weight, self.is_parameter, self.is_sparse, self.param_init)  # (var_name, shape, value, is_parameter, is_sparse)

        # The backend basic operation
        op_code = [target_name, 'sparse_mat_mult_weight', weight_name, input_name]
        connection_information = (pre_group, post_group, link_num, var_code, op_code)
        self.unit_connections.append(connection_information)

        pass

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

    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse_connect', 'conv','...'), policies=[],
                 max_delay=0, sparse_with_mask=True, pre_var_name='O', post_var_name='WgtSum', **kwargs):
        super(sparse_connect_mask, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name,
                                                  link_type=link_type, policies=policies, max_delay=max_delay,
                                                  sparse_with_mask=sparse_with_mask,
                                                  pre_var_name=pre_var_name, post_var_name=post_var_name, **kwargs)
        self.weight = kwargs.get('weight', None)
        self.w_std = kwargs.get('w_std', 0.05)
        self.w_mean = kwargs.get('w_mean', 0.005)
        self.density = kwargs.get('density', 0.1)
        self.w_max = kwargs.get('w_max', None)
        self.w_min = kwargs.get('w_min', None)
        self.param_init = kwargs.get('param_init', None)
        self.is_parameter = kwargs.get('is_parameter', True)
        self.is_sparse = kwargs.get('is_sparse', False)

    def unit_connect(self, pre_group, post_group):
        # The number of neurons in neuron group
        pre_num = pre_group.num
        post_num = post_group.num
        link_num = pre_num * post_num
        shape = (post_num, pre_num)

        if self.weight is None:
            # Connection weight
            sparse_matrix = self.w_std * sp.rand(post_num, pre_num, density=self.density, format='csr')
            weight = sparse_matrix.toarray()
            weight[weight.nonzero()] = weight[weight.nonzero()] + self.w_mean
        else:
            assert (self.weight.shape == shape), f"The size of the given weight {self.weight.shape} does not correspond to the size of synaptic matrix {shape} "
            weight = self.weight

        # The name for backend variables
        input_name = self.get_input_name(pre_group, post_group)
        weight_name = self.get_link_name(pre_group, post_group, 'weight')
        target_name = self.get_post_name(post_group, self.post_var_name)

        # The backend variable
        var_code = (weight_name, shape, weight, self.is_parameter, self.is_sparse, self.param_init)  # (var_name, shape, value, is_parameter, is_sparse)

        # The backend basic operation
        op_code = [target_name, 'mat_mult_weight', input_name, weight_name]

        connection_information = (pre_group, post_group, link_num, var_code, op_code)
        self.unit_connections.append(connection_information)

        if self.sparse_with_mask:
            mask = (weight != 0)
            mask_name = self.get_link_name(pre_group, post_group, 'mask')
            mask_var_code = (mask_name, shape, mask)
            mask_op = (weight_name, self.mask_operation, [weight_name, mask_name])
            mask_information = (mask_var_code, mask_op)
            self.mask_info.append(mask_information)

        pass

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

    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse_connect', 'conv','...'), policies=[],
                 max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):
        super(random_connect_sparse, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name,
                                                    link_type=link_type, policies=policies, max_delay=max_delay,
                                                    sparse_with_mask=sparse_with_mask,
                                                    pre_var_name=pre_var_name, post_var_name=post_var_name, **kwargs)
        self.weight = kwargs.get('weight', None)
        self.probability = kwargs.get('probability', 0.1)
        self.w_max = kwargs.get('w_max', None)
        self.w_min = kwargs.get('w_min', None)
        self.param_init = kwargs.get('param_init', None)
        self.is_parameter = kwargs.get('is_parameter', False)
        self.is_sparse = kwargs.get('is_sparse', True)

    def unit_connect(self, pre_group, post_group):
        # The number of neurons in neuron group
        pre_num = pre_group.num
        post_num = post_group.num
        link_num = pre_num * post_num
        shape = (post_num, pre_num)

        if self.weight is None:
            # Link_parameters
            prob_weight = np.random.rand(*shape)
            diag_index = np.arange(min([pre_num, post_num]))
            prob_weight[diag_index, diag_index] = 1
            index = (prob_weight < self.probability)
            # Connection weight
            weight = np.zeros(shape)
            weight[index] = prob_weight[index]
        else:
            assert (self.weight.shape == shape), f"The size of the given weight {self.weight.shape} does not correspond to the size of synaptic matrix {shape} "
            weight = self.weight

        # The name for backend variables
        input_name = self.get_input_name(pre_group, post_group)
        weight_name = self.get_link_name(pre_group, post_group, 'weight')
        target_name = self.get_post_name(post_group, self.post_var_name)

        # The backend variable
        var_code = (weight_name, shape, weight, self.is_parameter, self.is_sparse, self.param_init)   # (var_name, shape, value, is_parameter, is_sparse)

        # The backend basic operation
        op_code = [target_name, 'sparse_mat_mult_weight', weight_name, input_name]

        connection_information = (pre_group, post_group, link_num, var_code, op_code)
        self.unit_connections.append(connection_information)
        pass

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

    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse_connect', 'conv', '...'), policies=[],
                 max_delay=0, sparse_with_mask=True, pre_var_name='O', post_var_name='WgtSum', **kwargs):
        super(random_connect_mask, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name,
                                                  link_type=link_type, policies=policies, max_delay=max_delay,
                                                  sparse_with_mask=sparse_with_mask,
                                                  pre_var_name=pre_var_name, post_var_name=post_var_name, **kwargs)
        self.weight = kwargs.get('weight', None)
        self.probability = kwargs.get('probability', 0.1)
        self.w_max = kwargs.get('w_max', None)
        self.w_min = kwargs.get('w_min', None)
        self.param_init = kwargs.get('param_init', None)
        self.is_parameter = kwargs.get('is_parameter', True)
        self.is_sparse = kwargs.get('is_sparse', False)

    def unit_connect(self, pre_group, post_group):
        # The number of neurons in neuron group
        pre_num = pre_group.num
        post_num = post_group.num
        link_num = pre_num * post_num
        shape = (post_num, pre_num)

        if self.weight is None:
            # Link_parameters
            prob_weight = np.random.rand(*shape)
            diag_index = np.arange(min([pre_num, post_num]))
            prob_weight[diag_index, diag_index] = 1
            index = (prob_weight < self.probability)
            # Connection weight
            weight = np.zeros(shape)
            weight[index] = prob_weight[index]
        else:
            assert (self.weight.shape == shape), f"The size of the given weight {self.weight.shape} does not correspond to the size of synaptic matrix {shape} "
            weight = self.weight

        # The name for backend variables
        input_name = self.get_input_name(pre_group, post_group)
        weight_name = self.get_link_name(pre_group, post_group, 'weight')
        target_name = self.get_post_name(post_group, self.post_var_name)

        # The backend variable
        var_code = (weight_name, shape, weight, self.is_parameter, self.is_sparse, self.param_init)    # (var_name, shape, value, is_parameter, is_sparse)

        # The backend basic operation
        op_code = [target_name, 'mat_mult_weight', input_name, weight_name]

        connection_information = (pre_group, post_group, link_num, var_code, op_code)
        self.unit_connections.append(connection_information)

        if self.sparse_with_mask:
            mask = (weight != 0)
            mask_name = self.get_link_name(pre_group, post_group, 'mask')
            mask_var_code = (mask_name, shape, mask)
            mask_op = (weight_name, self.mask_operation, [weight_name, mask_name])
            mask_information = (mask_var_code, mask_op)
            self.mask_info.append(mask_information)
        pass

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


class DistDepd_connect(Connection):

    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse', 'conv','...'), policies=[],
                 max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):

        super(DistDepd_connect, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name, link_type=link_type,
                                             policies=policies, max_delay=max_delay, sparse_with_mask=sparse_with_mask,
                                             pre_var_name=pre_var_name, post_var_name=post_var_name, **kwargs)
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
            weights = weights * (dist!=0).float()
        # import matplotlib.pyplot as plt
        # plt.imshow(weights, aspect='auto')
        # plt.show()
        return weights
Connection.register('dist_depd', DistDepd_connect)


class reconnect(Connection):
    def __init__(self, pre_assembly, post_assembly, name=None, link_type=('full', 'sparse_connect', 'conv', '...'), policies=[],
                 max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='WgtSum', **kwargs):
        super(reconnect, self).__init__(pre_assembly=pre_assembly, post_assembly=post_assembly, name=name, link_type=link_type,
                                             policies=policies, max_delay=max_delay, sparse_with_mask=sparse_with_mask, pre_var_name=pre_var_name, post_var_name=post_var_name, **kwargs)
    def unit_connect(self, pre_group, post_group):
        pass

    def condition_check(self, pre_group, post_group):
        pass