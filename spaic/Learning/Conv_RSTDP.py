# encoding: utf-8
"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: Conv_RSTDP.py
@time:2021/12/30 13:49
@description:
"""
from .Learner import Learner
from ..Network.BaseModule import Op
from ..Network.Connections import conv_connect
from ..IO.utils import im2col
import numpy as np
import torch

class Conv2d_RSTDP(Learner):
    """
    Reward-modulated STDP. Adapted from `(Florian 2007)
    <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`.
    Args:
        lr (int or float): learning rate
        trainable: It can be network or neurongroups
    Attributes:
        tau_plus (int or float): Time constant for pre -synaptic firing trace determines the range of interspike intervals over which synaptic occur.
        tau_minus (int or float): Time constant for post-synaptic firing trace.
        a_plus (float): Learning rate for post-synaptic.
        a_minus (flaot): Learning rate for pre-synaptic.
    """

    def __init__(self, trainable=None, **kwargs):
        super(Conv2d_RSTDP, self).__init__(trainable=trainable)
        self.prefered_backend = ['pytorch']
        self.name = 'Conv2d_RSTDP'
        self.learning_rate = kwargs.get('lr', 0.1)

        self._tau_constant_variables = dict()
        self._tau_constant_variables['tau_plus'] = kwargs.get('tau_plus', 20.0)
        self._tau_constant_variables['tau_minus'] = kwargs.get('tau_minus', 20.0)

        self._constant_variables = dict()
        self._constant_variables['A_plus'] = kwargs.get('A_plus', 1.0)
        self._constant_variables['A_minus'] = kwargs.get('A_minus', -1.0)

    def weight_update(self, weight, eligibility, reward):
        """
        Conv2d_RSTDP learning rule for ``conv_connect`` subclass of ``Connection`` class.
        Args:
            weight : weight between pre and post neurongroup
            eligibility: a decaying memory of the relationships between the recent pairs of pre and postsynaptic spike pairs
            reward: reward signal
        """
        # Compute weight update based on the eligibility value of the past timestep.
        with torch.no_grad():
            assert reward.ndim <= 1, 'The reward for conv2d_rstdp should be a global reward'
            weight.add_(self.learning_rate * reward * eligibility)
            return weight

    def build(self, backend):
        super(Conv2d_RSTDP, self).build(backend)
        self.dt = backend.dt

        for (key, tau_var) in self._tau_constant_variables.items():
            tau_var = np.exp(-self.dt / tau_var)
            shape = ()
            self.variable_to_backend(key, shape, value=tau_var)

        for (key, var) in self._constant_variables.items():
            if isinstance(var, np.ndarray):
                if var.size > 1:
                    var_shape = var.shape
                    shape = (1, *var_shape)  # (1, shape)
                else:
                    shape = ()
            elif isinstance(var, list) or isinstance(var, tuple):
                if len(var) > 1:
                    var_len = len(var)
                    shape = (1, var_len)  # (1, shape)
                else:
                    shape = ()
            else:
                shape = ()
            self.variable_to_backend(key, shape, value=var)

        permute_name = 'conv2d_rstdp_permute_dim'
        permute_dim_value = [0, 2, 1]
        self.variable_to_backend(permute_name, shape=None, value=permute_dim_value, is_constant=True)
        reward_name = 'Output_Reward[updated]'

        # Traverse all trainable connections
        for conn in self.trainable_connections.values():
            if not isinstance(conn, conv_connect):
                raise ValueError('Conv2d_RSTDP can only modify the connection defined by conv_connect, not the %s'% str(type(conn)))
            preg = conn.pre
            postg = conn.post
            pre_name = conn.get_input_name(preg, postg)
            post_name = conn.get_group_name(postg, 'O')
            weight_name = conn.get_link_name(preg, postg, 'weight')
            out_channels, in_channels, kh, kw = backend._variables[weight_name].size()  # (out_channels, in_channels, kh, kw)
            padding = conn.padding
            stride = conn.stride

            # p_plus tracks the influence of presynaptic spikes
            p_plus_name = pre_name + '_{p_plus}'
            # p_minus tracks the influence of postsynaptic spikes
            p_minus_name = post_name + '_{p_minus}'
            eligibility_name = weight_name + '_{eligibility}'
            kh_name = weight_name + '_{kh}'
            kw_name = weight_name + '_{kw}'
            padding_name = weight_name + '_{padding}'
            stride_name = weight_name + '_{stride}'
            view_name = weight_name + '_{conv2d_rstdp_view_dim}'

            p_plus_value_temp = np.zeros(backend._variables[pre_name].shape)
            p_plus_value = im2col(p_plus_value_temp, kh, kw, stride, padding)
            p_minus_value_temp = np.zeros(backend._variables[post_name].shape)  # (batch_size, out_channels, height, width)
            p_minus_value = p_minus_value_temp.reshape(p_minus_value_temp.shape[0], p_minus_value_temp.shape[1], -1)
            view_dim_value = [out_channels, in_channels, kh, kw]

            self.variable_to_backend(p_plus_name, p_plus_value.shape, value=p_plus_value)
            self.variable_to_backend(p_minus_name, p_minus_value.shape, value=p_minus_value)
            self.variable_to_backend(eligibility_name, backend._variables[weight_name].shape, value=0.0)
            self.variable_to_backend(kh_name, shape=(), value=kh, is_constant=True)
            self.variable_to_backend(kw_name, shape=(), value=kw, is_constant=True)
            self.variable_to_backend(padding_name, shape=(), value=padding, is_constant=True)
            self.variable_to_backend(stride_name, shape=(), value=stride, is_constant=True)
            self.variable_to_backend(view_name, shape=None, value=view_dim_value, is_constant=True)

            sum_dim_name = 'conv2d_rstdp_sum'
            sum_dim_value = 0
            self.variable_to_backend(sum_dim_name, shape=(), value=sum_dim_value, is_constant=True)

            # Equations
            # pre_temp = im2col_indices(pre, kh, kw, padding, stride)
            # p_plus *= np.exp(-dt / tau_plus)
            # p_plus += A_plus * pre_temp
            # post_temp = post.view(batch_size, out_channels, -1)
            # p_minus *= np.exp(-dt / tau_minus)
            # p_minus += A_minus * post_temp
            # eligibility = torch.bmm(post, p_plus.permute((0, 2, 1))) + torch.bmm(p_minus, pre.permute((0, 2, 1)))
            # eligibility = torch.sum(eligibility, dim=0)
            # eligibility = eligibility.view([out_channels, in_channels, kh, kw])

            # Update p_plus values
            self.op_to_backend('pre_name_temp', 'im2col_indices', [pre_name, kh_name, kw_name, padding_name, stride_name])  # (batch_size, channels*kh*kw, height*width)
            # self.op_to_backend('p_plus_temp', 'var_mult', 'A_plus', 'pre_name_temp'))
            # self.op_to_backend(p_plus_name, 'var_linear', 'tau_plus', p_plus_name, 'p_plus_temp'))
            self.op_to_backend('p_plus_temp', 'var_mult', ['tau_plus', p_plus_name])
            self.op_to_backend(p_plus_name, 'var_linear', 'A_plus', 'pre_name_temp', 'p_plus_temp')

            # Update p_minus values
            self.op_to_backend('post_name_temp', 'conv2d_flatten', post_name)  # (batch_size, out_channels, height * width)
            # self.op_to_backend('p_minus_temp', 'var_mult', 'A_minus', 'post_name_temp'))
            # self.op_to_backend(p_minus_name, 'var_linear', 'tau_minus', p_minus_name, 'p_minus_temp'))
            self.op_to_backend('p_minus_temp', 'var_mult', ['tau_minus', p_minus_name])
            self.op_to_backend(p_minus_name, 'var_linear', ['A_minus', 'post_name_temp', 'p_minus_temp'])

            # Calculate point eligibility value
            self.op_to_backend('p_plus_permute', 'permute', [p_plus_name + '[updated]', permute_name])
            self.op_to_backend('pre_post', 'bmm', ['post_name_temp', 'p_plus_permute'])

            self.op_to_backend('pre_permute', 'permute', ['pre_name_temp', permute_name])
            self.op_to_backend('post_pre', 'bmm', [p_minus_name + '[updated]', 'pre_permute'])
            self.op_to_backend('eligibility_temp', ['add', 'pre_post', 'post_pre'])
            self.op_to_backend('eligibility_sum', 'reduce_sum', ['eligibility_temp', sum_dim_name])
            self.op_to_backend(eligibility_name, 'view', ['eligibility_sum', view_name])

            self.op_to_backend(weight_name, self.weight_update, [weight_name, eligibility_name, reward_name])

Learner.register('conv2d_rstdp', Conv2d_RSTDP)


