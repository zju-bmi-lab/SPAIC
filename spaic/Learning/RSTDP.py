# encoding: utf-8
"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: RSTDP.py
@time:2021/4/8 10:46
@description:
"""

from .Learner import Learner
import numpy as np
import torch

class RSTDP(Learner):
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
        super(RSTDP, self).__init__(trainable=trainable)
        self.prefered_backend = ['pytorch']
        self.name = 'RSTDP'
        self.learning_rate = kwargs.get('lr', 0.1)

        self._tau_constant_variables = dict()
        self._tau_constant_variables['tau_plus'] = kwargs.get('tau_plus', 20.0)
        self._tau_constant_variables['tau_minus'] = kwargs.get('tau_minus', 20.0)

        self._constant_variables = dict()
        self._constant_variables['A_plus'] = kwargs.get('A_plus', 1.0)
        self._constant_variables['A_minus'] = kwargs.get('A_minus', -1.0)

    def weight_update(self, weight, eligibility, reward):
        """
        RSTDP learning rule for ``Connection`` subclass of ``AbstractConnection`` class.
        Args:
            weight : weight between pre and post neurongroup
            eligibility: a decaying memory of the relationships between the recent pairs of pre and postsynaptic spike pairs
            reward: reward signal
        """
        # Compute weight update based on the eligibility value of the past timestep.
        with torch.no_grad():
            if len(reward.shape) > 1 and reward.shape[1] == eligibility.shape[0]:
                reward = reward.transpose(1, 0)
                reward = reward.repeat(1, eligibility.shape[1])
            weight.add_(self.learning_rate * reward * eligibility)
        return weight

    def build(self, backend):
        super(RSTDP, self).build(backend)
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
            elif isinstance(var, list):
                if len(var) > 1:
                    var_len = len(var)
                    shape = (1, var_len)  # (1, shape)
                else:
                    shape = ()
            else:
                shape = ()
            self.variable_to_backend(key, shape, value=var)

        permute_name = 'rstdp_permute_dim'
        permute_dim_value = [1, 0]
        self.variable_to_backend(permute_name, shape=None, value=permute_dim_value, is_constant=True)
        reward_name = 'Output_Reward[updated]'

        # Traverse all trainable connections
        for conn in self.trainable_connections.values():
            preg = conn.pre
            postg = conn.post
            pre_name = conn.get_input_name(preg, postg)
            post_name = conn.get_group_name(postg, 'O')
            weight_name = conn.get_link_name(preg, postg, 'weight')

            # p_plus tracks the influence of presynaptic spikes; p_minus tracks the influence of postsynaptic spikes
            p_plus_name = pre_name + '_{p_plus}'
            p_minus_name = post_name + '_{p_minus}'
            eligibility_name = weight_name + '_{eligibility}'
            pre_shape_temp = backend._variables[pre_name].shape
            if len(pre_shape_temp) > 2 and len(pre_shape_temp) == 4:
                pre_shape = [pre_shape_temp[0], pre_shape_temp[1] * pre_shape_temp[2] * pre_shape_temp[3]]
            else:
                pre_shape = pre_shape_temp
            self.variable_to_backend(p_plus_name, pre_shape, value=0.0)
            self.variable_to_backend(p_minus_name, backend._variables[post_name].shape, value=0.0)
            self.variable_to_backend(eligibility_name, backend._variables[weight_name].shape, value=0.0)

            # Equations
            # p_plus *= np.exp(-dt / tau_plus)
            # p_plus += A_plus * pre
            # p_minus *= np.exp(-dt / tau_minus)
            # p_minus += A_minus * post
            # eligibility = torch.matmul(post.transpose(1, 0), p_plus) + torch.matmul(p_minus.transpose(1, 0), pre)

            # Update p_plus values
            # self.op_to_backend('p_plus_temp', 'var_mult', 'A_plus', pre_name))
            # self.op_to_backend(p_plus_name, 'var_linear', 'tau_plus', p_plus_name, 'p_plus_temp'))
            self.op_to_backend('p_plus_temp', 'var_mult', ['tau_plus', p_plus_name])
            if len(pre_shape_temp) > 2 and len(pre_shape_temp) == 4:
                # if pre layer is 2d feature map
                self.op_to_backend('pre_name_temp', 'feature_map_flatten', pre_name)
                self.op_to_backend(p_plus_name, 'var_linear', ['A_plus', 'pre_name_temp', 'p_plus_temp'])
            else:
                self.op_to_backend(p_plus_name, 'var_linear', ['A_plus', pre_name, 'p_plus_temp'])

            # Update p_minus values
            # self.op_to_backend('p_minus_temp', 'var_mult', 'A_minus', post_name))
            # self.op_to_backend(p_minus_name, 'var_linear', 'tau_minus', p_minus_name, 'p_minus_temp'))
            self.op_to_backend('p_minus_temp', 'var_mult', ['tau_minus', p_minus_name])
            self.op_to_backend(p_minus_name, 'var_linear', ['A_minus', post_name, 'p_minus_temp'])

            # Calculate point eligibility value
            self.op_to_backend('post_permute', 'permute', [post_name, permute_name])
            self.op_to_backend('pre_post', 'mat_mult', ['post_permute', p_plus_name + '[updated]'])

            self.op_to_backend('p_minus_permute', 'permute', [p_minus_name + '[updated]', permute_name])
            if len(pre_shape_temp) > 2 and len(pre_shape_temp) == 4:
                self.op_to_backend('post_pre', 'mat_mult', ['p_minus_permute', 'pre_name_temp'])
            else:
                self.op_to_backend('post_pre', 'mat_mult', ['p_minus_permute', pre_name])
            self.op_to_backend(eligibility_name, 'add', ['pre_post', 'post_pre'])
            self.op_to_backend(weight_name, self.weight_update, [weight_name, eligibility_name, reward_name])
Learner.register('rstdp', RSTDP)


class RSTDPET(Learner):
    """
    Reward-modulated STDP with eligibility trace. Adapted from `(Florian 2007)
    <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`.
    Args:
        lr (int or float): learning rate
        trainable: It can be network or neurongroups
    Attributes:
        tau_plus (int or float): Time constant for pre -synaptic firing trace determines the range of interspike intervals over which synaptic occur.
        tau_minus (int or float): Time constant for post-synaptic firing trace.
        a_plus (float): Learning rate for post-synaptic.
        a_minus (flaot): Learning rate for pre-synaptic.
    Notes:
        Batch_size for network using RSTDPET as learning algorithm must be 1.
    """

    def __init__(self, lr, trainable=None, **kwargs):
        super(RSTDPET, self).__init__(trainable=trainable)

        self.prefered_backend = ['pytorch']
        self.name = 'RSTDPET'
        self.learning_rate = lr

        self._tau_constant_variables = dict()
        self._tau_constant_variables['tau_plus'] = kwargs.get('tau_plus', 20.0)
        self._tau_constant_variables['tau_minus'] = kwargs.get('tau_minus', 20.0)
        self._tau_constant_variables['tau_e_trace'] = kwargs.get('tau_e_trace', 25.0)

        self._tau_membrane_variables = dict()
        self._tau_membrane_variables['tau_e'] = kwargs.get('tau_e', 25.0)

        self._constant_variables = dict()
        self._constant_variables['A_plus'] = kwargs.get('A_plus', 1.0)
        self._constant_variables['A_minus'] = kwargs.get('A_minus', -1.0)

    def weight_update(self, weight, eligibility_trace, reward):
        """
        RSTDPET learning rule for ``Connection`` subclass of ``AbstractConnection`` class.
        Notes:
            The batch of pre and post should be 1
        Args:
            weight : weight between pre and post neurongroup
            eligibility_trace: a decaying memory of the relationships between the recent pairs of pre and postsynaptic spike pairs
            reward: reward signal
        """
        with torch.no_grad():
            # Keep parameters consistent through inplace operation
            if len(reward.shape) > 1 and reward.shape[1] == eligibility_trace.shape[0]:
                reward = reward.transpose(1, 0)
                reward = reward.repeat(1, eligibility_trace.shape[1])
            weight.add_(self.learning_rate * self.dt * reward * eligibility_trace)
        return weight

    def build(self, backend):
        super(RSTDPET, self).build(backend)

        view_name = 'rstdpet_view_dim'
        view_dim_value = [-1]
        self.variable_to_backend(view_name, shape=None, value=view_dim_value, is_constant=True)
        reward_name = 'Output_Reward[updated]'

        # Traverse all trainable connections
        for conn in self.trainable_connections.values():

            preg = conn.pre
            postg = conn.post
            pre_name = conn.get_input_name(preg, postg)
            post_name = conn.get_group_name(postg, 'O')
            weight_name = conn.get_link_name(preg, postg, 'weight')

            # p_plus tracks the influence of presynaptic spikes; p_minus tracks the influence of postsynaptic spikes
            p_plus_name = pre_name + '_{p_plus}'
            p_minus_name = post_name + '_{p_minus}'
            eligibility_name = weight_name + '_{eligibility}'
            eligibility_trace_name = weight_name + '_{eligibility_trace}'
            self.variable_to_backend(p_plus_name, backend._variables[pre_name].shape[1:], value=0.0)
            self.variable_to_backend(p_minus_name, backend._variables[post_name].shape[1:], value=0.0)
            self.variable_to_backend(eligibility_name, backend._variables[weight_name].shape, value=0.0)
            self.variable_to_backend(eligibility_trace_name, backend._variables[weight_name].shape, value=0.0)

            # Equations
            # pre = pre.view(-1)
            # post = post.view(-1)
            # p_plus *= np.exp(-dt / tau_plus)
            # p_plus += A_plus * pre
            # p_minus *= np.exp(-dt / tau_minus)
            # p_minus += A_minus * post
            # eligibility = torch.ger(post, p_plus) + torch.ger(p_minus, pre)
            # eligibility_trace *= np.exp(-dt / tau_e_trace)
            # eligibility_trace += eligibility / tau_e

            # Update p_plus values
            self.op_to_backend('pre_view', 'view', [pre_name, view_name])
            # self.op_to_backend('p_plus_temp', 'var_mult', 'A_plus', 'pre_view'))
            # self.op_to_backend(p_plus_name, 'var_linear', 'tau_plus', p_plus_name, 'p_plus_temp'))
            self.op_to_backend('p_plus_temp', 'var_mult', ['tau_plus', p_plus_name])
            self.op_to_backend(p_plus_name, 'var_linear', ['A_plus', 'pre_view', 'p_plus_temp'])

            # Update p_minus values
            self.op_to_backend('post_view', 'view', [post_name, view_name])
            # self.op_to_backend('p_minus_temp', 'var_mult', 'A_minus', 'post_view'))
            # self.op_to_backend(p_minus_name, 'var_linear', 'tau_minus', p_minus_name, 'p_minus_temp'))
            self.op_to_backend('p_minus_temp', 'var_mult', ['tau_minus', p_minus_name])
            self.op_to_backend(p_minus_name, 'var_linear', ['A_minus', 'post_view', 'p_minus_temp'])

            # Calculate point eligibility value
            self.op_to_backend('pre_post', 'ger', ['post_view', p_plus_name + '[updated]'])
            self.op_to_backend('post_pre', 'ger', [p_minus_name + '[updated]', 'pre_view'])
            self.op_to_backend(eligibility_name, 'add', ['pre_post', 'post_pre'])
            self.op_to_backend('eligibility_trace_temp', 'var_mult', ['tau_e', eligibility_name + '[updated]'])
            self.op_to_backend(eligibility_trace_name, 'var_linear', ['tau_e_trace', eligibility_trace_name, 'eligibility_trace_temp'])

            self.op_to_backend(weight_name, self.weight_update, [weight_name, eligibility_trace_name, reward_name])

Learner.register('rstdpet', RSTDPET)


class RewardSwitchSTDP(Learner):
    def __init__(self, trainable=None, **kwargs):
        super(RewardSwitchSTDP, self).__init__(trainable=trainable, **kwargs)
        self.trainable = trainable
        self.prefered_backend = ['pytorch']
        self.name = 'Reward_Switch_STDP'
        self._constant_variables = dict()
        self._constant_variables['Apost_pos'] = kwargs.get('Apost_pos', 5.0e-3)
        self._constant_variables['Apre_pos'] = kwargs.get('Apre_pos', -4.0e-3)
        self._constant_variables['Apost_neg'] = kwargs.get('Apost_neg', -5.0e-3)
        self._constant_variables['Apre_neg'] = kwargs.get('Apre_neg', 4.0e-3)
        self._constant_variables['pre_decay'] = kwargs.get('pre_decay', np.exp(-1/20.0))
        post_decay = kwargs.get('post_decay', np.exp(-1/20.0))
        self._constant_variables['post_decay'] = post_decay
        self._constant_variables['homoestatic'] = kwargs.get('homoestatic', 1.0e-10)
        m_rate = kwargs.get("m_rate", 20.0)
        self._constant_variables['m_rate'] = m_rate*(1-post_decay)/(1000.0*post_decay)

        self.w_min = kwargs.get('w_min', 0.0)
        self.w_max = kwargs.get('w_max', 0.5)
        self.w_norm = 1.2
        self.w_mean = None
        self.lr = kwargs.get('lr',0.1)
        self.param_run_update = True
        self.reward_name = kwargs.get('reward_name','Output_Reward[updated]')

    def update(self, input, output, reward, input_trace, output_trace, pre_decay,
               post_decay, Apost_pos, Apost_neg, Apre_pos, Apre_neg, m_rate, homoestatic, weight):
        if self.w_mean is None:
            self.w_mean = torch.mean(weight, dim=1, keepdim=True).detach()
            self.aw_mean = torch.mean(self.w_mean)
        if self.training:
            input_trace = pre_decay * input_trace * input.le(0.0) + input
            output_trace = post_decay * output_trace * output.le(0.0) + output
            # with torch.no_grad():
            #     self.w_mean = self.w_mean * self.aw_mean / torch.mean(self.w_mean)
            Apost = reward.gt(0) * Apost_pos + reward.le(0) * Apost_neg
            Apre = reward.gt(0) * Apre_pos + reward.le(0) * Apre_neg
            pre_post = torch.matmul(output.permute(1, 0), Apost*input_trace)
            post_pre = torch.matmul((Apre*output_trace).permute(1, 0), input)
            dw = (pre_post + post_pre)/(1.0*input.shape[0]) # + torch.mean((m_rate-output_trace)*homoestatic,dim=0).unsqueeze(dim=1)

            weight = weight + dw
            weight = torch.clamp(weight, 0, 0.1)
            return input_trace, output_trace, weight

    def build(self, backend):
        self._constant_variables['m_rate'] = self._constant_variables['m_rate']*backend.dt
        self._constant_variables['homoestatic'] = self._constant_variables['homoestatic'] / self._constant_variables['m_rate']
        super(RewardSwitchSTDP, self).build(backend)
        self.dt = backend.dt
        self.run_time = backend.runtime


        for conn in self.trainable_connections.values():
            preg = conn.pre
            postg = conn.post
            pre_name = conn.get_input_name(preg, postg)
            post_name = conn.get_group_name(postg, 'O')
            weight_name = conn.get_link_name(preg, postg, 'weight')

            input_trace_name = conn.id + '_{input_trace}'
            output_trace_name = conn.id + '_{output_trace[stay]}'
            self.variable_to_backend(input_trace_name, backend._variables[pre_name].shape, value=0.0)
            self.variable_to_backend(output_trace_name, backend._variables[post_name].shape, value=0.0)

            self.op_to_backend([input_trace_name, output_trace_name, weight_name], self.update,
                                   [pre_name, post_name, self.reward_name, input_trace_name, output_trace_name, 'pre_decay',
                                    'post_decay', 'Apost_pos', 'Apost_neg', 'Apre_pos', 'Apre_neg', 'm_rate', 'homoestatic', weight_name])
Learner.register('switch_rstdp', RewardSwitchSTDP)
