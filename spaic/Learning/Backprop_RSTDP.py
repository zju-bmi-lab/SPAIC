# -*- coding: utf-8 -*-
"""
Created on 2022/4/27
@project: SPAIC
@filename: Backprop-RSTDP
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description: 
"""
from .Learner import Learner
from ..Backend.Backend import Backend
import numpy as np
import torch


# a = torch.rand((2,1))
# b = torch.rand((1))
# print(torch.matmul(a,b))

class Backprop_RSTDP(Learner):
    _learner_count = 0
    
    def __init__(self, trainable=None, **kwargs):
        super(Backprop_RSTDP, self).__init__(trainable=trainable)
        self.prefered_backend = ['pytorch']
        self.name = 'Backprop-RSTDP' + str(Backprop_RSTDP._learner_count)
        Backprop_RSTDP._learner_count += 1
        self._tau_constant_variables = dict()
        self._constant_variables = dict()
        self._variables = dict()

        self.learning_rate = kwargs.get('lr', 0.1)
        self.homeostatic_rate = kwargs.get('hs_r',0.01)
        self.reward_var_names = kwargs.get('reward_vars', 'Output_Reward')
        self.reward_weight_min = kwargs.get('reward_wegith_min', -10.0)
        self.reward_weight_max = kwargs.get('reward_wegith_max', 10.0)

        self._tau_constant_variables['tau_pre'] = kwargs.get('tau_pre', 5.0)
        self._tau_constant_variables['tau_post'] = kwargs.get('tau_post', 12.0)
        self._constant_variables['A_plus'] = kwargs.get('A_plus', 1.0)
        self._constant_variables['A_minus'] = kwargs.get('A_minus', -0.4)
        self.traces = []
        self.spks = []


    def spk_trace_update(self, trace, beta, spk):
        with torch.no_grad():
            trace = beta*trace + (1-beta)*spk
            # if list(trace.shape) == [200,400]:
            #     self.traces.append(trace)
            #     self.spks.append(spk)
        return trace


    # def RSTDP_update(self, trace_pre, trace_post, beta_pre, beta_post, spk_pre, spk_post, A_plus, A_minus):
    #     with torch.no_grad():
    #         trace_pre = beta_pre*trace_pre + spk_pre
    #         trace_post = beta_post*trace_post + spk_post
    #         if list(trace_pre.shape) == [200,784]:
    #             self.traces.append(trace_post)
    #             self.spks.append(spk_post)
    #         eligibility = torch.mean(A_plus*trace_pre.unsqueeze(1)*spk_post.unsqueeze(2) + A_minus*trace_post.unsqueeze(2)*spk_pre.unsqueeze(1), 0)
    #         homeo_eligibility = torch.mean(trace_pre.unsqueeze(1)*(trace_post**2).unsqueeze(2), 0)
    #     return trace_pre, trace_post, eligibility, homeo_eligibility

    def RSTDP_update(self, weight, trace_pre, trace_post,  spk_pre, spk_post, A_plus, A_minus, reward_weight, reward):

        with torch.no_grad():
            reward = torch.matmul(reward, reward_weight.t()) # shape of post neuron (pre of reward_weight)
            reward = reward.unsqueeze(2)

            eligibility = torch.mean(A_plus*torch.matmul(spk_post.unsqueeze(2), trace_pre.unsqueeze(1)) + A_minus*torch.matmul(trace_post.unsqueeze(2), spk_pre.unsqueeze(1)), 0)
            homeo_eligibility = torch.mean(torch.matmul((trace_post - 0.1).unsqueeze(2),trace_pre.unsqueeze(1)), 0)

            delta_w = torch.mean(self.learning_rate*reward*eligibility - self.homeostatic_rate*homeo_eligibility,0)
            weight.add_(delta_w)
        return weight

    def build(self, backend: Backend):
        super(Backprop_RSTDP, self).build(backend)
        self.dt = backend.dt

        for (key, tau_var) in self._tau_constant_variables.items():
            tau_var = np.exp(-self.dt / tau_var)
            shape = ()
            self.variable_to_backend(self.get_varname(key), shape, value=tau_var)

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
            self.variable_to_backend(self.get_varname(key), shape, value=var)

        if isinstance(self.reward_var_names, str):
            single_reward = True
        else:
            assert len(self.reward_var_names) == len(self.trainable_connections)
            single_reward = False

        for ind, conn in enumerate(self.trainable_connections.values()):
            preg = conn.pre
            postg = conn.post
            pre_name = conn.get_pre_name(preg, 'O')
            post_name = conn.get_post_name(postg, 'O')
            weight_name = conn.get_link_name(preg, postg, 'weight')

            trace_pre_name = preg.get_labeled_name('spk_trace')
            if not backend.has_variable(trace_pre_name):
                self.variable_to_backend(trace_pre_name, backend.get_varialble(pre_name).shape, value=0.0)
                self.op_to_backend([trace_pre_name], self.spk_trace_update, [trace_pre_name, self.get_varname('tau_pre'), pre_name])
            trace_post_name = postg.get_labeled_name('spk_trace')
            if not backend.has_variable(trace_post_name):
                self.variable_to_backend(trace_post_name, backend.get_varialble(post_name).shape, value=0.0)
                self.op_to_backend([trace_post_name], self.spk_trace_update,
                                       [trace_post_name, self.get_varname('tau_post'), post_name])
            # eligibility_name = conn.get_link_name(preg, postg, 'eligibility')
            # homeo_eli_name = conn.get_link_name(preg, postg, 'homeostatic_eligibility')
            # self.variable_to_backend(eligibility_name, backend.get_varialble(weight_name).shape, value=0.0)
            # self.variable_to_backend(homeo_eli_name, backend.get_varialble(weight_name).shape, value=0.0)
            # self.op_to_backend([trace_pre_name, trace_post_name, eligibility_name, homeo_eli_name), self.RSTDP_update,
            #                        [trace_pre_name, trace_post_name, self.get_varname('tau_pre'), self.get_varname('tau_post'),
            #                         pre_name, post_name, self.get_varname('A_plus'), self.get_varname('A_minus')]])

            if single_reward:
                reward_name = self.reward_var_names
                reward_shape = backend.get_varialble(reward_name).shape
            else:
                reward_name = self.reward_var_names[ind]
                reward_shape = backend.get_varialble(reward_name).shape
            post_shape = backend.get_varialble(post_name).shape
            assert len(reward_shape) == 2 and len(post_shape) == 2
            reward_weight_shape = [post_shape[1], reward_shape[1]]

            if reward_weight_shape == [10,10]:
                reward_weight_value = np.eye(10)
            else:
                reward_weight_value = np.clip(0.2*np.random.randn(*reward_weight_shape), self.reward_weight_min, self.reward_weight_max)
            reward_weight_name = conn.get_link_name(preg, postg, 'reward_weight')

            self.variable_to_backend(reward_weight_name, reward_weight_shape, reward_weight_value)
            self.op_to_backend([weight_name], self.RSTDP_update, [weight_name, trace_pre_name, trace_post_name, pre_name,
                                                                       post_name, self.get_varname('A_plus'), self.get_varname('A_minus'),
                                                                       reward_weight_name, reward_name])

Learner.register('bp_rstdp', Backprop_RSTDP)













class Backprop_RSTDPET(Learner):
    _learner_count = 0

    def __init__(self, trainable=None, **kwargs):
        super(Backprop_RSTDPET, self).__init__(trainable=trainable)
        self.prefered_backend = ['pytorch']
        self.name = 'Backprop-RSTDP' + str(Backprop_RSTDPET._learner_count)
        Backprop_RSTDPET._learner_count += 1
        self._tau_constant_variables = dict()
        self._constant_variables = dict()
        self._variables = dict()

        self.learning_rate = kwargs.get('lr', 0.1)
        self.homeostatic_rate = kwargs.get('hs_r', 0.01)
        self.reward_var_names = kwargs.get('reward_vars', 'Output_Reward')
        self.reward_weight_min = kwargs.get('reward_wegith_min', -10.0)
        self.reward_weight_max = kwargs.get('reward_wegith_max', 10.0)

        self._tau_constant_variables['tau_pre'] = kwargs.get('tau_pre', 4.0)
        self._tau_constant_variables['tau_post'] = kwargs.get('tau_post', 10.0)
        self._tau_constant_variables['tau_et'] = kwargs.get('tau_et', 5.0)
        self._constant_variables['A_plus'] = kwargs.get('A_plus', 1.0)
        self._constant_variables['A_minus'] = kwargs.get('A_minus', -0.5)
        self.traces = []
        self.spks = []

    def spk_trace_update(self, trace, beta, spk):
        with torch.no_grad():
            trace = beta * trace + (1 - beta) * spk
            # if list(trace.shape) == [200,400]:
            #     self.traces.append(trace)
            #     self.spks.append(spk)
        return trace

    # def RSTDP_update(self, trace_pre, trace_post, beta_pre, beta_post, spk_pre, spk_post, A_plus, A_minus):
    #     with torch.no_grad():
    #         trace_pre = beta_pre*trace_pre + spk_pre
    #         trace_post = beta_post*trace_post + spk_post
    #         if list(trace_pre.shape) == [200,784]:
    #             self.traces.append(trace_post)
    #             self.spks.append(spk_post)
    #         eligibility = torch.mean(A_plus*trace_pre.unsqueeze(1)*spk_post.unsqueeze(2) + A_minus*trace_post.unsqueeze(2)*spk_pre.unsqueeze(1), 0)
    #         homeo_eligibility = torch.mean(trace_pre.unsqueeze(1)*(trace_post**2).unsqueeze(2), 0)
    #     return trace_pre, trace_post, eligibility, homeo_eligibility

    def RSTDPET_update(self, weight, et, trace_pre, trace_post, spk_pre, spk_post, tau_et, A_plus, A_minus, reward_weight, reward):

        with torch.no_grad():
            reward = torch.matmul(reward, reward_weight.t())  # shape of post neuron (pre of reward_weight)
            reward = reward.unsqueeze(2)

            eligibility = torch.mean(
                A_plus * torch.matmul(spk_post.unsqueeze(2), trace_pre.unsqueeze(1)),0)
            et = et*tau_et + eligibility
            homeo_eligibility =  torch.matmul(trace_post.unsqueeze(2), spk_pre.unsqueeze(1))*(trace_post-0.2).unsqueeze(2)

            delta_w = torch.mean(self.learning_rate * reward * et - self.homeostatic_rate * homeo_eligibility, 0)
            weight.add_(delta_w)
        return et

    def build(self, backend: Backend):
        super(Backprop_RSTDPET, self).build(backend)
        self.dt = backend.dt

        for (key, tau_var) in self._tau_constant_variables.items():
            tau_var = np.exp(-self.dt / tau_var)
            shape = ()
            self.variable_to_backend(self.get_varname(key), shape, value=tau_var)

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
            self.variable_to_backend(self.get_varname(key), shape, value=var)

        if isinstance(self.reward_var_names, str):
            single_reward = True
        else:
            assert len(self.reward_var_names) == len(self.trainable_connections)
            single_reward = False

        for ind, conn in enumerate(self.trainable_connections.values()):
            preg = conn.pre
            postg = conn.post
            pre_name = conn.get_pre_name(preg, 'O')
            post_name = conn.get_post_name(postg, 'O')
            weight_name = conn.get_link_name(preg, postg, 'weight')


            trace_pre_name = preg.get_labeled_name('spk_trace')
            if not backend.has_variable(trace_pre_name):
                self.variable_to_backend(trace_pre_name, backend.get_varialble(pre_name).shape, value=0.0)
                self.op_to_backend([trace_pre_name], self.spk_trace_update,
                                       [trace_pre_name, self.get_varname('tau_pre'), pre_name])
            trace_post_name = postg.get_labeled_name('spk_trace')
            if not backend.has_variable(trace_post_name):
                self.variable_to_backend(trace_post_name, backend.get_varialble(post_name).shape, value=0.0)
                self.op_to_backend([trace_post_name], self.spk_trace_update,
                                       [trace_post_name, self.get_varname('tau_post'), post_name])
            # eligibility_name = conn.get_link_name(preg, postg, 'eligibility')
            # homeo_eli_name = conn.get_link_name(preg, postg, 'homeostatic_eligibility')
            eligibility_trace_name = conn.get_link_name(preg, postg, 'et')
            self.variable_to_backend(eligibility_trace_name, backend.get_varialble(weight_name).shape, value=0.0)
            # self.variable_to_backend(homeo_eli_name, backend.get_varialble(weight_name).shape, value=0.0)
            # self.op_to_backend([trace_pre_name, trace_post_name, eligibility_name, homeo_eli_name), self.RSTDP_update,
            #                        [trace_pre_name, trace_post_name, self.get_varname('tau_pre'), self.get_varname('tau_post'),
            #                         pre_name, post_name, self.get_varname('A_plus'), self.get_varname('A_minus')]])

            if single_reward:
                reward_name = self.reward_var_names
                reward_shape = backend.get_varialble(reward_name).shape
            else:
                reward_name = self.reward_var_names[ind]
                reward_shape = backend.get_varialble(reward_name).shape
            post_shape = backend.get_varialble(post_name).shape
            assert len(reward_shape) == 2 and len(post_shape) == 2
            reward_weight_shape = [post_shape[1], reward_shape[1]]

            if reward_weight_shape == [10, 10]:
                reward_weight_value = np.eye(10)
            else:
                reward_weight_value = np.clip(np.random.randn(*reward_weight_shape), self.reward_weight_min,
                                              self.reward_weight_max)
            reward_weight_name = conn.get_link_name(preg, postg, 'reward_weight')

            self.variable_to_backend(reward_weight_name, reward_weight_shape, reward_weight_value)
            self.op_to_backend(
                ([eligibility_trace_name], self.RSTDPET_update, [weight_name, eligibility_trace_name, trace_pre_name,
                                                                 trace_post_name, pre_name, post_name,
                                                                 self.get_varname('tau_et'), self.get_varname('A_plus'),
                                                      self.get_varname('A_minus'), reward_weight_name, reward_name]))


Learner.register('bp_rstdpet', Backprop_RSTDP)



