from .Learner import Learner
import torch
import numpy as np

class Base_STDP(Learner):
    def __init__(self, trainable=None, *args, **kwargs):
        super(Base_STDP, self).__init__(trainable, *args, **kwargs)
        self.gradient_based = False
    
    def build(self, backend):
        super(Base_STDP, self).build(backend)


class nearest_online_STDP(Base_STDP):
    '''
        nearest_online STDP learning rule.

        Args:
            Apost(num) : The parameter Apost of nearest_online STDP learning model.
            Apre(num) : The parameter Apre of nearest_online STDP learning model.
            trace_decay(num) : The parameter trace_decay of nearest_online STDP learning model.
            preferred_backend(list) : The backend prefer to use, should be a list.
            name(str) : The name of this learning model. Should be 'nearest_online_STDP'.

        Methods:
            initial_param(self, input, output): initialize the output_trace and the input_trace for each batch.
            nearest_online_stdp_weightupdate(self, input, output, weight): calculate the update of weight
            build(self, backend): Build the backend, realize the algorithm of nearest_online STDP learning model.

        Example:
            self._learner = BaseLearner(algorithm='nearest_online_STDP', lr=0.5, trainable=self, conn=self.connection1)

        Reference:
            Unsupervised learning of digit recognition using spike-timing-dependent plasticity.
            doi: 10.3389/fncom.2015.00099.
            url: http://journal.frontiersin.org/article/10.3389/fncom.2015.00099/abstract
    '''

    def __init__(self, trainable=None, *args, **kwargs):
        super(nearest_online_STDP, self).__init__(trainable=trainable)

        self.prefered_backend = ['pytorch']
        self.name = 'nearest_online_STDP'
        self._constant_variables = dict()
        self._constant_variables['Apost'] = kwargs.get('Apost', 1e-2)
        self._constant_variables['Apre'] = kwargs.get('Apre', 1e-4)
        self._constant_variables['trace_decay'] = kwargs.get('trace_decay', np.exp(-1/20))
        self._constant_variables['spike'] = kwargs.get('spike', 1)
        self.lr = kwargs.get('lr', 0.01)
        self.w_norm = 78.4
        self.trainable = trainable

    def nearest_online_stdp_weightupdate(self, dw, weight):
        '''

            Args:
                dw: the change of weight
                weight: weight between pre and post neurongroup

            Returns:
                Updated weight.

        '''
        with torch.no_grad():
            weight.add_(dw)

            # if self._backend.n_time_step < self.total_step:
            #     pass
            # else:
                # for i in range(weight.shape[0]):
                #     weight[i] = (weight[i] * self.w_norm) / torch.sum(torch.abs(weight[i]))

            weight[...] = (self.w_norm * torch.div(weight, torch.sum(torch.abs(weight), 1, keepdim=True)))


            weight.clamp_(0.0, 1.0)
            return weight


    def build(self, backend):

        super(nearest_online_STDP, self).build(backend)

        self._backend = backend
        self.dt = backend.dt
        self.run_time = backend.runtime
        self.total_step = self.run_time / self.dt - 1

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

        for conn in self.trainable_connections.values():
            preg = conn.pre
            postg = conn.post
            # pre_name = conn.get_input_name(preg, postg)
            # post_name = conn.get_target_output_name(postg)
            # weight_name = conn.get_weight_name(preg, postg)
            pre_name = conn.get_input_name(preg, postg)
            post_name = conn.get_group_name(postg, 'O')
            weight_name = conn.get_link_name(preg, postg, 'weight')

            input_trace_name = pre_name + '_{input_trace}'
            output_trace_name = post_name + '_{output_trace}'
            dw_name = weight_name + '_{dw}'
            self.variable_to_backend(input_trace_name, backend._variables[pre_name].shape, value=0.0)
            self.variable_to_backend(output_trace_name, backend._variables[post_name].shape, value=0.0)
            self.variable_to_backend(dw_name, backend._variables[weight_name].shape, value=0.0)
            # input_trace_s = (self.input_trace * self.trace_decay)
            # input_trace_s = input + (1 - input) * input_trace_s
            # self.input_trace = input_trace_s
            #
            # output_trace_s = (self.output_trace * self.trace_decay)
            # output_trace_s = output + (1 - output) * output_trace_s
            # self.output_trace = output_trace_s
            #
            ##calculate dw, dw = Apost * (output_spike * input_trace) – Apre * (output_trace * input_spike)
            #
            # dw = self.Apost * torch.matmul(output_spike.permute(1, 0), input_trace_s) \
            #              - self.Apre * torch.matmul(output_trace_s.permute(1, 0), input_spike)  #

            self.op_to_backend('input_trace_s', 'var_mult', [input_trace_name, 'trace_decay'])
            self.op_to_backend('input_temp', 'minus', ['spike', pre_name])
            self.op_to_backend(input_trace_name, 'var_linear', ['input_temp', 'input_trace_s', pre_name])

            self.op_to_backend('output_trace_s', 'var_mult', [output_trace_name, 'trace_decay'])
            self.op_to_backend('output_temp', 'minus', ['spike', post_name])
            self.op_to_backend(output_trace_name, 'var_linear', ['output_temp', 'output_trace_s', post_name])

            self.op_to_backend('pre_post_temp', 'mat_mult_pre', [post_name, input_trace_name+'[updated]'])
            self.op_to_backend('pre_post', 'var_mult', ['Apost', 'pre_post_temp'])
            self.op_to_backend('post_pre_temp', 'mat_mult_pre', [output_trace_name+'[updated]', pre_name])
            self.op_to_backend('post_pre', 'var_mult', ['Apre', 'post_pre_temp'])
            self.op_to_backend(dw_name, 'minus', ['pre_post', 'post_pre'])
            self.op_to_backend(weight_name, self.nearest_online_stdp_weightupdate, [dw_name, weight_name])
Learner.register("nearest_online_stdp", nearest_online_STDP)



class full_online_STDP(Base_STDP):
    '''
        nearest_online STDP learning rule.

        Args:
            Apost(num) : The parameter Apost of full_online STDP learning model.
            Apre(num) : The parameter Apre of full_online STDP learning model.
            trace_decay(num) : The parameter trace_decay of full_online STDP learning model.
            preferred_backend(list) : The backend prefer to use, should be a list.
            name(str) : The name of this learning model. Should be 'full_online_STDP'.

        Methods:
            initial_param(self, input, output): initialize the output_trace and the input_trace for each batch.
            nearest_online_stdp_weightupdate(self, input, output, weight): calculate the update of weight
            build(self, backend): Build the backend, realize the algorithm of full_online STDP learning model.

        Example:
            self._learner = BaseLearner(algorithm='full_online_STDP', lr=0.5, trainable=self, conn=self.connection1)

        Reference:
            Unsupervised learning of digit recognition using spike-timing-dependent plasticity.
            doi: 10.3389/fncom.2015.00099.
            url: http://journal.frontiersin.org/article/10.3389/fncom.2015.00099/abstract
        '''

    def __init__(self, trainable=None, *args, **kwargs):
        super(full_online_STDP, self).__init__(trainable=trainable)
        self.prefered_backend = ['pytorch']
        self.firing_func = None
        self._constant_variables = dict()
        self._constant_variables['Apost'] = kwargs.get('Apost', 0.01)
        self._constant_variables['Apre'] = kwargs.get('Apre', 1e-4)
        self._constant_variables['trace_decay'] = kwargs.get('trace_decay', np.exp(-1 / 20))
        self.lr = kwargs.get('lr', 0.01)
        self.name = 'full_online_STDP'
        self.w_norm = 78.4

    def full_online_stdp_weightupdate(self, dw, weight):
        '''

            Args:
                dw: the change of weight
                weight: weight between pre and post neurongroup

            Returns:
                Updated weight.

        '''
        with torch.no_grad():
            weight.add_(dw)

            if self._backend.n_time_step < self.total_step:
                pass
            else:
                weight[...] = (self.w_norm * torch.div(weight, torch.sum(torch.abs(weight), 1, keepdim=True)))

            weight.clamp_(0.0, 1.0)
            return weight


    def build(self, backend):
        super(full_online_STDP, self).build(backend)
        self._backend = backend
        self.dt = backend.dt
        self.run_time = backend.runtime
        self.total_step = self.run_time / self.dt - 1

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

        for conn in self.trainable_connections.values():
            preg = conn.pre
            postg = conn.post
            pre_name = conn.get_input_name(preg, postg)
            post_name = conn.get_group_name(postg, 'O')
            weight_name = conn.get_link_name(preg, postg, 'weight')

            input_trace_name = pre_name + '_{input_trace}'
            output_trace_name = post_name + '_{output_trace}'
            dw_name = weight_name + '_{dw}'
            self.variable_to_backend(input_trace_name, backend._variables[pre_name].shape, value=0.0)
            self.variable_to_backend(output_trace_name, backend._variables[post_name].shape, value=0.0)
            self.variable_to_backend(dw_name, backend._variables[weight_name].shape, value=0.0)

            # input_trace_s = (self.input_trace * self.trace_decay) + input
            # self.input_trace = input_trace_s
            #
            # output_trace_s = (self.output_trace * self.trace_decay) + output
            # self.output_trace = output_trace_s
            #
            #calculate dw, dw = Apost * (output_spike * input_trace) – Apre * (output_trace * input_spike)
            #
            # dw = self.Apost * torch.matmul(output_spike.permute(1, 0), input_trace_s) \
            #              - self.Apre * torch.matmul(output_trace_s.permute(1, 0), input_spike)  #

            self.op_to_backend('input_trace_temp', 'var_mult', [input_trace_name, 'trace_decay'])
            self.op_to_backend(input_trace_name, 'add', [pre_name, 'input_trace_temp'])

            self.op_to_backend('output_trace_temp', 'var_mult', [output_trace_name, 'trace_decay'])
            self.op_to_backend(output_trace_name, 'add', [post_name, 'output_trace_temp'])

            self.op_to_backend('pre_post_temp', 'mat_mult_pre', [post_name, input_trace_name+'[updated]'])
            self.op_to_backend('pre_post', 'var_mult', ['Apost', 'pre_post_temp'])
            self.op_to_backend('post_pre_temp', 'mat_mult_pre', [output_trace_name+'[updated]', pre_name])
            self.op_to_backend('post_pre', 'var_mult', ['Apre', 'post_pre_temp'])
            self.op_to_backend(dw_name, 'minus', ['pre_post', 'post_pre'])
            self.op_to_backend(weight_name, self.full_online_stdp_weightupdate,[dw_name, weight_name])



Learner.register("full_online_STDP", full_online_STDP)


class Meta_nearest_online_STDP(Base_STDP):
    def __init__(self, trainable=None, *args, **kwargs):
        super(Meta_nearest_online_STDP, self).__init__(trainable=trainable, **kwargs)
        self.trainable = trainable
        self.prefered_backend = ['pytorch']
        self.name = 'meta_nearest_online_STDP'
        self._constant_variables = dict()
        self._constant_variables['Apost'] = kwargs.get('Apost', 1e-2)
        self._constant_variables['Apre'] = kwargs.get('Apre', 1e-4)
        self._constant_variables['pre_decay'] = kwargs.get('pre_decay', np.exp(-1/30))
        self._constant_variables['post_decay'] = kwargs.get('post_decay', np.exp(-1 / 200))
        self.w_min = kwargs.get('w_min', 0.0)
        self.w_max = kwargs.get('w_max', 1.0)
        self.w_norm = 1.2
        self.w_mean = None
        self.lr = kwargs.get('lr', 0.01)
        self.param_run_update = True

    def update(self, input, output, input_trace, output_trace, pre_decay, post_decay, Apost, Apre, weight):
        if self.w_mean is None:
            self.w_mean = torch.mean(weight, dim=1, keepdim=True).detach()
            self.aw_mean = torch.mean(self.w_mean)
        if self.training:
            input_trace = pre_decay*input_trace*input.le(0) + input
            output_trace = post_decay*output_trace*output.le(0) + output
            pre_post = torch.matmul(output.permute(1, 0), input_trace)
            post_pre = torch.matmul(output_trace.permute(1,0), input)
            dw = Apost*pre_post - Apre*post_pre
            with torch.no_grad():
                self.w_mean = self.w_mean - 0.1*Apre * (
                            torch.mean(output_trace, 0) - torch.mean(output_trace)).unsqueeze(1)
                self.w_mean = self.w_mean*self.aw_mean/torch.mean(self.w_mean)
            with torch.no_grad():
                self.w_mean = self.w_mean - 0.1 * Apre * (
                        torch.mean(output_trace, 0) - torch.mean(output_trace)).unsqueeze(1)
                self.w_mean = self.w_mean * self.aw_mean / torch.mean(self.w_mean)

            soft_clamp = (dw.lt(0)*torch.exp(-0.2/torch.clamp_min(weight-self.w_min, 1.0e-4))
                          + dw.gt(0)*torch.exp(-0.2/torch.clamp_min(self.w_max-weight, 1.0e-4))).detach()
            weight = weight*self.w_mean/(torch.clamp_min(torch.mean(weight, dim=1, keepdim=True), 1e-6)).detach() + dw*soft_clamp
        # mw = torch.mean(weight, dim=1)
        # print(torch.amax(mw), torch.amin(mw))
        return input_trace, output_trace, weight

    def build(self, backend):

        super(Meta_nearest_online_STDP, self).build(backend)

        self._backend = backend
        self.dt = backend.dt
        self.run_time = backend.runtime

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

        for conn in self.trainable_connections.values():
            preg = conn.pre
            postg = conn.post
            pre_name = conn.get_input_name(preg, postg)
            post_name = conn.get_group_name(postg, 'O')
            weight_name = conn.get_link_name(preg, postg, 'weight')

            input_trace_name = pre_name + '_{input_trace}'
            output_trace_name = post_name + '_{output_trace[stay]}'
            self.variable_to_backend(input_trace_name, backend._variables[pre_name].shape, value=0.0)
            self.variable_to_backend(output_trace_name, backend._variables[post_name].shape, value=0.0)

            self.op_to_backend([input_trace_name, output_trace_name, weight_name], self.update,
                                   [pre_name, post_name, input_trace_name, output_trace_name, 'pre_decay', 'post_decay', 'Apost', 'Apre', weight_name])

Learner.register("meta_nearest_online_stdp", Meta_nearest_online_STDP)


class PostPreInt(Base_STDP):
    # language=rst
    """
    Simple STDP rule involving both pre- and post-synaptic spiking activity, based on integer
    arithmetic. By default, pre-synaptic update is negative and the post-synaptic update is
    positive.
    """
    def __init__(self, trainable=None, *args, **kwargs):
        super(PostPreInt, self).__init__(trainable=trainable)
        self.prefered_backend = ['pytorch']
        self.name = 'postpreint'
        self._constant_variables = dict()
        self._constant_variables['trace_decay'] = kwargs.get('trace_decay', 31169)
        self._constant_variables['shift_spike_trace'] = kwargs.get('shift_spike_trace', 15)
        self._constant_variables['trace_scale'] = kwargs.get('trace_scale', 127)
        self._constant_variables['max_threshold'] = kwargs.get('max_threshold', 4915)

        self.shift = kwargs.get('shift', 14)
        self.nu0 = kwargs.get('nu0', 14)
        self.nu1 = kwargs.get('nu1', 141)
        self.wmin = kwargs.get('w_min', 0)
        self.wmax = kwargs.get('w_max', 109)


    def update(self, input, output, input_trace, output_trace, trace_decay, trace_scale, shift_spike_trace, max_threshold, post_thresh, weight):

        def stochastic_round(x):
            p = x & ((1 << self.shift) - 1)
            return (x >> self.shift) \
                   + (torch.randint(0, 1 << self.shift, p.size()) < p).int()

        input_trace = (trace_decay*input_trace.int()) >> shift_spike_trace
        output_trace = (trace_decay*output_trace.int()) >> shift_spike_trace
        input_trace.masked_fill_(input.bool(), trace_scale if shift_spike_trace > 0 else 1)
        output_trace.masked_fill_(output.bool(), trace_scale if shift_spike_trace > 0 else 1)

        # Pre-synaptic update.
        source_s = input.unsqueeze(1).int()  # spike  1, 1, 784
        target_x = output_trace.unsqueeze(2).int() * self.nu0  # spike_trace  1, 100, 1
        target_x = target_x.repeat((1, 1, source_s.size(2)))
        target_x = stochastic_round(target_x)
        weight += torch.squeeze(torch.mul(target_x, source_s), dim=0)
        del source_s, target_x
        # p = target_x & ((1 << self.shift) - 1)
        # target_x = (target_x >> self.shift) + (torch.randint(0, 1 << self.shift, p.size()) < p).int()
        # weight -= torch.squeeze(torch.bmm(target_x, source_s), dim=0)
        # del source_s, target_x, p

        # Post-synaptic update.
        target_s = output.unsqueeze(2).int()   # 1,100,1
        source_x = input_trace.unsqueeze(1).int() * self.nu1  # 1,1,784
        source_x = source_x.repeat((1, target_s.size(1), 1))
        source_x = stochastic_round(source_x)
        weight += torch.squeeze(torch.mul(target_s, source_x), dim=0)
        del source_x, target_s
        # p = source_x & ((1 << self.shift) - 1)
        # source_x = (source_x >> self.shift) + (torch.randint(0, 1 << self.shift, p.size()) < p).int()
        # weight += torch.squeeze(torch.bmm(target_s, source_x), dim=0)
        # del source_x, target_s, p

        weight >>= (post_thresh.view(-1) > max_threshold).unsqueeze(1)

        if self.wmin != -np.inf or self.wmax != np.inf:
            weight.clamp_(self.wmin, self.wmax)
        return input_trace, output_trace, weight

    def build(self, backend):
        super(PostPreInt, self).build(backend)
        self.dt = backend.dt

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


        # Traverse all trainable connections
        for conn in self.trainable_connections.values():
            preg = conn.pre
            postg = conn.post
            pre_name = conn.get_input_name(preg, postg)
            post_name = conn.get_group_name(postg, 'O')
            post_thresh_name = conn.get_group_name(postg, 'thresh[updated]')
            weight_name = conn.get_link_name(preg, postg, 'weight')

            # input_trace tracks the trace of presynaptic spikes; output_trace tracks the trace of postsynaptic spikes
            input_trace_name = pre_name + '_{input_trace}'
            output_trace_name = post_name + '_{output_trace}'
            eligibility_name = weight_name + '_{eligibility}'

            self.variable_to_backend(input_trace_name, backend._variables[pre_name].shape, value=0.0)
            self.variable_to_backend(output_trace_name, backend._variables[post_name].shape, value=0.0)
            self.variable_to_backend(eligibility_name, backend._variables[weight_name].shape, value=0.0)

            pre_name_updated = conn.get_group_name(preg, 'O[updated]')
            post_name_updated = conn.get_group_name(postg, 'O[updated]')

            self.op_to_backend([input_trace_name, output_trace_name, weight_name], self.update,
                               [pre_name_updated, post_name_updated, input_trace_name,
                                output_trace_name, 'trace_decay', 'trace_scale',
                                'shift_spike_trace', 'max_threshold', post_thresh_name, weight_name])

Learner.register('postpreint', PostPreInt)