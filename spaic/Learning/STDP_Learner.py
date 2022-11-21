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
