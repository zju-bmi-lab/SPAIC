from .Learner import Learner
import torch
import numpy as np
class nearest_online_STDP(Learner):
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
            preg = conn.pre_assembly
            postg = conn.post_assembly
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

            backend.add_operation(['input_trace_s', 'var_mult', input_trace_name, 'trace_decay'])
            backend.add_operation(['input_temp', 'minus', 'spike', pre_name])
            backend.add_operation([input_trace_name, 'var_linear', 'input_temp', 'input_trace_s', pre_name])

            backend.add_operation(['output_trace_s', 'var_mult', output_trace_name, 'trace_decay'])
            backend.add_operation(['output_temp', 'minus', 'spike', post_name])
            backend.add_operation([output_trace_name, 'var_linear', 'output_temp', 'output_trace_s', post_name])

            backend.add_operation(['pre_post_temp', 'mat_mult_pre', post_name, input_trace_name+'[updated]'])
            backend.add_operation(['pre_post', 'var_mult', 'Apost', 'pre_post_temp'])
            backend.add_operation(['post_pre_temp', 'mat_mult_pre', output_trace_name+'[updated]', pre_name])
            backend.add_operation(['post_pre', 'var_mult', 'Apre', 'post_pre_temp'])
            backend.add_operation([dw_name, 'minus', 'pre_post', 'post_pre'])
            backend.add_operation([weight_name, self.nearest_online_stdp_weightupdate, dw_name, weight_name])
Learner.register("nearest_online_stdp", nearest_online_STDP)



class full_online_STDP(Learner):
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
            preg = conn.pre_assembly
            postg = conn.post_assembly
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

            backend.add_operation(['input_trace_temp', 'var_mult', input_trace_name, 'trace_decay'])
            backend.add_operation([input_trace_name, 'add', pre_name, 'input_trace_temp'])


            backend.add_operation(['output_trace_temp', 'var_mult', output_trace_name, 'trace_decay'])
            backend.add_operation([output_trace_name, 'add', post_name, 'output_trace_temp'])

            backend.add_operation(['pre_post_temp', 'mat_mult_pre', post_name, input_trace_name+'[updated]'])
            backend.add_operation(['pre_post', 'var_mult', 'Apost', 'pre_post_temp'])
            backend.add_operation(['post_pre_temp', 'mat_mult_pre', output_trace_name+'[updated]', pre_name])
            backend.add_operation(['post_pre', 'var_mult', 'Apre', 'post_pre_temp'])
            backend.add_operation([dw_name, 'minus', 'pre_post', 'post_pre'])
            backend.add_operation([weight_name, self.full_online_stdp_weightupdate, dw_name, weight_name])



Learner.register("full_online_STDP", full_online_STDP)


