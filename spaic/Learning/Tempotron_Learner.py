from .Learner import Learner
import torch
import numpy as np

class Tempotron(Learner):
    '''
        Tempotron learning rule.
        Args:
            tau(num) : The parameter tau of tempotron learning model.
            tau_s(num) : The parameter tau_s of tempotron learning model.
            V0(num) : The parameter V0 of tempotron learning model.
            dw_i(num) : The parameter dw_i of tempotron learning model.
            time_window(num) : The parameter time_window of tempotron learning model.
            preferred_backend(list) : The backend prefer to use, should be a list.
            name(str) : The name of this learning model. Should be 'Tempotron'.
        Methods:
            initial_param(self, weight): initialize some parameters for each batch.
            tempotron_update(self, input, output, weight, output_V): calculate the update of weight
            build(self, simulator): Build the simulator, realize the algorithm of Tempotron learning model.
        Example:
            self._learner = BaseLearner(algorithm='Tempotron', lr=0.5, trainable=self, conn=self.connection1)
        Reference:
            The tempotron: a neuron that learns spike timing_based decisions.
            doi: 10.1038/nn1643.
            url: http:www.nature.com/natureneuroscience
    '''

    def __init__(self, trainable=None, lr=0.01, **kwargs):
        super(Tempotron, self).__init__(trainable=trainable)

        self.prefered_backend = ['pytorch']
        self.firing_func = None
        self.lr = lr
        self.tau = 20
        self.tau_s = 5
        self.required_grad = kwargs.get('required_grad')
        self.trainable = trainable

        self.fake = 1
        self.v_th = 10
        self.V_max = 0

    def initial_param(self, weight):
        '''
            Initialize some parameters for each batch.
            Args:
                weight: the weight for the connection which needed to learned.
        '''
        self.total_V = []
        self.run_time = self._backend.runtime
        self.total_Input = []
        self.total_Output = []
        self.total_step = self.run_time / self.dt - 1
        self.dw = torch.zeros(weight.shape)
        dw_i = [np.exp(-t / self.tau) - np.exp(-t / self.tau_s) for t in range(int(self.run_time / self.dt))]
        # dw_i = dw_i[:: -1]  # 将dw翻转一下
        # self.V0 = 1 / (max(dw_i))
        self.V0 = 6.75 # V0
        # self.dw_i = torch.mul(self.V0.to(self.device), torch.tensor(dw_i).to(self.device))
        self.dw_i = torch.tensor(dw_i).to(self.device)

        return self.dw_i

    def tempotron_update(self, input, output, weight, output_V):
        '''
            Args:
                input: input spikes.
                output: output spikes.
                weight: the weight for the connection which needed to learned.
                output_V: the voltage of postsynaptic neurons
        Returns:
             Updated weight.
        '''

        if self.fake:
            self.fake -= 1
            return weight
        input = torch.unsqueeze(input, 0)
        output = torch.unsqueeze(output, 0)
        output_V = torch.unsqueeze(output_V, 0)
        output_spikes = output.data
        input_spikes = input.data
        # output_num = self.trainable.output.num
        input_num = self.trainable[0].pre.num

        if self.total_V == []:
            self.total_V = output_V
        else:
            self.total_V = torch.cat((self.total_V, output_V), dim=0)

        if self.total_Input == []:
            self.total_Input = input_spikes
        else:
            self.total_Input = torch.cat((self.total_Input, input_spikes), dim=0)
        if self.total_Output == []:
            self.total_Output = output_spikes
        else:
            self.total_Output = torch.cat((self.total_Output, output_spikes), dim=0)

        if self._backend.n_time_step < self.total_step:
            pass

        else:

            label = self.trainable[1].source
            b = torch.zeros(weight.shape)

            # output_label = (torch.argmax(torch.sum(self.total_Output, dim=0), dim=1)) #先将所有时间步的output求和，然后再根据output那一列找到最大的output索引
            # self.total_output维度(总时间步,batch_size, output_num)

            update = []

            for i in range(output_V.shape[1]):  # 把每个batch分开算
                Input_each = self.total_Input[:, i, :]
                output_each = self.total_Output[:, i, :]
                V_each = self.total_V[:, i, :]
                # import matplotlib.pyplot as plt
                # plt.plot(V_each[:, 0].cpu().detach().numpy())
                #
                # plt.show()

                # 判断output的tmax，看v的最大值和output发放的时间点谁更早
                # t_max_all = torch.argmax(output_each, dim=0) if 1 in output else torch.argmax(V_each, dim=0)
                t_max_all = torch.argmax(V_each, dim=0)  # 十个output神经元的最大V的index

                # 最大值的V有没有超过阈值，如果超过阈值，那么不动，如果没超过阈值，那么V要增加
                # 除了最大值的V其他的V都不应该超过阈值，如果有超过阈值的，需要减小V
                # t_o = torch.where(output_each)

                for o in range(t_max_all.shape[0]):

                    V = V_each[t_max_all[o], o]
                    # print(V)
                    if o == label[i] and V < self.v_th :

                        b[o, :] = 1

                    elif o != label[i] and V >= self.v_th :

                        b[o, :] = -1
                # Input_dw = torch.zeros_like(weight)
                for o in range(t_max_all.shape[0]):
                    a_1 = torch.linspace(0, t_max_all[o] - 1, steps=t_max_all[o]).to(
                        self.device).type(torch.long)
                    Input_window = torch.index_select(Input_each, 0, a_1).permute(1, 0)  # (784, t_max_all[i][j])
                    dw = torch.flip(self.dw_i[:t_max_all[o]], dims=[0])
                    # dw = self.dw_i[:t_max_all[o]][::-1]
                    Input_dw_each = torch.sum((Input_window * dw), dim=1)
                    self.dw[o] = Input_dw_each
                    # if t_max_all[o] - self.time_window + 1 > 0:
                    #     a = torch.linspace(t_max_all[o] - self.time_window + 1, t_max_all[o],
                    #                        steps=self.time_window).to(
                    #         self.device).type(torch.long)
                    #     Input_window = torch.index_select(Input_each, 0, a).permute(1, 0)  # (784, self.t)
                    # else:
                    #     a_1 = torch.linspace(0, t_max_all[o], steps=t_max_all[o]+1).to(
                    #         self.device).type(torch.long)
                    #     Input_window = torch.index_select(Input_each, 0, a_1).permute(1, 0)  # (784, t_max_all[i][j])
                    #     dw = torch.flip(self.dw_i[:t_max_all[o] +1], dims=[0])
                    #     # dw = self.dw_i[:t_max_all[o]][::-1]
                    #     Input_dw_each = torch.sum((Input_window * dw), dim=1)
                    #     self.dw[o] = Input_dw_each
                    #     # pad_num = self.time_window - t_max_all[o] - 1
                    #     # pad = torch.zeros((input_num, pad_num)).to(self.device)
                    #     # Input_window = torch.cat((Input_window, pad), 1)

                    # c = Input_window * self.dw_i.to(self.device)
                    # Input_dw = torch.sum(c, 1)
                    # Input_dw = torch.sum(Input_window * self.dw_i.to(self.device), 1)#(把每一个神经元时间窗里的值都加起来)
                    # self.dw[o, :] = Input_dw

                self.dw = self.lr * b * self.dw

                if update == []:
                    update = torch.unsqueeze(self.dw, 0)
                else:
                    update = torch.cat((update, torch.unsqueeze(self.dw, 0)), dim=0)
            update = torch.mean(update, dim=0)

            if self.required_grad == False:
                with torch.no_grad():
                    # print('update1', update)
                    weight.add_(update.to(self.device))
                    weight.clamp_(0.0, 2.0)
            else:
                # print('update2', update)
                weight[...] = weight + update.to(self.device)  # 注意tensor会出现假赋值的情况，就是他没有赋值给原来的tensor，而是自己新建了一个tensor
            # self.V_max = torch.max(self.total_V, dim=0).values
            # self._simulator._variables['autoname1<net>_layer1<neg>:{V_max[stay]}'] = self.V_max
            # print([self.V_max])
            # print('a')

        return weight

    def build(self, backend):
        '''
            Build the simulator, realize the algorithm of Tempotron model.
            Args:
                simulator: The simulator we used to compute.
        '''

        super(Tempotron, self).build(backend)
        self.device = backend.device0
        self._backend = backend
        self.sim_name = backend.backend_name
        self.dt = backend.dt

        if backend.backend_name == 'pytorch':
            import torch

            # 为online_stdp函数的input, output, weight 等参数命名（取名函数在Connection中）
            for conn in self.trainable_connections.values():
                preg = conn.pre
                postg = conn.post
                # pre_name = conn.get_input_name(preg, postg)
                # post_name = conn.get_target_output_name(postg)
                pre_name = conn.get_input_name(preg, postg)
                post_name = conn.get_group_name(postg, 'O')
                weight_name = conn.get_link_name(preg, postg, 'weight')

                V_name = conn.get_group_name(postg, 'V[updated]')
                # V0 = conn.get_group_name(postg, 'V0')
                # V_max = conn.get_group_name(postg, 'V_max')
                # V_name = conn.get_V_updated_name(postg)
                # V0 = conn.post_assembly.id + ':' + '{V0}'
                # V_max = conn.post_assembly.id + ':' + '{V_max}'

                # register_standalone(self,output_name: str, function, input_names(后端输入的名字): list):
                self.init_op_to_backend(None, self.initial_param, [weight_name])
                # simulator.add_variable(V_max, [], self.V_max)
                self.op_to_backend(weight_name, self.tempotron_update,
                                              [pre_name, post_name, weight_name, V_name])


Learner.register("Tempotron", Tempotron)