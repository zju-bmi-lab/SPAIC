import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import spaic

from matplotlib.pyplot import *
global_list = list()
global_tv_dict = dict()
global_dv_dict = dict()
global_te_dict = dict()
global_de_dict = dict()

MIN_DV = 0.001
MAX_DV = 100.0

# global_v_dict[name]

class NeuronFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs_w: torch.Tensor, inputs_t: torch.Tensor, history_output: torch.Tensor,
                template_tv: torch.Tensor, template_dv: torch.Tensor, template_te: torch.Tensor, template_de: torch.Tensor,
                 threshold: torch.Tensor):
        # input shape : (batch_size, neuron_num, time_step)
        old_shape = list(inputs_t.shape)
        inputs = inputs_t.view(old_shape[0], -1, old_shape[-1])
        tmp_shape = inputs.shape

        temp_tv =  template_tv[:, :, :tmp_shape[2]]
        temp_dv = -template_dv[:, :, :tmp_shape[2]]
        temp_te =  template_te[:, :, :tmp_shape[2]]
        temp_de = -template_de[:, :, :tmp_shape[2]]
        current_tv = torch.matmul(inputs, temp_tv.unsqueeze(-1)) + torch.matmul(history_output, temp_te.unsqueeze(-1))
        current_dv = torch.matmul(inputs, temp_dv.unsqueeze(-1)) + torch.matmul(history_output, temp_de.unsqueeze(-1))


        current_spike = current_tv.gt(threshold).float()

        # for probabilistic spike
        # current_spike = torch.zeros_like(current_tv)
        # current_spike_mask = current_tv.gt(threshold)
        # current_spike.mask_fill_(torch.bitwise_not(current_spike_mask), F.relu(current_tv/threshold)/5.0)
        # current_spike.mask_fill_(current_spike_mask, 0.5*F.sigmoid(current_dv)+0.5)

        ctx.save_for_backward(inputs_w, history_output, current_spike, current_dv, temp_tv, temp_dv, temp_de)
        old_shape.append(1)

        return current_spike.view(old_shape)

    @staticmethod
    def backward(ctx, grad_t):

        inputs, history_output, current_spike, current_dv, partial_dv, partial_tv, partial_de = ctx.saved_tensors
        current_dv = torch.clamp(current_dv, MIN_DV, MAX_DV)

        old_shape = inputs.shape
        inputs = inputs.view(old_shape[0], -1, old_shape[-1])
        input_spike = torch.nonzero(inputs).float()

        partial_input_t = inputs         * partial_dv/current_dv
        partial_input_w = input_spike    * partial_tv/current_dv
        partial_history = history_output * partial_de/current_dv
        partial_input_t = partial_input_t.view(old_shape)*grad_t
        partial_input_w = partial_input_w.view(old_shape)*grad_t
        partial_history = partial_history.view(old_shape)*grad_t

        if ctx.needs_input_grad[3]:
            partial_tv = grad_t.view()*inputs / current_dv
            partial_te = history_output / current_dv
        else:
            partial_tv = partial_te = None




        return partial_input_w, partial_input_t, partial_history, partial_tv, None, partial_te, None, None

# func = UnitFunc.apply

class SpikePropUnit(nn.Module):

    def __init__(self, shape, name=None, threshold=1.0, n_step=1000, tau_m=10.0, tau_s=6.0,  dt=0.1, vt_curve=None, reset_curve=None, dv_curve=None, dreset_curve=None, istrain_neurparam=0):
        '''
        Initialize a Neruongroup with SpikeProp Learning rule.
        :param shape: shape of the neurongroup
        :param name: name of the neurongroup
        :param threshold:
        :param n_step:
        :param tau_m:
        :param tau_s:
        :param dt:
        :param vt_curve: user defined post-spike response potential (PSP)
        :param reset_curve:
        :param dv_curve:
        :param dreset_curve:
        :param train_neurparam: 0 -> don't train neural parameters, 1 -> train and use identical pararmeters for neuron group, 2 -> train and use different pararmeters for each neuron in the group
        NeuroFunc(inputs_w, inputs_t, history_output, template_tv, template_dv, template_te, template_de, threshold)
        '''

        super(SpikePropUnit, self).__init__()
        self.name = name
        self.shape = shape
        self.num = np.prod(shape, dtype=int)
        self.dt = dt
        self.n_step = n_step # max time step of grad backprop
        self.threshold = torch.tensor(threshold, dtype=torch.float)
        self.UserDefSRM = False
        self.op = NeuronFunc.apply
        self.istrain_neurparam = istrain_neurparam
        self.tau_m = tau_m
        self.tau_s = tau_s

        self.inputs_w_list = []
        self.inputs_t_list = []
        self.history_output_list = []

        if (vt_curve is not None) and (reset_curve is not None):
            # build user defined SRM model with input curve
            self.UserDefSRM = True
            if len(vt_curve) != n_step:
                raise ValueError("the lenght of user defined vt_curve is not accordant with n_step")
            if len(reset_curve) != n_step:
                raise ValueError("the lenght of user defined reset_curve is not accordant with n_step")
            tv = torch.tensor(vt_curve, dtype=torch.float)
            te = torch.tensor(reset_curve, dtype=torch.float)
            if dv_curve is None:
                dv = torch.zeros_like(tv)
                dv[1:] = (tv[1:] - tv[:-2])/dt
            elif len(dv_curve) != n_step:
                raise ValueError("the lenght of user defined dv_curve is not accordant with n_step")
            else:
                dv = torch.tensor(dv_curve, dtype=torch.float)
            if dreset_curve is None:
                de = torch.zeros_like(te)
                de[1:] = (te[1:]-te[:-2])/dt
            elif len(dreset_curve) != n_step:
                raise ValueError("the lenght of user defined dreset_curve is not accordant with n_step")
            else:
                de = torch.tensor(dreset_curve, dtype=torch.float)

            if istrain_neurparam == 0:
                self.tv = tv.view(1, 1, -1)
                self.dv = dv.view(1, 1, -1)
                self.te = te.view(1, 1, -1)
                self.de = de.view(1, 1, -1)
            elif istrain_neurparam == 1:
                self.tv = nn.Parameter(tv.view(1, 1, -1))
                self.dv = nn.Parameter(dv.view(1, 1, -1))
                self.te = nn.Parameter(te.view(1, 1, -1))
                self.de = nn.Parameter(de.view(1, 1, -1))
            elif istrain_neurparam == 2:
                self.tv = nn.Parameter(tv.view(1, 1, -1).repeat(1, self.num, 1))
                self.dv = nn.Parameter(dv.view(1, 1, -1).repeat(1, self.num, 1))
                self.te = nn.Parameter(te.view(1, 1, -1).repeat(1, self.num, 1))
                self.de = nn.Parameter(de.view(1, 1, -1).repeat(1, self.num, 1))
        else:
            # use default SRM model
            tt = torch.arange(1, n_step + 1, dtype=torch.float) * dt
            tt = tt.view(1, 1, -1)
            if istrain_neurparam == 0:
                self.tv = (torch.exp(-tt / self.tau_m) - torch.exp(-tt / self.tau_s)) / (self.tau_m - self.tau_s)
                self.dv = (torch.exp(-tt / self.tau_s) / self.tau_s - torch.exp(-tt / self.tau_m) / self.tau_m) / (self.tau_m - self.tau_s)
                self.te = -torch.exp(-tt / self.tau_m)
                self.de = torch.exp(-tt / self.tau_m) / self.tau_m
            elif istrain_neurparam == 1:
                self.tau_m = nn.Parameter(torch.tensor(self.tau_m))
                self.tau_s = nn.Parameter(torch.tensor(self.tau_s))
                self.tv = (torch.exp(-tt / self.tau_m) - torch.exp(-tt / self.tau_s)) / (self.tau_m - self.tau_s)
                self.dv = (torch.exp(-tt / self.tau_s) / self.tau_s - torch.exp(-tt / self.tau_m) / self.tau_m) / (self.tau_m - self.tau_s)
                self.te = -torch.exp(-tt / self.tau_m)
                self.de = torch.exp(-tt / self.tau_m) / self.tau_m
            elif istrain_neurparam == 2:
                self.tau_m = nn.Parameter(torch.tensor(self.tau_m).view(1, 1, 1).repeat(1, self.num, 1))
                self.tau_s = nn.Parameter(torch.tensor(self.tau_s).view(1, 1, 1).repeat(1, self.num, 1))
                self.tv = (torch.exp(-tt / self.tau_m) - torch.exp(-tt / self.tau_s)) / (self.tau_m - self.tau_s)
                self.dv = (torch.exp(-tt / self.tau_s) / self.tau_s - torch.exp(-tt / self.tau_m) / self.tau_m) / (self.tau_m - self.tau_s)
                self.te = -torch.exp(-tt / self.tau_m)
                self.de = torch.exp(-tt / self.tau_m) / self.tau_m

    def clear(self):
        self.inputs_w_list = []
        self.inputs_t_list = []
        self.history_output_list = []

    def forward(self, input_w, input_t):
        '''
        NeuroFunc(inputs_w, inputs_t, history_output, template_tv, template_dv, template_te, template_de, threshold)
        :param input_w:
        :param input_t:
        :return:
        '''

        self.inputs_w_list.append(input_w)
        self.inputs_t_list.append(input_t)
        if len(self.history_output_list) == 0:
            self.history_output_list.append(torch.zeros_like(input_w))
        inputs_w = torch.cat(self.inputs_w_list, dim=-1)
        inputs_t = torch.cat(self.inputs_t_list, dim=-1)
        history_outputs = torch.cat(self.history_output_list, dim=-1)
        output = self.op(inputs_w, inputs_t, history_outputs, self.tv, self.dv, self.te, self.de, self.threshold)
        self.history_output_list.append(output)
        return output


class SpkLinear(nn.Module):

    def __init__(self, input_n, output_n, weight_init=nn.init.xavier_uniform_):
        super(SpkLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(output_n, input_n))
        self.neuron = SpikePropUnit((output_n,))
        weight_init(self.weight)

    def forward(self, input):
        input_w = F.linear(input.detach(), self.weight)
        input_t = F.linear(input, self.weight.detach())
        output = self.neuron(input_w, input_t)
        return output

    def clear(self):
        self.neuron.clear()




class TestNet(nn.Module):

    def __init__(self):
        super(TestNet, self).__init__()
        self.layer1 = SpkLinear(784, 100)
        self.layer2 = SpkLinear(100, 10)

    def forward(self, spk):
        spk = self.layer1(spk)
        spk = self.layer2(spk)
        return spk

    def clear(self):
        self.layer1.clear()
        self.layer2.clear()


bat_size = 100
root = '../Datasets/MNIST'
train_set = spaic.MNIST(root, is_train=True)
test_set = spaic.MNIST(root, is_train=False)

# 创建DataLoader迭代器
train_loader = spaic.Dataloader(train_set, batch_size=bat_size, shuffle=True, drop_last=False)
test_loader = spaic.Dataloader(test_set, batch_size=bat_size, shuffle=False)
net = TestNet()
for item in train_loader:
    data, label = item
    inp = torch.tensor(data, dtype=torch.float)
    out = net(inp)
    print(out)


