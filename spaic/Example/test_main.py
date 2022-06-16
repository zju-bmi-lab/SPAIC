# -*- coding: utf-8 -*-
"""
Created on 2022/1/4
@project: SPAIC
@filename: test_main
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description: 
"""
import spaic as SPAIC
# Spike-based Artificial Intelligence Computing



from matplotlib.pyplot import *
import torch
import torch.nn.functional as F
from matplotlib.pyplot import *


# 创建数据集
from spaic.IO.Dataset import MNIST
root = 'D://Datasets/MNIST'
train_set = MNIST(root, is_train=True)
test_set = MNIST(root, is_train=False)

# 创建DataLoader迭代器
bat_size = 200
train_loader = SPAIC.Dataloader(train_set, batch_size=bat_size, shuffle=True, drop_last=False)
test_loader = SPAIC.Dataloader(test_set, batch_size=bat_size, shuffle=False)

beta = 12.0/8.0
V0 = (1/(beta-1))*(beta**(beta/(beta-1)))
Rec = []
@SPAIC.NeuronGroup.custom_model(input_vars=['M', 'S', 'E', 'O','Isyn', 'tauM', 'tauS', 'tauE', 'Vth'],
                                output_vars=['V', 'M', 'S', 'E'],
                                new_vars_dict={'M':0, 'S':0, 'E':0, 'O':0, 'V':0, 'Isyn':0, 'tauM':12.0, 'tauS':8.0, 'tauE':20.0, 'Vth':1},
                                equation_type='exp_euler_iterative')
def custom_clif(M, S, E, O, Isyn, tauM, tauS, tauE, Vth):
    M = tauM*M + V0*Isyn
    S = tauS*S + V0*Isyn
    E = tauE*E + Vth*O
    V = M-S-E
    # Rec.append(V)
     # print(len(Rec))
    return V, M, S, E


class TestNet(SPAIC.Network):
    def __init__(self):
        super(TestNet, self).__init__()

        # frontend setting
        # coding
        self.input = SPAIC.Encoder(num=784, coding_method='poisson',unit_conversion=1.0)
        # neuron group
        self.layer1 = SPAIC.NeuronGroup(400, neuron_model=custom_clif)
        self.layer2 = SPAIC.NeuronGroup(10, neuron_model=custom_clif)
        # decoding
        self.output = SPAIC.Decoder(num=10, dec_target=self.layer2, coding_method='spike_counts')

        # Connection
        self.connection1 = SPAIC.Connection(self.input, self.layer1, link_type='full', w_mean=0.005, w_std=0.05)
        self.connection2 = SPAIC.Connection(self.layer1, self.layer2, link_type='full', w_mean=0.005, w_std=0.05)

        # Monitor
        # self.mon_V = SPAIC.StateMonitor(self.layer1, 'V', index=[[1, 3, 5]]) # index=[[0, 0, 0], [1, 3, 5]]
        self.SpkM = SPAIC.SpikeMonitor(self.layer1)

        # Learner
        self.learner = SPAIC.Learner(trainable=self, algorithm='STCA', alpha=0.5)
        self.learner.set_optimizer('Adam', 0.001)


Net = TestNet()
Net.set_backend('pytorch',device='cuda')
Net.build()
print(Net)
print(Net.connection1.get_value('weight'))

# for data, label in train_loader:
#     Net.input(data)
#     Net.run(10.0)
#     plot(Net.mon_V.values[0,...].transpose())
#     show()
#     break

run_time = 50.0
from tqdm import tqdm
for epoch in range(5):

    # 训练阶段
    print("Start training")
    train_loss = 0
    train_acc = 0
    pbar = tqdm(total=len(train_loader))
    for i, item in enumerate(train_loader):
        # 前向传播
        data, label = item
        Net.input(0.5*data)
        Net.output(label)
        Net.run(run_time)
        output = Net.output.predict
        output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
        label = torch.tensor(label, device='cuda')
        batch_loss = torch.nn.functional.cross_entropy(output, label)
        # 反向传播
        Net.learner.optim_zero_grad()
        batch_loss.backward(retain_graph=False)
        Net.learner.optim_step()

        pbar.set_description_str("[loss:%f]Batch progress: " % batch_loss.item())
        pbar.update()

    # pbar.close()
    # spk_t = Net.SpkM.spk_times[0]
    # spk_i = Net.SpkM.spk_index[0]
    # plot(spk_t,spk_i, '.')
    # show()


# ======================================== Mean Field ======================================

#
#
# # 定义权重位置
# weight1 = np.zeros((1, 1, 29, 29))
# weight2 = np.zeros((1, 1, 29, 29))
# a = 2.5
# b = 4.0
# w = 10.0
# for x in range(29):
#     for y in range(29):
#         dist = ((x-14.0)**2 + (y-14.0)**2)**0.5
#         weight1[0,0,x,y] = w*np.exp(-dist/a)/a
#         weight2[0,0,x,y] = w*np.exp(-dist/b)/b
#
# imshow(weight2[0,0,...])
# show()
#
#
# CANNNet = SPAIC.Network()
# with CANNNet:
#     input = SPAIC.Generator(num=200*200, shape=(1, 200, 200), coding_method='poisson_generator')
#     exc_layer = SPAIC.NeuronGroup(neuron_number=200 * 200, shape=(1, 200, 200),  neuron_model='meanfield', tau=1.0)
#     inh_layer = SPAIC.NeuronGroup(neuron_number=200 * 200, shape=(1, 200, 200), neuron_model='meanfield', tau=2.0)
#     inp_link = SPAIC.Connection(input, exc_layer, link_type='conv', in_channels=1, out_channels=1, kernel_size=(29,29), maxpool_on=False, padding=14, weight=weight1, post_var_name='Iext')
#     ee_link = SPAIC.Connection(exc_layer, exc_layer, link_type='conv', in_channels=1, out_channels=1, kernel_size=(29,29), maxpool_on=False, padding=14, weight=weight2, post_var_name='WgtSum')
#     ei_link = SPAIC.Connection(exc_layer, inh_layer, link_type='conv', in_channels=1, out_channels=1,
#                                  kernel_size=(29, 29), maxpool_on=False, padding=14, weight=weight2,
#                                  post_var_name='WgtSum')
#     ie_link = SPAIC.Connection(inh_layer, exc_layer, link_type='conv', in_channels=1, out_channels=1,
#                                  kernel_size=(29, 29), maxpool_on=False, padding=14, weight=-3.0*weight1,
#                                  post_var_name='WgtSum')
#     ii_link = SPAIC.Connection(inh_layer, inh_layer, link_type='conv', in_channels=1, out_channels=1,
#                                  kernel_size=(29, 29), maxpool_on=False, padding=14, weight=-weight1,
#                                  post_var_name='WgtSum')
#
#     om = SPAIC.StateMonitor(exc_layer, 'O')
#
#
#
# CANNNet.set_backend('pytorch')
# CANNNet.set_simulation_dt(0.2)
#
# # ion()
# # inp = np.zeros((1,1,200, 200))
# # inp[0,0, 100, 100] = 5.0
# for kk in range(25):
#     # if kk == 1:
#     #     inp[0, 0, 100, 100] = 0.0
#     input(0.01)
#     CANNNet.run_continue(10.0)
#     out = om.values
#     # imshow(CANNNet._backend._variables['CANNNet<net>_inter_link<con>:CANNNet<net>_layer<neg><-CANNNet<net>_layer<neg>:{weight}'].detach().numpy())
#     # show()
#     om.init_record()
#     timelen = out.shape[-1]
#     print(kk)
#     fig = figure(1)
#     for ii in range(timelen):
#         imshow(out[0,0,:,:,ii])
#         draw()
#         pause(0.1)
#         clf()
#
#
#
#
#
#
