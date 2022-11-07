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
import os
os.chdir("../../")
root = './spaic/Datasets/MNIST'
train_set = MNIST(root, is_train=True)
test_set = MNIST(root, is_train=False)

# 创建DataLoader迭代器
bat_size = 100
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

latWeight = 0.001*np.random.randn(200, 200)*(1.0-np.eye(200))

class TestNet(SPAIC.Network):
    def __init__(self):
        super(TestNet, self).__init__()

        # frontend setting
        # coding
        self.input = SPAIC.Encoder(num=784, coding_method='constant_current', input_norm=True, amp=0.5)#,unit_conversion=1.0)
        # neuron group
        self.layer1 = SPAIC.NeuronGroup(400, model='complex', tau_m=20.0, tau_r=100.0)
        self.layer2 = SPAIC.NeuronGroup(300, model='complex')
        self.layer3 = SPAIC.NeuronGroup(10, model='complex')
        # decoding
        # self.output = SPAIC.Decoder(num=10, dec_target=self.layer3, coding_method='complex_count')
        self.output = SPAIC.Decoder(num=10, dec_target=self.layer3, coding_method='complex_phase')

        # Connection
        self.connection1 = SPAIC.Connection(self.input, self.layer1, link_type='full', w_mean=0.002, w_std=0.005)
        self.connection2 = SPAIC.Connection(self.layer1, self.layer2, link_type='full', w_mean=0.002, w_std=0.005)
        self.connection3 = SPAIC.Connection(self.layer2, self.layer3, link_type='full', w_mean=0.002, w_std=0.005)
        # self.connection4 = SPAIC.Connection(self.layer2, self.layer2, link_type='full', w_mean=-0.005, w_std=0.000)
        # self.connection4 = SPAIC.Connection(self.layer1, self.layer1, link_type='full', w_mean=-0.005, w_std=0.0)
  #                                          ,synapse_type='delay_complex_synapse', syn_kwargs={'max_delay':10.0, 'delay_len':5})

        # Monitor
        self.mon_V = SPAIC.StateMonitor(self.layer2, 'V', get_grad=False) #  index=[[1, 3, 5]],index=[[0, 0, 0], [1, 3, 5]]
        # self.SpkM = SPAIC.SpikeMonitor(self.input)
        self.SpkM1 = SPAIC.SpikeMonitor(self.layer1)
        self.SpkM2 = SPAIC.SpikeMonitor(self.layer2)
        self.SpkM3 = SPAIC.SpikeMonitor(self.layer3)


        # Learner
        self.learner = SPAIC.Learner(trainable=self, algorithm='STCA', alpha=0.5)#[self.connection1,self.connection2,self.connection3]
        self.learner.set_optimizer('Adam', 0.0001)
        self.learner.set_schedule('ExponentialLR', gamma=0.96)


Net = TestNet()
Net.set_backend('pytorch',device='cuda')
Net.set_backend_dt(0.2)
Net.build()

print(Net)
# print(Net.connection1.get_value('weight'))

# for data, label in train_loader:
#     Net.input(data)
#     Net.run(10.0)
#     plot(Net.mon_V.values[0,...].transpose())
#     show()
#     break

figure()
ion()
run_time = 50.0
from tqdm import tqdm
for epoch in range(100):

    # 训练阶段
    Net.train()
    print("Start training")
    train_loss = 0
    train_acc = 0
    pbar = tqdm(total=len(train_loader))
    # trydata, trylabel = train_loader.try_fetch()
    for i, item in enumerate(train_loader):
        # 前向传播
        data, label = item
        # data = trydata#np.concatenate([trydata[:1,:], data], axis=0)
        # label = trylabel#np.concatenate([trylabel[:1], label], axis=0)
        Net.input(0.2*data)
        Net.output(label)
        Net.run(run_time)
        output, rate = Net.output.predict
        rate = torch.mean(rate)
        # mean_out = torch.mean(output)
        # output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
        label = torch.tensor(label, device='cuda')
        batch_loss = torch.nn.functional.cross_entropy(output, label)
        # 反向传播
        Net.learner.optim_zero_grad()
        (batch_loss).backward(retain_graph=False)
        Net.learner.optim_step()
        predict_labels = torch.argmax(output, 1)
        num_correct = (predict_labels == label).sum().item()  # 记录标签正确的个数
        acc = num_correct / data.shape[0]
        train_acc += acc

        # plot(Net.mon_V.values[0,0,:].real)
        # show()



        # if i > 1:
        #     clf()
        #     ave = Net.layer1.model.running_ave.cpu().numpy()
        #     std = Net.layer1.model.running_std.cpu().numpy()
        #     subplot(211)
        #     hist(ave)
        #     subplot(212)
        #     hist(std)
        #     draw()
        #     pause(0.1)
        # show()
        # slow_ave = Net.layer1.model.fast_ave_o.cpu().numpy()
        # bias = Net.layer1.model.running_bias.cpu().numpy()
        # scale = Net.layer3.model.running_scale.cpu().numpy()
        # clf()
        # subplot(221)
        # hist(fast_ave)
        # subplot(222)
        # hist(slow_ave)
        # subplot(223)
        # hist(bias)
        # subplot(224)
        # hist(scale)
        clf()
        # for ii in range(100):
        subplot(211)
        # tim = Net.SpkM.spk_times[0]
        # ind = Net.SpkM.spk_index[0]
        tim1 = Net.SpkM1.spk_times[0]
        ind1 = Net.SpkM1.spk_index[0]
        tim2 = Net.SpkM2.spk_times[0]
        ind2 = Net.SpkM2.spk_index[0]
        tim3 = Net.SpkM3.spk_times[0]
        ind3 = Net.SpkM3.spk_index[0]
        label_tim = []
        label_ind = []
        for nn, ni in enumerate(ind3):
            if ni == label[0]:
                label_ind.append(ni)
                label_tim.append(tim3[nn])
        label_ind = np.array(label_ind)
        # plot(tim, ind, '.')
        plot(tim1, ind1, '.')
        plot(tim2, ind2+400, '.')
        plot(tim3, ind3+700, '.')
        plot(label_tim, label_ind+700,'.r')
        subplot(212)
        plot(Net.mon_V.values[0, :10, :].transpose().real)
        draw()
        pause(0.1)



        # count = len(Net.SpkM.spk_times[0])
        # values = Net.mon_V.values[0]
        # grads = Net.mon_V.grads[0]
        # subplot(311)
        # plot(values[:, :].real.transpose())
        # subplot(312)
        # plot(grads[:, :].real.transpose())
        # subplot(313)
        # plot(grads[:, :].imag.transpose())
        # show()
        # gw1 = torch.std(Net.connection1.get_value('weight').grad).item()
        # gw2 = torch.std(Net.connection2.get_value('weight').grad).item()
        # gw3 = torch.std(Net.connection3.get_value('weight').grad).item()
        # print(gw1,gw2,gw3)
        if hasattr(Net.connection3, 'running_ave') and isinstance(Net.connection3.running_ave, torch.Tensor):
            ave = torch.mean(Net.connection3.running_ave).item()
            std = torch.mean(Net.connection3.running_std).item()
            bias = torch.mean(Net.connection3.running_bias).item()
            scale = torch.mean(Net.connection3.running_scale).item()
        else:
            ave = 0.0
            std = 0.0
            bias = 0.0
            scale = 0.0
        pbar.set_description_str("[epoch:%d][loss:%.4f acc:%.4f, rate:%.4f, mean_gw:%.4f:%.4f:%.4f:%.4f]Batch progress: "
                                 %(epoch+1,batch_loss.item(), train_acc/(i+1.0), rate.item(), ave, std, bias, scale))#gw.item()mean_out.item()
        pbar.update()
    pbar.close()
    Net.learner.optim_shedule()

    Net.eval()
    eval_loss = 0
    eval_acc = 0
    pbarTest = tqdm(total=len(test_loader))
    with torch.no_grad():
        for i, item in enumerate(test_loader):
            data, label = item
            Net.input(0.5*data)
            Net.run(run_time)
            output,_ = Net.output.predict
            label = torch.tensor(label, device='cuda')
            batch_loss = F.cross_entropy(output, label)
            eval_loss += batch_loss.item()

            _, pred = output.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / data.shape[0]
            eval_acc += acc
            pbarTest.set_description_str("[loss:%f]Batch progress: " % batch_loss.item())
            pbarTest.update()
    pbarTest.close()
    print('epoch:{},Test Loss:{:.4f},Test Acc:{:.4f}'.format(epoch, eval_loss / len(test_loader),
                                                             eval_acc / len(test_loader)))
    # b1 = torch.mean(Net.layer1.model.running_bias).item()
    # s1 = torch.mean(Net.layer1.model.running_scale).item()
    # r1 = torch.mean(Net.layer1.model.running_ave).item()
    # std1 = torch.mean(Net.layer1.model.running_std).item()
    # b2 = torch.mean(Net.layer2.model.running_bias).item()
    # s2 = torch.mean(Net.layer2.model.running_scale).item()
    # r2 = torch.mean(Net.layer2.model.running_ave).item()
    # std2 = torch.mean(Net.layer2.model.running_std).item()
    # b3 = torch.mean(Net.layer3.model.running_bias).item()
    # s3 = torch.mean(Net.layer3.model.running_scale).item()
    # r3 = torch.mean(Net.layer3.model.running_ave).item()
    # std3 = torch.mean(Net.layer3.model.running_std).item()
    #
    # print('l2{','rate:%.4f'%r2, 'std:%.4f'%std2, 'bias:%.4f'%b2, 'scale:%.4f'%s2,'}',
    #       'l3{','rate:%.4f'%r3, 'std:%.4f'%std3, 'bias:%.4f'%b3, 'scale:%.4f'%s3, '}')


        # spk_t = Net.SpkM.spk_times[0]
        # spk_i = Net.SpkM.spk_index[0]
        # plot(spk_t,spk_i, '.')
        # values = Net.mon_V.values[0]
        # grads = Net.mon_V.grads[0]
        # plot(grads[0,:].real)
        # plot(grads[:, :].imag.transpose())
        # # hist(Net.connection1.get_value('weight').grad.cpu().view(-1).numpy(),bins=100)
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
