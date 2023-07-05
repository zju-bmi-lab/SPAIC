#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件说明: 测试网络的保存和加载
@Time: 2023-04-13 22:05:28
@Author: 
"""

import os

os.chdir("../../")

from tqdm import tqdm
import torch
import torch.nn.functional as F

import spaic
from spaic.Library import Network_loader, Network_saver
from spaic.IO.Dataset import MNIST as dataset


# ⌊(i + 2p - k) / s⌋ + 1
class TestNet(spaic.Network):
    def __init__(self, dt, run_time):
        super(TestNet, self).__init__()
        input_shape = (1, 28, 28)
        layer1_neuron_shape = (4, 13, 13)
        layer3_neuron_shape = (10,)
        conn1_link_type = 'conv'
        conn1_in_channels = 1
        conn1_out_channels = 4
        conn1_kernel_size = (3, 3)
        conn1_stride = 2
        conn1_padding = 0

        conn3_link_type = 'full'

        state_monitor = 'V'

        self.input = spaic.Encoder(shape=input_shape, dt=dt, time=run_time, coding_method='poisson')
        self.layer1 = spaic.NeuronGroup(shape=layer1_neuron_shape, model='if')
        self.layer3 = spaic.NeuronGroup(shape=layer3_neuron_shape, model='if')

        self.connection1 = \
            spaic.Connection(self.input, self.layer1,
                             link_type=conn1_link_type, syn_type=['conv'], in_channels=conn1_in_channels,
                             out_channels=conn1_out_channels, kernel_size=conn1_kernel_size, stride=conn1_stride,
                             padding=conn1_padding)

        self.connection3 = spaic.Connection(self.layer1, self.layer3,
                                            link_type=conn3_link_type, syn_type=['flatten', 'basic'])

        self._learner = spaic.Learner(algorithm='STCA', trainable=[self])
        self.output = spaic.Decoder(num=10, dec_target=self.layer3, time=run_time, coding_method='spike_counts')
        self.monitor2 = spaic.SpikeMonitor(self.layer3)
        self.monitor3 = spaic.StateMonitor(self.layer3, state_monitor)


dt = 0.1
run_time = 3
# 设备设置
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# 创建训练数据集
root = './spaic/Datasets/MNIST'
train_set = dataset(root, is_train=True)

bat_size = 100
# 创建DataLoader迭代器
train_loader = spaic.Dataloader(train_set, batch_size=bat_size, shuffle=False, drop_last=False)
backend = spaic.Torch_Backend(device)
sim_name = backend.backend_name.lower()
net = TestNet(dt=dt, run_time=run_time)
net.set_backend(backend)
net.set_backend_dt(dt)
net.build(backend)  # 创建网络
param = net.get_testparams()  # 得到网络模型参数
optim = torch.optim.Adam(param, lr=0.001)  # 创建优化器对象，并传入网络模型的参数
print("Start running")
losses = []
acces = []
# with torch.autograd.set_detect_anomaly(True):
for epoch in range(1):
    # 训练阶段
    pbar = tqdm(total=len(train_loader))
    train_loss = 0
    train_acc = 0

    for i, item in enumerate(train_loader):
        # 前向传播
        data, label = item
        # print(train_loader.batch_size)
        data = data.reshape((data.shape[0], 1, 28, 28))
        net.input(data)
        net.output(label)
        net.run(run_time)
        output = net.output.predict
        if sim_name == 'pytorch':
            label = torch.tensor(label, device=device, dtype=torch.long)  # .unsqueeze(dim=1)

        batch_loss = F.cross_entropy(output, label)
        optim.zero_grad()
        batch_loss.backward(retain_graph=False)
        optim.step()

        train_loss += batch_loss.item()
        predict_labels = torch.argmax(output, 1)
        num_correct = (predict_labels == label.view(-1)).sum().item()  # 记录标签正确的个数
        acc = num_correct / data.shape[0]
        train_acc += acc
        # out1 = np.mean(np.sum(Net.layer1_O.values[0,...], axis=-1))

        pbar.set_description_str(
            "[loss:%f, acc:%f]Batch progress: " % (batch_loss.item(), acc))
        pbar.update()
        # break
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    print('train_acc', train_acc / len(train_loader))
    pbar.close()

# 保存网络
Network_saver.network_save(
    Net=net,
    path=os.getcwd(),
    filename="saved_net",
    combine=False,
    save_weight=True)

load_net = Network_loader.network_load(
    path=os.getcwd(),
    filename="saved_net", device=str(device),
    load_weight=True)

print("finish")
