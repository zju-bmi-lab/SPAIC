# encoding: utf-8
"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: testSpeech.py
@time:2021/4/14 14:34
@description:
"""
import spaic
import torch
import math

from spaic.Learning.STCA_Learner import STCA
from tqdm import tqdm
import torch.nn.functional as F
from spaic.IO.Dataset import MNIST as dataset
import numpy as np
import matplotlib.pyplot as plt

# 设备设置
if torch.cuda.is_available():
    device = 'cuda'
    print('cuda')
else:
    device = 'cpu'
    print('cpu')

# 创建训练数据集
# root = r'F:\GitCode\Python\dataset\Heidelberg Spiking Data Sets\Spiking Heidelberg Digits'
# root = r'F:\GitCode\Python\dataset\Heidelberg Spiking Data Sets\Spiking Speech Command'
root = 'D:\Datasets\MNIST'
train_set = dataset(root, is_train=True)
test_set = dataset(root, is_train=False)

run_time = 50.0
node_num = dataset.maxNum
label_num = dataset.class_number
bat_size = 100
# 创建DataLoader迭代器
train_loader = spaic.Dataloader(train_set, batch_size=bat_size, shuffle=True, drop_last=False)
test_loader = spaic.Dataloader(test_set, batch_size=bat_size, shuffle=False)
class TestNet(spaic.Network):
    def __init__(self):
        super(TestNet, self).__init__()

        # frontend setting
        # coding

        self.input = spaic.Encoder(num=node_num, coding_method='poisson')

        # neuron group
        self.layer1 = spaic.NeuronGroup(node_num, neuron_model='lif')

        self.layer2 = spaic.NeuronGroup(label_num, neuron_model='lif')

        # decoding
        self.output = spaic.Decoder(num=label_num, dec_target=self.layer2, coding_method='spike_counts')

        # Connection
        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full')
        self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='full')

        # # Minitor
        # self.mon_O2 = spaic.StateMonitor(self.layer2, 'O', get_grad=True, nbatch=False)
        # self.mon_I2 = spaic.StateMonitor(self.layer2, 'I', get_grad=True, nbatch=False)
        # self.mon_W2 = spaic.StateMonitor(self.connection2, 'weight', get_grad=True, nbatch=False)
        # self.mon_O1 = spaic.StateMonitor(self.layer1, 'O', get_grad=True, nbatch=False)
        # self.mon_I1 = spaic.StateMonitor(self.layer1, 'I', get_grad=True, nbatch=False)
        # self.mon_W1 = spaic.StateMonitor(self.connection1, 'weight', get_grad=True, nbatch=False)
        # Learner

        self.learner = spaic.Learner(trainable=self, algorithm='STCA')
        self.learner.set_optimizer('Adam', 0.001)



Net = TestNet()
Net.set_backend('torch', device)
print("Start running")

eval_losses = []
eval_acces = []
losses = []
acces = []
num_correct = 0
num_sample = 0
for epoch in range(4):

    # 训练阶段
    print("Start training")
    train_loss = 0
    train_acc = 0
    pbar = tqdm(total=len(train_loader))
    for i, item in enumerate(train_loader):
        # 前向传播
        data, label = item
        Net.input(data)
        Net.output(label)
        Net.run(run_time)
        output = Net.output.predict
        output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
        label = torch.tensor(label, device=device)
        batch_loss = F.cross_entropy(output, label)

        # 反向传播
        Net.learner.optim_zero_grad()
        batch_loss.backward(retain_graph=False)
        Net.learner.optim_step()

        # mon_O1 = Net.mon_O1.values
        # spikes = mon_O1.sum(2)
        # mean_spikes = np.sum(spikes, axis=1, keepdims=True)
        # spikes = spikes/mean_spikes
        # plt.imshow(spikes)
        #
        # plt.figure()
        # grads_o2 = Net.mon_O2.grads
        # plt.subplot(3, 1, 1)
        # plt.imshow(grads_o2[0, :, :])
        # grads_I2 = Net.mon_I2.grads
        # plt.subplot(3, 1, 2)
        # plt.imshow(grads_I2[0, :, :])
        # grads_W2 = Net.mon_W2.grads
        # plt.subplot(3, 1, 3)
        # plt.imshow(grads_W2[:, :, 0])
        # grads_o1 = Net.mon_O1.grads
        # plt.figure()
        # plt.subplot(3, 1, 1)
        # plt.imshow(grads_o1[0, :, :])
        # grads_I1 = Net.mon_I1.grads
        # plt.subplot(3, 1, 2)
        # plt.imshow(grads_I1[0, :, :])
        # grads_W1 = Net.mon_W1.grads
        # plt.subplot(3, 1, 3)
        # plt.imshow(grads_W1[:, :, 0])
        # plt.show()



        # 记录误差
        train_loss += batch_loss.item()
        predict_labels = torch.argmax(output, 1)
        num_correct = (predict_labels == label).sum().item()  # 记录标签正确的个数
        acc = num_correct / data.shape[0]
        train_acc += acc

        pbar.set_description_str("[loss:%f]Batch progress: " % batch_loss.item())
        pbar.update()
    pbar.close()
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    print('epoch:{},Train Loss:{:.4f},Train Acc:{:.4f}'.format(epoch, train_loss / len(train_loader), train_acc / len(train_loader)))

    # 测试阶段
    eval_loss = 0
    eval_acc = 0
    print("Start testing")
    pbarTest = tqdm(total=len(test_loader))
    with torch.no_grad():
        for i, item in enumerate(test_loader):
            data, label = item
            Net.input(data)
            Net.run(run_time)
            output = Net.output.predict
            output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
            label = torch.tensor(label, device=device)
            batch_loss = F.cross_entropy(output, label)
            eval_loss += batch_loss.item()

            _, pred = output.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / data.shape[0]
            eval_acc += acc
            pbarTest.set_description_str("[loss:%f]Batch progress: " % batch_loss.item())
            pbarTest.update()
        eval_losses.append(eval_loss / len(test_loader))
        eval_acces.append(eval_acc / len(test_loader))
    pbarTest.close()
    print('epoch:{},Test Loss:{:.4f},Test Acc:{:.4f}'.format(epoch,eval_loss / len(test_loader), eval_acc / len(test_loader)))