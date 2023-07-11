# encoding: utf-8
"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: test_Speech.py
@time:2023/7/11 16:50
@description:
"""
import os

os.chdir("../../")

import spaic
import torch

from spaic.Learning.STCA_Learner import STCA
from tqdm import tqdm
import torch.nn.functional as F
from spaic.IO.Dataset import MNISTVoices as dataset
import numpy as np

# 参数设置
if torch.cuda.is_available():
    device = 'cuda'
    print('cuda')
else:
    device = 'cpu'
    print('cpu')

# 创建训练数据集
root = '../datasets/MNISTVoices/AudioMNIST'

train_set = dataset(root, is_train=True, preprocessing='mfcc')
test_set = dataset(root, is_train=False, preprocessing='mfcc')

run_time = train_set.maxTime
node_num = train_set.maxNum
label_num = dataset.class_number
bat_size = 20
# 创建DataLoader迭代器
train_loader = spaic.Dataloader(train_set, batch_size=bat_size, shuffle=True, drop_last=True)
test_loader = spaic.Dataloader(test_set, batch_size=bat_size, shuffle=False, drop_last=True)
class TestNet(spaic.Network):
    def __init__(self):
        super(TestNet, self).__init__()

        # frontend setting
        # coding
        self.input = spaic.Encoder(num=node_num, coding_method='poisson', unit_conversion=0.01)

        # neuron group
        self.layer1 = spaic.NeuronGroup(500, model='lif')
        self.layer2 = spaic.NeuronGroup(500, model='lif',)
        self.layer3 = spaic.NeuronGroup(label_num, model='lif')

        # decoding0
        self.output = spaic.Decoder(num=label_num, dec_target=self.layer3, coding_method='spike_counts')

        # Connection
        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full')   # , w_std=0.002, w_mean=0.01
        self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='full')  # , w_std=0.001, w_mean=0.01
        self.connection3 = spaic.Connection(self.layer2, self.layer3, link_type='full')  # , w_std=0.0001, w_mean=0.01

        # # Minitor
        # self.mon_O = spaic.StateMonitor(self.input, 'O', nbatch=False)
        # self.mon_O1 = spaic.StateMonitor(self.layer1, 'O', nbatch=False)
        # self.mon_V1 = spaic.StateMonitor(self.layer1, 'V', nbatch=False)
        # self.mon_O2 = spaic.StateMonitor(self.layer2, 'O', nbatch=False)
        # self.mon_V2 = spaic.StateMonitor(self.layer2, 'V', nbatch=False)

        # Learner
        # self._learner1 = STCA(self,)
        self.learner = spaic.Learner(trainable=self, algorithm='sbp')
        self.learner.set_optimizer('Adam', 0.001)
        self.learner.set_schedule(lr_schedule_name='StepLR', step_size=2, gamma=0.1)
        self.set_backend('torch', device)

epoch_num = 30
repeat_run_num = 10
eval_losses = np.zeros([repeat_run_num, epoch_num])
eval_acces = np.zeros([repeat_run_num, epoch_num])
losses = np.zeros([repeat_run_num, epoch_num])
acces = np.zeros([repeat_run_num, epoch_num])
maxAcc = 0
network_param_name = 'DigitsVoice50Epoch_LIF_STCA_MFCC_Best'
for rep_time in range(repeat_run_num):
    Net = TestNet()
    print("\n Start running")

    num_correct = 0

    for epoch in range(30):

        # 训练阶段
        print("Start training")
        train_loss = 0
        train_acc = 0
        pbar = tqdm(total=len(train_loader))
        for i, item in enumerate(train_loader):
            # 前向传播
            data, label = item
            Net.input(data)
            Net.run(run_time)
            output = Net.output.predict
            output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.0001)
            label = torch.tensor(label, device=device)
            batch_loss = F.cross_entropy(output, label)

            # 反向传播
            Net.learner.optim_zero_grad()
            batch_loss.backward(retain_graph=False)
            Net.learner.optim_step()


            # 记录误差
            train_loss += batch_loss.item()
            predict_labels = torch.argmax(output, 1)
            num_correct = (predict_labels == label).sum().item()  # 记录标签正确的个数
            acc = num_correct / data.shape[0]
            train_acc = train_acc + acc


            pbar.set_description_str("[loss:%f]Batch progress: " % batch_loss.item())
            pbar.update()
        pbar.close()
        losses[rep_time][epoch] = (train_loss / len(train_loader))
        acces[rep_time][epoch] = (train_acc / len(train_loader))
        print('epoch:{},Train Loss:{:.4f},Train Acc:{:.4f}'.format(epoch, train_loss / len(train_loader), train_acc / len(train_loader)))

        if train_acc > maxAcc:
            maxAcc = train_acc
            Net.save_state(network_param_name)

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

            eval_losses[rep_time][epoch] = (eval_loss / len(test_loader))
            eval_acces[rep_time][epoch] = (eval_acc / len(test_loader))
        pbarTest.close()
        print('epoch:{},Test Loss:{:.4f},Test Acc:{:.4f}'.format(epoch, eval_loss / len(test_loader), eval_acc / len(test_loader)))


    acces_root = os.path.join(root, "acces_new.pt")
    torch.save(acces, acces_root)

    losses_root = os.path.join(root, "losses_new.pt")
    torch.save(losses, losses_root)

    eval_acces_root = os.path.join(root, "eval_acces_new.pt")
    torch.save(eval_acces, eval_acces_root)

    eval_losses_root = os.path.join(root, "eval_losses_new.pt")
    torch.save(eval_losses, eval_losses_root)