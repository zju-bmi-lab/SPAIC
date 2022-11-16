import os

os.chdir("../../")

import spaic
import torch

from tqdm import tqdm
import torch.nn.functional as F
from spaic.IO.Dataset import MNIST as dataset
import numpy as np

# 设备设置
if torch.cuda.is_available():
    device = 'cuda'
    print('cuda')
else:
    device = 'cpu'
    print('cpu')

# 创建训练数据集
root = './spaic/Datasets/MNIST'
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
        # coding
        self.input = spaic.Encoder(num=784, coding_method='poisson')
        # neuron group
        self.layer1 = spaic.NeuronGroup(10, model='lif')
        # decoding
        self.output = spaic.Decoder(num=10, dec_target=self.layer1, coding_method='spike_counts')

        # Connection
        self.connection1 = spaic.Connection(pre=self.input, post=self.layer1, link_type='full_connection')
        # Learner
        self.learner = spaic.Learner(trainable=self, algorithm='STCA')
        self.learner.set_optimizer('Adam', 0.001)
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        backend = spaic.Torch_Backend(device)
        backend.dt = 0.1
        self.set_backend(backend)

        self.mon_V = spaic.StateMonitor(self.layer1, 'V')
        self.mon_I = spaic.StateMonitor(self.input, 'O')
        self.spk_l1 = spaic.SpikeMonitor(self.layer1, 'O')

Net = TestNet()
Net.build()

print("Start running")

eval_losses = []
eval_acces = []
losses = []
acces = []
num_correct = 0
num_sample = 0
train_accuracy = []
test_accuracy = []
for epoch in range(100):

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

        # 记录误差
        train_loss += batch_loss.item()
        predict_labels = torch.argmax(output, 1)
        num_correct = (predict_labels == label).sum().item()  # 记录标签正确的个数
        acc = num_correct / data.shape[0]
        train_acc += acc

        pbar.set_description_str("[loss:%f]Batch progress: " % batch_loss.item())
        pbar.update()
    # train_accuracy.append(train_acc)
    pbar.close()
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader) * 100)
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
    test_accuracy.append(eval_acc)
    pbarTest.close()
    print('epoch:{},Test Loss:{:.4f},Test Acc:{:.4f}'.format(epoch,eval_loss / len(test_loader), eval_acc / len(test_loader)))

from matplotlib import pyplot as plt
plt.subplot(2, 1, 1)
plt.plot(acces)
plt.title('Train Accuracy')
plt.ylabel('Acc')
plt.xlabel('epoch')

plt.subplot(2, 1, 2)
plt.plot(test_accuracy)
plt.title('Test Accuracy')
plt.ylabel('Acc')
plt.xlabel('epoch')

plt.show()