import os

os.chdir("../../")

import spaic
import torch

from tqdm import tqdm
import torch.nn.functional as F
from spaic.IO.Dataset import MNIST as dataset
from spaic.Network.ConnectPolicy import IncludedTypePolicy
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


from spaic.Neuron import NeuronModel


@spaic.NeuronGroup.custom_model(input_vars=['V', 'tau_m', 'dt', 'Isyn[updated]', 'O'], output_vars=['V'],
                                new_vars_dict={'V':0, 'tau_m':20.0, 'dt':0.1, 'O':0, 'Isyn':0,'Vth':1},
                                equation_type='iterative')
def MYLIF1(V, tau_m, dt, Isyn, O):
    tauM = torch.exp(-dt/tau_m)
    V = tauM * V + Isyn
    V = V - O * V
    return V


class MYLIF2Model(NeuronModel):
    """
    LIF model:
    # V(t) = tuaM * V^n[t-1] + Isyn[t]   # tauM: constant membrane time (tauM=RmCm)
    O^n[t] = spike_func(V^n[t-1])
    """

    def __init__(self, **kwargs):
        super(MYLIF2Model, self).__init__()

        self._variables['V'] = 0.0
        self._variables['O'] = 0.0
        self._variables['Isyn'] = 0.0

        self._parameter_variables['Vth'] = kwargs.get('v_th', 1)
        self._constant_variables['Vreset'] = kwargs.get('v_reset', 0.0)

        self._tau_variables['tauM'] = kwargs.get('tau_m', 20.0)  # _tau_variables 计算式为 torch.exp(-dt / tau_var)


        self._operations.append(('Vtemp', 'var_linear', 'tauM', 'V', 'Isyn[updated]'))
        self._operations.append(('O', 'threshold', 'Vtemp', 'Vth'))
        self._operations.append(('Resetting', 'var_mult', 'Vtemp', 'O[updated]'))
        self._operations.append(('V', 'minus', 'Vtemp', 'Resetting'))

NeuronModel.register("mylif", MYLIF2Model)


class Assemb1(spaic.Assembly):
    def __init__(self):
        super(Assemb1, self).__init__()
        self.assemb_layer1 = spaic.NeuronGroup(100, model=MYLIF1)
        self.assemb_layer2 = spaic.NeuronGroup(78, model=MYLIF1, neuron_type=['exc'])
        self.assemb_layer3 = spaic.NeuronGroup(10, model='lif', neuron_type=['exc'])

        self.assemb_conn12 = spaic.Connection(self.assemb_layer1, self.assemb_layer2, link_type='full',
                                              syn_type=['electrical'])
        self.assemb_conn23 = spaic.Connection(self.assemb_layer2, self.assemb_layer3, link_type='full')
        self.assemb_conn13 = spaic.Connection(self.assemb_layer1, self.assemb_layer3, link_type='full')

class Assemb2(spaic.Assembly):
    def __init__(self):
        super(Assemb2, self).__init__()
        self.assemb_layer1 = spaic.NeuronGroup(10, model='mylif')
        self.assemb_layer2 = spaic.NeuronGroup(10, model='lif')

        self.assemb_conn12 = spaic.Connection(self.assemb_layer1, self.assemb_layer2, link_type='one_to_one')

class TestNet(spaic.Network):
    def __init__(self):
        super(TestNet, self).__init__()
        self.input = spaic.Encoder(num=784, coding_method='poisson')

        self.layer1 = spaic.NeuronGroup(10, model='mylif')
        # self.layer2_part1 = Assemb1()
        # self.layer2_part2 = Assemb2()
        self.layer3 = Assemb1()
        self.layer4 = spaic.NeuronGroup(10, model='lif')
        self.output = spaic.Decoder(num=10, dec_target=self.layer4, coding_method='spike_counts')

        self.conn1 = spaic.Connection(self.input, self.layer1, link_type='full')
        self.conn1_21 = spaic.Projection(self.layer3, self.layer4, link_type='full',
                                         policies=[IncludedTypePolicy(pre_types=['exc'])])
        self.conn1_22 = spaic.Projection(self.layer3, self.layer4, link_type='full')
        self.conn2_3 = spaic.Projection(self.layer3, self.layer4, link_type='sparse')
        self.conn3_4 = spaic.Projection(self.layer3, self.layer4, link_type='full')

        self.learner = spaic.Learner(trainable=self, algorithm='STCA')
        self.learner.set_optimizer('Adam', 0.001)

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
