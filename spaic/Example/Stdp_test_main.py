import os

os.chdir("../../")
import spaic
import torch
import numpy as np
# from spaic.Learning.STCA_Learner import STCA
from tqdm import tqdm
import torch.nn.functional as F

from spaic.Learning.Learner import Learner

from spaic.IO.Dataset import MNIST as dataset
from spaic.Library.Network_saver import network_save

# import pandas as pd
# 参数设置

# 设备设置
SEED = 0
np.random.seed(SEED)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device = 'cpu'
print(device)
backend = spaic.Torch_Backend(device)
backend.dt = 0.1
sim_name = backend.backend_name
sim_name = sim_name.lower()

# 创建训练数据集

root = './spaic/Datasets/MNIST'
train_set = dataset(root, is_train=False)
test_set =dataset(root, is_train=False)

run_time = 256 * backend.dt
node_num = 784
label_num = 100
bat_size = 1

# 创建DataLoader迭代器
train_loader = spaic.Dataloader(train_set, batch_size=bat_size, shuffle=False, drop_last=False)
test_loader = spaic.Dataloader(test_set, batch_size=bat_size, shuffle=False)


class TestNet(spaic.Network):
    def __init__(self):
        super(TestNet, self).__init__()

        # coding
        # self.input = spaic.Encoder(num=node_num, coding_time=run_time, coding_method='blank_encoder')
        self.input = spaic.Encoder(num=node_num, coding_time=run_time, coding_method='poisson', unit_conversion=0.6375)

        # neuron group
        self.layer1 = spaic.NeuronGroup(label_num, neuron_model='lifstdp_ex')
        self.layer2 = spaic.NeuronGroup(label_num, neuron_model='lifstdp_ih')
        # self.layer3 = TestAss()

        # decoding
        self.output = spaic.Decoder(num=label_num, dec_target=self.layer1, coding_time=run_time,
                                      coding_method='spike_counts')

        # Connection
        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full',
                                              weight=(np.random.rand(label_num, 784) * 0.3))
        self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='full',
                                              weight=(np.diag(np.ones(label_num))) * 22.5)
        self.connection3 = spaic.Connection(self.layer2, self.layer1, link_type='full', weight=(np.ones(
            (label_num, label_num)) - (np.diag(np.ones(label_num)))) * (-120))


        # Learner
        # self._learner = STCA(0.5, self)
        self._learner = Learner(algorithm='nearest_online_stdp', trainable=self.connection1, run_time=run_time)

        # self._learner = Learner(algorithm='nearest_online_stdp_inial', trainable=self.connection1, run_time=run_time)
        # self._learner = Learner(algorithm='nearest_online_STDP', trainable=self.connection1, run_time=run_time)

        # Minitor
        # self.mon_V1 = spaic.StateMonitor(self.layer1, 'V')
        # self.mon_th_theta = spaic.StateMonitor(self.layer1, 'Vth_theta')
        # self.mon_O1 = spaic.StateMonitor(self.layer1, 'O')
        # self.mon_O2 = spaic.StateMonitor(self.layer2, 'O')
        self.mon_weight = spaic.StateMonitor(self.connection1, 'weight', nbatch=-1)
        # self.mon_I1 = spaic.StateMonitor(self.input, 'O')
        # self.mon_weight2 = spaic.StateMonitor(self.connection2, 'weight')

        self.set_backend(backend)


Net = TestNet()
Net.build(backend)
# Net.mon_weight.plot_heatmap(time_id=-1, linewidths=0, linecolor='white', reshape=True, new_shape=(280,28))

print("Start running")

eval_losses = []
eval_acces = []
losses = []
acces = []
spike_output = [[]] * 10
# spike_output_test = [[]] * 10
im = None
# with torch.autograd.set_detect_anomaly(True):
for epoch in range(1):
    # 训练阶段
    pbar = tqdm(total=len(train_loader))
    train_loss = 0
    train_acc = 0

    for i, item in enumerate(train_loader):
        # if i > 100:
        #     break
        # 前向传播
        data, label = item
        Net.input(data)
        Net.output(label)
        Net.run(run_time)
        # print(label)

        output = Net.output.predict
        if spike_output[label[0]] == []:
            spike_output[label[0]] = [output]
        else:
            spike_output[label[0]].append(output)

        if sim_name == 'pytorch':
            label = torch.tensor(label, device=device, dtype=torch.long)

        # assign_label = []
        # if i % 500 == 0 and i > 0:
        #     a = []
        #     for i in range(len(spike_output)):
        #         a.append(sum(spike_output[i]) / len(spike_output[i]))
        #     assign_label = torch.argmax(torch.cat((a), 0), 0)

        # im = Net.mon_weight.plot_weight(time_id=-1, linewidths=0, linecolor='white', reshape=True, n_sqrt=10, side=28, im=im)

        # out1 = np.mean(np.sum(Net.layer1_O.values[0,...], axis=-1))

        # pbar.set_description_str("[loss:%f, acc:%f]Batch progress: " % (batch_loss.item(), acc))
        pbar.update()

    a = [sum(spike_output[i]) / len(spike_output[i]) for i in range(len(spike_output))]
    # for i in range(len(spike_output)):
    #     a.append(sum(spike_output[i]) / len(spike_output[i]))
        # train_label.append([torch.argmax(a[i])])
    assign_label = torch.argmax(torch.cat((a), 0), 0)
    # Net.save_state('dict')
    # im = Net.mon_weight.plot_weight(time_id=-1, linewidths=0, linecolor='white', reshape=True, n_sqrt=10, side=28,im=im)
    # network_dir = network_save(Net, 'TestNet', 'json')
    # Net.mon_weight.plot_heatmap(time_id=-1, linewidths=0, linecolor='white', reshape=True, new_shape=(280, 28))

    # for epoch in range(5):
    #     train()

    torch.save(Net.get_testparams(), 'tained_weight.pt')
    # data = torch.load('tained_weight.pt')
    # value = data[0].cpu().detach().numpy()
    # value = value.reshape(10, 10, 28, 28)
    # value = value.transpose(0, 2, 1, 3)
    # value = value.reshape(280, 280)
    # square_weights = value
    # import matplotlib.pyplot as plt
    # plt.imshow(square_weights)
    # plt.show()

    # def test():
    # 测试阶段
    eval_loss = 0
    eval_acc = 0
    pbarTest = tqdm(total=len(test_loader))
    with torch.no_grad():
        for i, item in enumerate(test_loader):
            # if i > 500 :
            #     break
            data, label = item
            Net.input(data)
            Net.run(run_time)
            output = Net.output.predict
            # output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
            if sim_name == 'pytorch':
                label = torch.tensor(label, device=device, dtype=torch.long)
            spike_output_test = [[]] * 10
            for o in range(assign_label.shape[0]):
                if spike_output_test[assign_label[o]] == []:
                    spike_output_test[assign_label[o]] = [output[:, o]]
                else:
                    spike_output_test[assign_label[o]].append(output[:, o])

            # test_output = [sum(spike_output_test[j])/len(spike_output_test[j]) for j in range(len(spike_output_test))]

            test_output = []
            for o in range(len(spike_output_test)):
                if spike_output_test[o] == []:
                    pass
                else:

                    test_output.append([sum(spike_output_test[o]) / len(spike_output_test[o])])
            predict_label = torch.argmax(torch.tensor(test_output, device=label.device))
            # predict_label = torch.argmax(torch.stack(test_output), 0)
            num_correct = (predict_label == label).sum().item()
            acc = num_correct / data.shape[0]
            eval_acc += acc

            # batch_loss = F.cross_entropy(output, label)
            # eval_loss += batch_loss.item()
            #
            # _, pred = output.max(1)
            # num_correct = (pred == label).sum().item()
            # acc = num_correct / data.shape[0]
            # eval_acc += acc

            # pbarTest.set_description_str("[loss:%f]Batch progress: " % batch_loss.item())
            pbarTest.update()


        print(assign_label)
        print(predict_label)
        print(label)
        a = eval_acc / len(test_loader)
        print(a)
        # for t in range(assign_label.shape[0]):
        #     if predict_label[t] == label[t]:
        #         eval_acc += 1
        #     else:
        #         pass

        # eval_losses.append(eval_loss / len(test_loader))
        # eval_acces.append(eval_acc / len(test_loader))
    pbarTest.close()
    print('epoch:{},Test Acc:{:.4f}'
          .format(epoch, eval_acc / len(test_loader)))
    print("")
