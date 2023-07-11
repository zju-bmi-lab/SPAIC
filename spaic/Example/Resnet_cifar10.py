# encoding: utf-8
"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: Resnet18_Cifar10.py
@time:2022/1/28 14:32
@description:
"""
import spaic
import torch
from torch import nn
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import csv

# 设备设置
if torch.cuda.is_available():
    device = 'cuda'
    print('cuda')
else:
    device = 'cpu'

parser = argparse.ArgumentParser()

# Settings.
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

parser.add_argument("--model", type=str, default="ifsoftreset")
parser.add_argument("--v_th", type=float, default=1.0)
parser.add_argument("--run_time", type=float, default=15)
parser.add_argument("--dt", type=float, default=1)
parser.add_argument("--lr", type=float, default=0.0025)
parser.add_argument("--maxepoch", type=int, default=300)

args = parser.parse_args()
time_step = int(args.run_time/args.dt)
node_num = 32*32
bat_size = 32
simulator = spaic.Torch_Backend(device)
sim_name = simulator.backend_name
sim_name = sim_name.lower()

run_time = args.run_time
simulator.dt = args.dt

# 准备数据集并预处理
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
root = '../../../datasets/CIFAR10'
trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform) #训练数据集
train_loader = torch.utils.data.DataLoader(trainset, batch_size=bat_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=bat_size, shuffle=False)


# The parameters of network structure
datachanel = 3
inplanes = 64
expansion = 2

figure_size = [32, 32]

inchannel1 = inplanes
outchannel1 = inchannel1
feature_map1 = [32, 32]
stride1 = 1

inchannel2 = outchannel1
outchannel2 = inchannel2*expansion
feature_map2 = [16, 16]
stride2 = 2

inchannel3 = outchannel2
outchannel3 = inchannel3*expansion
feature_map3 = [8, 8]
stride3 = 2

inchannel4 = outchannel3
outchannel4 = inchannel4*expansion
feature_map4 = [4, 4]
stride4 = 2
label_num = 10


class postprocess_module(nn.Module):
    def __init__(self, output_size):
        super(postprocess_module, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=output_size)
        self.linear = nn.Linear(in_features=outchannel4, out_features=label_num, bias=True)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=1):
        super(conv, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel)
        )
        for m in self.conv2d.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv2d(x)
        return x

class Residual_block(spaic.Assembly):
    expansion = 2
    def __init__(self, input_obj, feature_shape, inchannel, outchannel, stride=1):
        super(Residual_block, self).__init__()
        self.layer1 = spaic.NeuronGroup(shape=[outchannel, *feature_shape], model=args.model, v_th=args.v_th)
        self.layer2 = spaic.NeuronGroup(shape=[outchannel, *feature_shape], model=args.model, v_th=args.v_th)

        self.input_layer1_con = spaic.Module(conv(in_channel=inchannel, out_channel=outchannel, kernel_size=3, stride=stride, padding=1),
                                               input_targets=input_obj, input_var_names='O[updated]',
                                               output_targets=self.layer1, output_var_names='Isyn')

        self.layer1_layer2_con = spaic.Module(conv(in_channel=outchannel, out_channel=outchannel, kernel_size=3, stride=1, padding=1),  # 3*3
                                                input_targets=self.layer1, input_var_names='O[updated]',
                                                output_targets=self.layer2, output_var_names='Isyn')

        if stride != 1 or inchannel != outchannel:
            self.shortcut = spaic.Module(
                conv(in_channel=inchannel, out_channel=outchannel, kernel_size=1, stride=stride, padding=0),
                input_targets=input_obj, input_var_names='O[updated]',
                output_targets=self.layer2, output_var_names='Isyn')   # 1*1
        else:
            self.input_layer2_con = spaic.Connection(input_obj, self.layer2, link_type='null', syn_type='directpass')


class SpikingResNet(spaic.Network):

    def __init__(self):
        super(SpikingResNet, self).__init__()
        self.input = spaic.Encoder((3, 32, 32), coding_method='null')

        self.preprocess_input = spaic.NeuronGroup(shape=[inplanes, *figure_size], model=args.model)

        self.preprocess = spaic.Module(conv(in_channel=datachanel, out_channel=inplanes, kernel_size=3, stride=1, padding=1),
                                         input_targets=self.input, input_var_names='O[updated]',
                                         output_targets=self.preprocess_input, output_var_names='Isyn')

        self.make_resnet_block(input_obj=self.preprocess_input, feature_map=feature_map1,
                                             inchannel=inchannel1, outchannel=outchannel1, stride=stride1, name_blocks=['block11', 'block12'])   # 第一个block

        self.make_resnet_block(input_obj=self.block12.layer2, feature_map=feature_map2,
                               inchannel=inchannel2, outchannel=outchannel2, stride=stride2,
                               name_blocks=['block21', 'block22'])  # 第二个block

        self.make_resnet_block(input_obj=self.block22.layer2, feature_map=feature_map3,
                               inchannel=inchannel3, outchannel=outchannel3, stride=stride3,
                               name_blocks=['block31', 'block32'])  # 第三个block

        self.make_resnet_block(input_obj=self.block32.layer2, feature_map=feature_map4,
                               inchannel=inchannel4, outchannel=outchannel4, stride=stride4,
                               name_blocks=['block41', 'block42'])  # 第四个block


        self.out_layer = spaic.NeuronGroup(label_num, model='null')

        self.flatten = spaic.Module(postprocess_module(output_size=(1, 1)), input_targets=self.block42.layer2,
                                                                    input_var_names='O[updated]', output_targets=self.out_layer, output_var_names='Isyn')

        self.output = spaic.Decoder(num=label_num, dec_target=self.out_layer, coding_method='spike_counts')
        # self.mon_v = snnflow.StateMonitor(self.block42.layer2, 'V')

        self.learner = spaic.Learner(trainable=self, algorithm='sbp', alpha=2.0)
        self.learner.set_optimizer('Adam', optim_lr=args.lr)

    def make_resnet_block(self, input_obj, feature_map, inchannel, outchannel, stride, name_blocks: list):
        self.add_assembly(name=name_blocks[0],
                          assembly=Residual_block(input_obj=input_obj, feature_shape=feature_map,
                                                  inchannel=inchannel, outchannel=outchannel, stride=stride))

        self.add_assembly(name=name_blocks[1], assembly=Residual_block(input_obj=self._groups[name_blocks[0]].layer2,
                                                                                      feature_shape=feature_map,
                                                                                      inchannel=outchannel,
                                                                                      outchannel=outchannel, stride=1))


Net = SpikingResNet()
Net.set_backend(simulator)
Net.build(strategy=1)
print("Start running")
num_correct = 0
num_sample = 0
maxacc = 0
import time
flag = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
network_param_name = f'cifar10_resnet18_{args.model}_{args.run_time}_{args.v_th}_{args.lr}_best_weight_'+flag
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Net.learner.optim, T_max=args.maxepoch / 16)
csv_name = f"resnet18_{args.model}_{args.run_time}_{args.v_th}_{args.lr}_"+flag
with open(csv_name, "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
for epoch in range(args.maxepoch):


    # 训练阶段
    print("Start training")
    train_loss = 0
    train_acc = 0
    pbar = tqdm(total=len(train_loader))
    for i, item in enumerate(train_loader):
        # 前向传播
        data, label = item
        data = np.expand_dims(data, 1).repeat(time_step, axis=1)
        label = torch.tensor(label, device=device)

        Net.input(data)
        Net.output(label)
        Net.run(run_time)
        # output_v = Net.mon_v.values
        # print(output_v)

        output = Net.output.predict
        batch_loss = F.cross_entropy(output, label)

        # 反向传播
        Net.learner.optim_zero_grad()
        batch_loss.backward(retain_graph=False)
        Net.learner.optim_step()
        scheduler.step()

        # 记录误差
        train_loss += batch_loss.item()
        predict_labels = torch.argmax(output, 1)
        num_correct = (predict_labels == label).sum().item()  # 记录标签正确的个数
        acc = num_correct / data.shape[0]
        train_acc += acc
        pbar.set_description_str("[loss:%f]Batch progress: " % batch_loss.item())
        pbar.update()
    pbar.close()
    print('epoch:{},Train Loss:{:.4f},Train Acc:{:.4f}'.format(epoch, train_loss / len(train_loader), train_acc / len(train_loader)))

    # 测试阶段
    eval_loss = 0
    eval_acc = 0
    print("Start testing")
    pbarTest = tqdm(total=len(test_loader))
    with torch.no_grad():
        for i, item in enumerate(test_loader):
            data, label = item
            data = np.expand_dims(data, 1).repeat(time_step, axis=1)
            Net.input(data)
            Net.run(run_time)


            output = Net.output.predict
            label = torch.tensor(label, device=device)
            batch_loss = F.cross_entropy(output, label)
            eval_loss += batch_loss.item()
            _, pred = output.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / data.shape[0]
            eval_acc += acc
            pbarTest.set_description_str("[loss:%f]Batch progress: " % batch_loss.item())
            pbarTest.update()
        if eval_acc>maxacc:
            maxacc=eval_acc
            Net.save_state(network_param_name)

    pbarTest.close()
    print('epoch:{},Test Loss:{:.4f},Test Acc:{:.4f}'.format(epoch, eval_loss / len(test_loader), eval_acc / len(test_loader)))
    with open(csv_name, "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [epoch, "{:.4f}".format(train_loss / len(train_loader)), "{:.4f}".format(train_acc / len(train_loader)),
             "{:.4f}".format(eval_loss / len(test_loader)), "{:.4f}".format(eval_acc / len(test_loader))])