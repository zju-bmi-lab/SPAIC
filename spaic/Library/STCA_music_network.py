# -*- coding: utf-8 -*-
"""
Created on 2020/8/12
@project: SPAIC
@filename: STCA_music_network
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
建立网络的例子
"""
import spaic as sf
from torch import nn
# nn.Module
class STCANet(sf.Network):

    def __init__(self):
        super(STCANet, self).__init__()

        self.input = sf.Node(100)
        self.layer1 = sf.NeuronGroup(500, 'default')
        self.layer2 = sf.NeuronGroup(10, 'default')
        self.output = sf.Node(10)

        self.connect(self.input, self.layer1)
        self.link2 = sf.Connection(self.layer1, self.layer2, 'full')
        self.connect(self.layer2, self.output, 'one_hot')



if __name__ == "__main__":
    # a = sf.Network()
    a = STCANet()
    # a.layer1 = sf.Node(100)

    print(a.get_groups())
