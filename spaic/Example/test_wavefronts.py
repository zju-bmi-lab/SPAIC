# -*- coding: utf-8 -*-
"""
Created on 2022/1/6
@project: SPAIC
@filename: test_wavefronts
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description: 
"""

import spaic
import numpy as np
from matplotlib.pyplot import *




# 定义权重位置
weight1 = np.zeros((1, 1, 29, 29))
weight2 = np.zeros((1, 1, 29, 29))
a = 2.5
b = 4.0
w = 10.0
for x in range(29):
    for y in range(29):
        dist = ((x-14.0)**2 + (y-14.0)**2)**0.5
        weight1[0,0,x,y] = w*np.exp(-dist/a)/a
        weight2[0,0,x,y] = w*np.exp(-dist/b)/b

# imshow(weight2[0,0,...])
# show()

#
# # 基本距离权重关系
# r = np.arange(-14.0, 14.0, 1.0)
# a = 2.0
# b = 4.0
# y = np.exp(-np.abs(r)/a)/a - np.exp(-np.abs(r)/b)/b
# plot(y)
# show()

CANNNet = spaic.Network()
with CANNNet:
    input = spaic.Generator(num=200*200, shape=(1, 200, 200), coding_method='poisson_generator')
    # input = spaic.Encoder(num=200*200, shape=(1, 200, 200), coding_method='poisson_generator')
    exc_layer = spaic.NeuronGroup(num=200 * 200, shape=(1, 200, 200),  model='meanfield', tau=1.0)
    inh_layer = spaic.NeuronGroup(num=200 * 200, shape=(1, 200, 200), model='meanfield', tau=2.0)
    inp_link = spaic.Connection(input, exc_layer, link_type='conv', in_channels=1, out_channels=1, kernel_size=(29,29),
                                syn_type=['conv'], padding=14, weight=weight1, post_var_name='Iext')
    ee_link = spaic.Connection(exc_layer, exc_layer, link_type='conv', in_channels=1, out_channels=1, kernel_size=(29,29),
                               syn_type=['conv'], padding=14, weight=weight2, post_var_name='WgtSum')
    ei_link = spaic.Connection(exc_layer, inh_layer, link_type='conv', in_channels=1, out_channels=1,
                                 kernel_size=(29, 29), syn_type=['conv'],  padding=14, weight=weight2,
                                 post_var_name='WgtSum')
    ie_link = spaic.Connection(inh_layer, exc_layer, link_type='conv', in_channels=1, out_channels=1,
                                 kernel_size=(29, 29), syn_type=['conv'],  padding=14, weight=-2.0*weight1,
                                 post_var_name='WgtSum')
    ii_link = spaic.Connection(inh_layer, inh_layer, link_type='conv', in_channels=1, out_channels=1,
                                 kernel_size=(29, 29),syn_type=['conv'], padding=14, weight=-weight1,
                                 post_var_name='WgtSum')

    om = spaic.StateMonitor(exc_layer, 'O')



CANNNet.set_backend('pytorch')
CANNNet.set_backend_dt(0.2)

# ion()
inp = np.zeros((1,1,200, 200))
inp[0,0, 100, 100] = 1.0
for kk in range(25):
    # # if kk == 1:
    # inp[0, 0, 100, 100] = 0.0
    input(inp)
    CANNNet.run_continue(10.0)
    out = om.values
    # imshow(CANNNet._backend._variables['CANNNet<net>_inter_link<con>:CANNNet<net>_layer<neg><-CANNNet<net>_layer<neg>:{weight}'].detach().numpy())
    # show()
    om.init_record()
    timelen = out.shape[-1]
    print(kk)
    for ii in range(timelen):
        # fig = figure(1)
        clf()
        imshow(out[0,0,:,:,ii])
        # show()
        draw()
        pause(0.01)


