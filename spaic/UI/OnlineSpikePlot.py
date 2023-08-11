# -*- coding: utf-8 -*-
"""
Created on 2021/8/9
@project: main.py
@filename: OnlineSpikePlot
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
在训练过程中实现放电脉冲栅图更新
"""
import spaic
import matplotlib.pyplot as plt
from multiprocessing import Queue, Process, Lock
from matplotlib.animation import FuncAnimation


class SpikePlot(spaic.BaseModule):

    def __init__(self, spike_monitors = [], plot_interval=1):
        super(SpikePlot, self).__init__()
        self.spike_monitors = spike_monitors
        self.interval = plot_interval
        self.draw_proc = None
        self.queue = Queue(maxsize=10)
        self.mutex = Lock()


