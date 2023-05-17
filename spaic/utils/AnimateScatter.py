# -*- coding: utf-8 -*-
"""
Created on 2022/11/11
@project: SPAIC
@filename: AnimateScatter
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description: 
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class AnimatedScatter(object):

    def __init__(self, positions, values):
        self.positions = positions
        self.values = values
        self.fig, self.ax = plt.subplots()
        self.ani = animation.FuncAnimation()

    def setup_plot(self):
        x, y, c = next(self.stream)
        self.scat = self.ax.scatter(x, y, c=c, animation=True)
        return self.scat

    def data_stream(self):
        pass

    def update(self, i):
        posi, color = next(self.stream)
        self.scat.set_offsets(posi)
        self.scat.set_array(color)
        return self.scat


    def show(self):
        plt.show()