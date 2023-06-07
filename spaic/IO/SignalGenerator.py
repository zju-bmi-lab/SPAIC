# -*- coding: utf-8 -*-
"""
Created on 2020/10/23
@project: SPAIC
@filename: SignalGenerator
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description: 
"""
from .Pipeline import Pipline


class SigGenerator(Pipline):
    '''
    Generate signals such as sin/cos wave, random process...
    '''

    def __init__(self):
        super(SigGenerator, self).__init__()
        pass
