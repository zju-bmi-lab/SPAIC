import os
import psutil
import sys
import torch

def get_cpu_mem():
    '''
    Get current process memory size in MiB.
    '''
    memory = round(psutil.Process(os.getpid()).memory_info().rss / 1048576, 4)
    return memory

def get_tens_mem(tensor):
    '''
    Get pytorch tensor memory size in MiB. 
    '''
    if torch.is_tensor(tensor):
        a = tensor.element_size() * tensor.nelement()
    else:
        a = 0
    tensor_memory = round(a / 1048576, 4)
    return tensor_memory

def get_object_size(object):
    '''
    Get object size in MiB.
    '''
    size = round(sys.getsizeof(object) / 1048576, 4)
    return size