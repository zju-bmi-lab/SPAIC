# -*- coding: utf-8 -*-
"""
Created on 2021/4/1
@project: SPAIC
@filename: DelayQueue
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
定义网络传递延迟的基本组件
"""
import torch
from abc import abstractmethod


class DelayQueue(object):
    _delayqueue_subclasses = dict()

    def __init__(self, var_name=None, max_len=None, backend=None):
        super(DelayQueue, self).__init__()
        self.max_len = max_len
        self.dt = backend.dt
        self.backend = backend
        self.var_name = var_name

    def __new__(cls, var_name=None, max_len=None, backend=None):

        if cls is not DelayQueue:
            return super().__new__(cls)
        if backend.backend_name in cls._delayqueue_subclasses:
            return cls._delayqueue_subclasses[backend.backend_name](var_name, max_len, backend)
        else:
            raise ValueError("No DelayQueue type for : %s backend" % backend.backend_name)

    @staticmethod
    def register(name, deque_class):
        '''
        Register a DelayQueue class. Registered DelayQueue classes can be referred to
        # via their name.

        Parameters
        ----------
        name : str
            A short name for the backend (e.g. `'pytorch'`)
        deque_class : `ConnectionModel`
            The subclass of Delaydeque object.
        '''

        # only deal with lower case names -- we don't want to have 'LIF' and
        # 'lif', for example
        name = name.lower()
        if name in DelayQueue._delayqueue_subclasses:
            raise ValueError(('A delayqueue class with the name "%s" has already been registered') % name)

        if not issubclass(deque_class, DelayQueue):
            raise ValueError(
                ('Given model of type %s does not seem to be a valid NeuronModel.' % str(type(deque_class))))

        DelayQueue._delayqueue_subclasses[name] = deque_class

    @abstractmethod
    def push(self, input):
        NotImplementedError()

    @abstractmethod
    def select(self, delay):
        NotImplementedError()

    @abstractmethod
    def initial(self, var):
        NotImplementedError()


class TorchDelayQueue(DelayQueue):

    def __init__(self, var_name, max_len, backend):
        super(TorchDelayQueue, self).__init__(var_name, max_len, backend)
        self.device = backend.device0
        self._backend = backend

    def initial(self, var=None, batch_size=1):
        self.count = 0
        # self.queue = None #torch.zeros(queue_shape, device=self.device)
        self.var_shape = list(var.shape)
        self.var_shape[0] = batch_size
        queue_shape = [self.max_len] + self.var_shape
        self.queue = torch.zeros(queue_shape, device=self.device, dtype=var.dtype)

    def transform_delay_output(self, input, delay):
        if input.dim() == 2:
            output = input.unsqueeze(-1).expand(-1, -1, delay.shape[0])
        else:
            output = input.unsqueeze(-1).expand(-1, -1, -1, delay.shape[0])
        return output

    def push(self, input):
        # if self.queue is None:
        #     self.var_shape = input.shape
        #     queue_shape = [self.max_len] + list(self.var_shape)
        #     self.queue = torch.zeros(queue_shape, device=self.device)

        if input.requires_grad:
            self.queue = torch.cat([self.queue[:self.count], input.unsqueeze(0), self.queue[self.count + 1:]], dim=0)
        else:
            self.queue[self.count, ...] = input

        self.count += 1
        self.count = self.count % self.max_len

        return input

    def select(self, delay: torch.Tensor):
        # Only for one-dim neurongroup for now
        delay = delay.clip(0, self.max_len * self.dt)
        # queue.shape = (delay_len, batch_size, pre_num)
        if self.queue.dim() == delay.dim() + 1:
            # delay.shape = (post_num, pre_num)
            delay = delay.unsqueeze(1).expand(-1, self.var_shape[0], -1)  # (post_num, batch)
            ind = (delay / self.dt).long()
            ind = torch.fmod(self.max_len - ind + self.count, self.max_len)
            output = torch.gather(self.queue, 0, ind).permute(1, 0, 2)

        elif self.queue.dim() == delay.dim() + 2:
            delay = delay.unsqueeze(1).unsqueeze(1).expand(-1, self.var_shape[0], 2, -1)
            ind = (delay / self.dt).long()
            ind = torch.fmod(self.max_len - ind + self.count + 1, self.max_len)
            output = torch.gather(self.queue, 0, ind)
            output[:, :, 1, :] -= (delay - ind * self.dt)[:, :, 1, :] / 10.0
            output = output.permute(1, 0, 2, 3)
        elif self.queue.dim() == delay.dim():
            delay = delay.expand(-1, self.var_shape[0], -1)
            ind = (delay / self.dt).long()
            ind = torch.fmod(self.max_len - ind + self.count, self.max_len)
            output = torch.gather(self.queue, 0, ind).permute(1, 0, 2)

        return output


DelayQueue.register('pytorch', TorchDelayQueue)


def test_queue():
    class Backend:
        dt = 0.1
        device = 'cpu'

    backend = Backend()
    # with torch.autograd.set_detect_anomaly(True):
    Queue = TorchDelayQueue('test', 60, backend)
    States = [torch.randn((100, 500)) for ii in range(10)]
    delay = 10.0 * torch.randn(200, 500)
    delay.requires_grad = True
    Queue.initial()
    output = []

    for ii in range(10):
        States[ii].requires_grad = True
        Queue.push(States[ii])
        output.append(Queue.select(delay))

    loss = torch.sum(torch.cat(output))
    loss.backward()
    # print(delay.grad)
    print(States[0].grad)

# test_queue()
