# -*- coding: utf-8 -*-
"""
Created on 2020/8/11
@project: SNNFlow
@filename: Node
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
定义神经网络的输入输出接口
"""
from ..Network.Assembly import Assembly
import torch
import numpy as np

class Node(Assembly):
    '''Base class for input encoder and output decoders.
    '''
    _class_label = '<nod>'
    _is_terminal = True

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Node, self).__init__()
        # Node Parameters
        # self.dataloader = dataloader

        # if coding_time is not None:
        #     self._time = coding_time
        # else:
        #     raise ValueError('The coding time window can not be None. Please set coding time window through coding_time parameter. The value should be equal to the simulation time of one epoch.')

        self._dt = dt
        self._time = None
        self.coding_var_name = coding_var_name

        if coding_method == ('poisson', 'spike_counts', '...'):
            raise ValueError('Please specify the coding method such as poisson or spike_counts')
        else:
            self.coding_method = coding_method.lower()  # The name of coding method

        self.coding_var_name = coding_var_name

        if coding_method == 'null':
            self.is_encoded = True
        else:
            self.is_encoded = kwargs.get('is_encoded', False)

        # 单神经元多脉冲的语音数据集的shape包含脉冲时间以及发放神经元标签，所以不能通过np.prod(shape)获取num，最好还是外部输入num
        if num is None:
            raise ValueError('Please set the number of node')
        else:
            self.num = num

        if shape is None:
            if self.is_encoded:
                self.shape = (1, 1, num)
            else:
                self.shape = (1, num)
        else:
            self.shape = tuple([1] + list(shape))

        if node_type == ('excitatory', 'inhibitory', 'pyramidal', '...'):
            self.type = None
        else:
            self.type = node_type

        if coding_method == ('poisson', 'spike_counts', '...'):
            raise ValueError('Please specify the coding method such as poisson or spike_counts')
        else:
            self.coding_method = coding_method.lower()  # The name of coding method

        self.coding_var_name = coding_var_name

        # Coding parameters
        self.coding_param = kwargs
        self.dec_target = dec_target
        self.source = np.random.rand(*self.shape)
        #TODO:现在 Encoder, Decoder, Generator功能拆开了，是不是应该把这段放到Decoder类里了, 上边的coding_method也是同理
        if self.dec_target is not None:
            # The size of reward and predict is equal to batch_size
            self.reward = np.zeros((1,))
            self.predict = np.zeros((1,))

        # Parameters of initial operation
        self.index = 0
        self.records = []
        self._var_names = list()

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt

    @property
    def time(self):
        self._time = self._simulator.runtime
        return self._time

    @property
    def time_step(self):
        return int(self._simulator.runtime / self._dt)


    @time.setter
    def time(self, time):
        self._time = time

    def get_var_names(self):
        return self._var_names

    @staticmethod
    def register(name, coding_class):
        '''
        Register a coding class. Registered encoding or decoding classes can be referred to
        # via their name.
        Parameters
        ----------
        name : str
            A short name for the state updater (e.g. 'poisson', 'spike_counts')
        coding_class :
            The subclass of coding object, e.g. an 'PoissonEncoding', 'Spike_Counts'.
        '''

        raise NotImplementedError

    def torch_coding(self, source, target, device):
        '''

        Args:
            source (): It is input spike trains for encoding class and output spike trains for decoding class.
            target (): It is None  for encodoing class and labels for decoding class.
            device (): CPU or CUDA, this parameter is taken from simulator

        Returns:

        '''
        raise NotImplementedError

    def numpy_coding(self, source, target, device):
        raise NotImplementedError

    def tensorflow_coding(self, source, target, device):
        raise NotImplementedError

    def build(self, simulator):
        self._simulator = simulator
        raise NotImplementedError

    def __call__(self, data):
        self.source = data
        if isinstance(self, Encoder) or isinstance(self, Generator):
            if isinstance(data, np.ndarray):
                batch_size = data.shape[0]
            elif isinstance(data, list):
                batch_size = len(data)
            else:
                self.source = np.array(data)
                batch_size = 1

            if self._simulator is None:
                self.batch_size = batch_size
            else:
                self._simulator.set_batch_size(batch_size)
                self.batch_size = None

# ======================================================================================================================
# Encoders
# ======================================================================================================================
class Encoder(Node):
    '''
        Five encoding method are provided, as shown below (key: class):
        1. 'sstb': SigleSpikeToBinary,
        2. 'mstb': MultipleSpikeToBinary
        3. 'poisson': PoissonEncoding
        4. 'latency': Latency
        5. 'relative_latency': Relative_Latency
    '''
    _coding_subclasses = dict()
    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Encoder, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.batch_size = None
        coding_method = coding_method.lower()
        if coding_method == 'null':
            self.is_encoded = True
        else:
            self.is_encoded = kwargs.get('is_encoded', False)

    def __new__(cls, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        coding_method = coding_method.lower()
        if cls is not Encoder:
            return super().__new__(cls)
        elif coding_method in Encoder._coding_subclasses:
            return super().__new__(Encoder._coding_subclasses[coding_method])
        else:
            raise ValueError("No coding method: %s in Encoding classes" % coding_method)

    @staticmethod
    def register(name, coding_class):
        '''
        Register a coding class. Registered encoding or decoding classes can be referred to
        # via their name.
        Parameters
        ----------
        name : str
            A short name for the state updater (e.g. 'poisson', 'spike_counts')
        coding_class :
            The subclass of coding object, e.g. an 'PoissonEncoding', 'Spike_Counts'.
        '''

        # only deal with lower case names
        name = name.lower()
        if name in Encoder._coding_subclasses:
            raise ValueError(('A coding class with the name "%s" has already been registered') % name)

        if not issubclass(coding_class, Encoder):
            raise ValueError(
                ('Given class of type %s does not seem to be a valid encoding class.' % str(type(coding_class))))

        Encoder._coding_subclasses[name] = coding_class

    # initial operation: encoding input features into spike patterns
    def get_input(self):
        self.index = 0
        if self.sim_name == 'pytorch':
            spikes = self.torch_coding(self.source, self.device)
        else:
            spikes = self.numpy_coding(self.source, self.device)
        self.all_spikes = spikes
        return self.all_spikes

    # stand alone operation: get spike pattern in every time step
    def next_stage(self):
        self.index += 1
        return self.all_spikes[self.index-1]

    def build(self, simulator):
        self._simulator = simulator
        self.sim_name = simulator.simulator_name
        self.device = simulator.device

        if self.dt is None:
            self.dt = simulator.dt
        if self.batch_size is not None:
            self._simulator.set_batch_size(self.batch_size)

        if self.sim_name == 'pytorch':
            spikes = self.torch_coding(self.source, self.device)  # (time_step, batch_size, neuron_shape)
        else:
            spikes = self.numpy_coding(self.source, self.device)
        self.all_spikes = spikes

        if self.is_encoded:
            self.shape = spikes.shape
            singlestep_spikes = 0*spikes
        else:
            singlestep_spikes = 0*spikes[0]
            self.shape = singlestep_spikes.shape

        key = self.id + ':' + '{'+self.coding_var_name+'}'
        self._var_names.append(key)
        simulator.add_variable(key, self.shape, value=singlestep_spikes)
        simulator.register_initial(None, self.get_input, [])
        simulator.register_standalone(key, self.next_stage, [])


# ======================================================================================================================
# Decoders
#=======================================================================================================================
class Decoder(Node):
    '''
        Eleven decoding method are provided, as shown below (key: class):
        1. 'spike_counts': Spike_Counts
        2. 'first_spike': First_Spike
        3. 'time_spike_counts': TimeSpike_Counts
        4. 'time_softmax': Time_Softmax

        5. 'step_reward': EachStep_Reward
        6. 'global_reward': Global_Reward

        7. 'pop_rate_action': PopulationRate_Action
        8. 'softmax_action': Softmax_Action
        9. 'highest_spikes_action': Highest_Spikes_Action
        10. 'first_spike_action': First_Spike_Action
        11. 'random_action': Random_Action
    '''
    _coding_subclasses = dict()
    def __init__(self, shape=None, num=None, dec_target=None,  dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Decoder, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def __new__(cls, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        coding_method = coding_method.lower()
        if cls is not Decoder:
            return super().__new__(cls)
        elif coding_method in Decoder._coding_subclasses:
            return super().__new__(Decoder._coding_subclasses[coding_method])
        else:
            raise ValueError("No coding method: %s in Decoding classes" % coding_method)

    @staticmethod
    def register(name, coding_class):
        '''
        Register a coding class. Registered encoding or decoding classes can be referred to
        # via their name.
        Parameters
        ----------
        name : str
            A short name for the state updater (e.g. 'poisson', 'spike_counts')
        coding_class :
            The subclass of coding object, e.g. an 'PoissonEncoding', 'Spike_Counts'.
        '''

        # only deal with lower case names
        name = name.lower()
        if name in Decoder._coding_subclasses:
            raise ValueError(('A coding class with the name "%s" has already been registered') % name)

        if not issubclass(coding_class, Decoder):
            raise ValueError(
                ('Given class of type %s does not seem to be a valid decoding class.' % str(type(coding_class))))

        Decoder._coding_subclasses[name] = coding_class

    def init_state(self):
        self.index = 0

    # stand alone operation: decoding spike patterns. Predict can be predict labels or RL action
    def get_output(self, output):
        if (self.index % self.time_step) == 0:
            shape = list(output.shape)
            dec_shape = [self.time_step] + shape
            if type(output).__name__ == 'Tensor':
                self.records = torch.zeros(dec_shape, device=self.device)
            else:
                self.records = np.zeros(dec_shape)
        # self.records = self.records + output
        self.records[self.index, :] = output
        self.index += 1
        if self.index >= self.time_step:
            if self.sim_name == 'pytorch':
                self.predict = self.torch_coding(self.records, self.source, self.device)
            else:
                self.predict = self.numpy_coding(self.records, self.source, self.device)
        return 0

    # stand alone operation: get reward from output activities
    def get_global_reward(self, output):

        if (self.index % self.time_step) == 0:
            shape = list(output.shape)
            dec_shape = [self.time_step] + shape
            if type(output).__name__ == 'Tensor':
                self.records = torch.zeros(dec_shape, device=self.device)
            else:
                self.records = np.zeros(dec_shape)
        reward = torch.zeros(1, device=self.device)
        self.records[self.index, :] = output
        self.index += 1
        if self.index >= self.time_step:
            if self.sim_name == 'pytorch':
                reward = self.torch_coding(self.records, self.source, self.device)
                self.predict = self.records.sum(0)  # (batch_size, label_num)
                self.reward = reward
            else:
                reward = self.numpy_coding(self.records, self.source, self.device)
        return reward

    def get_step_reward(self, output):
        if (self.index % self.time_step) == 0:
            if type(output).__name__ == 'Tensor':
                self.reward = torch.zeros(self.shape, device=self.device)
                self.predict = torch.zeros(output.shape, device=self.device)
            else:
                self.reward = np.zeros(*self.shape)
                self.predict = np.zeros(*self.shape)
        self.index += 1
        if self.sim_name == 'pytorch':
            reward = self.torch_coding(output, self.source, self.device)
            self.reward = reward
            self.predict += output  # (batch_size, label_num)
        else:
            reward = self.numpy_coding(output, self.source, self.device)
        return reward

    def build(self, simulator):
        self._simulator = simulator
        self.sim_name = simulator.simulator_name
        self.device = simulator.device
        if self.dt is None:
            self.dt = simulator.dt
        # self.time_step = int(self.time / self.dt)


        output_name = self.dec_target.id + ':' + '{'+self.coding_var_name+'}'
        simulator.register_initial(None, self.init_state, [])
        if self.coding_method == 'step_reward':
            reward_name = 'Output_Reward'
            simulator.add_variable(reward_name, (1, ), value=self.reward)
            simulator.register_standalone(reward_name, self.get_step_reward, [output_name])
        elif self.coding_method == 'global_reward':
            # reward_name = self.dec_target.id + ':' + '{Reward}'
            reward_name = 'Output_Reward'
            simulator.add_variable(reward_name, (1, ), value=self.reward)
            simulator.register_standalone(reward_name, self.get_global_reward, [output_name])
        else:
            # register_standalone(self, output_name: str, function, input_names: list):
            simulator.register_standalone(None, self.get_output, [output_name])


class Generator(Node):
    '''
        Two generator method are provided, as shown below (key: class):
        1. 'poisson_generator': Poisson_Generator,
        2. 'cosine_generator': Cosine_Generator
    '''
    _coding_subclasses = dict()
    def  __init__(self, shape=None, num=None, dec_target=None,  dt=None,
                  coding_method=('poisson_generator', 'cc_generator', '...'),
                  coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Generator, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

    def __new__(cls, shape=None, num=None, dec_target=None, dt=None,
                coding_method=('poisson_generator', 'cc_generator', '...'),
                coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        if cls is not Generator:
            return super().__new__(cls)
        elif coding_method in Generator._coding_subclasses:
            return super().__new__(Generator._coding_subclasses[coding_method])
        else:
            raise ValueError("No coding method: %s in Generator classes" % coding_method)

    @staticmethod
    def register(name, coding_class):
        '''
        Register a coding class. Registered encoding or decoding classes can be referred to
        # via their name.
        Parameters
        ----------
        name : str
            A short name for the state updater (e.g. 'poisson', 'spike_counts')
        coding_class :
            The subclass of coding object, e.g. an 'PoissonEncoding', 'Spike_Counts'.
        '''

        # only deal with lower case names
        name = name.lower()
        if name in Generator._coding_subclasses:
            raise ValueError(('A coding class with the name "%s" has already been registered') % name)

        if not issubclass(coding_class, Generator):
            raise ValueError(
                ('Given class of type %s does not seem to be a valid generator class.' % str(type(coding_class))))

        Generator._coding_subclasses[name] = coding_class

    # initial operation: encoding input features into spike patterns
    def get_input(self):
        self.index = 0
        if self.sim_name == 'pytorch':
            spikes = self.torch_coding(self.source, self.device)
        else:
            spikes = self.numpy_coding(self.source, self.device)
        self.all_spikes = spikes
        return self.all_spikes

    # stand alone operation: get spike pattern in every time step
    def next_stage(self):
        self.index += 1
        return self.all_spikes[self.index-1]

    def build(self, simulator):
        self._simulator = simulator
        self.sim_name = simulator.simulator_name
        self.device = simulator.device
        if self.dt is None:
            self.dt = simulator.dt

        if self.sim_name == 'pytorch':
            singlestep_spikes = torch.zeros(self.shape, device=self.device)
        else:
            singlestep_spikes = np.zeros(self.shape)

        if self.dec_target is None:
            key = self.id + ':' + '{'+self.coding_var_name+'}'
            simulator.add_variable(key, self.shape, value=singlestep_spikes)
        else:
            key = self.dec_target.id + ':' + '{'+self.coding_var_name+'}'
        self._var_names.append(key)
        simulator.register_initial(None, self.get_input, [])
        simulator.register_standalone(key, self.next_stage, [])
