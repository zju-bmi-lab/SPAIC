# -*- coding: utf-8 -*-
"""
Created on 2020/8/11
@project: SPAIC
@filename: Node
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
定义神经网络的输入输出接口
"""
from ..Network.Assembly import Assembly
import torch
import numpy as np
from ..Backend.Backend import Op


class Node(Assembly):
    '''Base class for input encoder and output decoders.
    '''
    _class_label = '<nod>'
    _is_terminal = True

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Node, self).__init__()

        self._dt = dt
        self._time = kwargs.get('time', None)
        self.coding_var_name = coding_var_name
        position = kwargs.get('position', None)
        if position == None:
            self.position = []
        else:
            position = np.array(position)
            assert position.shape[0] == num, " Neuron_position not equal to neuron number"
            self.position = position

        if coding_method == ('poisson', 'spike_counts', '...'):
            raise ValueError('Please specify the coding method such as poisson or spike_counts')
        else:
            self.coding_method = coding_method.lower()  # The name of coding method

        self.coding_var_name = coding_var_name

        if coding_method.lower() == 'null':
            self.is_encoded = True
        else:
            self.is_encoded = kwargs.get('is_encoded', False)

        # 单神经元多脉冲的语音数据集的shape包含脉冲时间以及发放神经元标签，所以不能通过np.prod(shape)获取num，最好还是外部输入num
        assert num is not None or shape is not None, "One of the shape and number must not be None"
        if num is None:
            if coding_method == 'mstb' or coding_method == 'sstb':
                raise ValueError('Please set the number of node')
            self.num = np.prod(shape)
        else:
            self.num = num

        self.num = int(self.num)  # 统一数据格式为Python内置格式

        if shape is None:
            if self.is_encoded:
                self.shape = [1, 1, self.num]
            else:
                self.shape = [1, self.num]
        else:
            if coding_method == 'mstb' or coding_method == 'sstb':
                self.shape = (1, self.num)
            elif self.is_encoded:
                self.shape = [1, 1] + list(shape)
            else:
                self.shape = [1] + list(shape)

        if node_type == ('excitatory', 'inhibitory', 'pyramidal', '...'):
            self.type = []
        elif isinstance(node_type, list):
            self.type = node_type
        else:
            self.type = [node_type]



        # Coding parameters
        self.coding_param = kwargs
        self.dec_target = dec_target
        self.source = np.random.rand(*self.shape)
        #TODO:现在 Encoder, Decoder, Generator功能拆开了，是不是应该把这段放到Decoder类里了, 上边的coding_method也是同理
        if self.dec_target is not None:
            # The size of predict, reward and action is equal to batch_size
            self.predict = np.zeros((1,))
            # self.reward = np.zeros((1,))
            # self.action = np.zeros((1,))

        # Parameters of initial operation
        self.index = 0
        self.records = []
        self._var_names = list()

    def init_state(self):
        self.index = 0

    @property
    def dt(self):
        if self._dt is None and self._backend is not None:
            return self._backend.dt
        else:
            return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt

    @property
    def time(self):
        if self._time is None and self._backend is not None:
            return self._backend.runtime
        else:
            return self._time

    @property
    def time_step(self):
        return int(np.ceil(self.time / self.dt))


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

    def torch_coding(self, source: torch.Tensor, target: torch.Tensor , device: str) -> torch.Tensor:
        '''
        Args:
            source : It is input spike trains for encoding class and output spike trains for decoding class.
            target : It is None  for encodoing class and labels for decoding class.
            device : CPU or CUDA, this parameter is taken from backend
        Returns:
        '''

        raise NotImplementedError

    def numpy_coding(self, source, target, device):
        raise NotImplementedError

    def tensorflow_coding(self, source, target, device):
        raise NotImplementedError

    def build(self, backend):
        self._backend = backend
        self.data_type = backend.data_type

    def __call__(self, data=None):

        if isinstance(self, Encoder) or isinstance(self, Generator) or isinstance(self, Reward):
            if isinstance(data, np.ndarray):
                self.source = data
                batch_size = data.shape[0]
            elif isinstance(data, torch.Tensor):
                self.source = data
                batch_size = data.shape[0]
            elif isinstance(data, list) and self.coding_method=="mstb":
                self.source = data
                batch_size = len(self.source)
            elif hasattr(data, '__iter__'):
                self.source = np.array(data)
                batch_size = self.source.shape[0]
            else:
                self.source = np.array([data])
                batch_size = 1

            if self._backend is None:
                self.batch_size = batch_size
            else:
                self._backend.set_batch_size(batch_size)
                self.batch_size = None

            self.new_input = True

        elif isinstance(self, Decoder):
            if isinstance(data, np.ndarray):
                self.source = data
            elif isinstance(data, torch.Tensor):
                self.source = data
            elif hasattr(data, '__iter__'):
                self.source = np.array(data)
            else:
                self.source = np.array([data])

            return self.predict


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
    _node_sub_class = '<encoder>'
    _coding_subclasses = dict()
    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Encoder, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.batch_size = None
        self.new_input = True
        # coding_method = coding_method.lower()
        # if coding_method == 'null':
        #     self.is_encoded = True
        # else:
        #     self.is_encoded = kwargs.get('is_encoded', False)

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

    def init_state(self):
        self.index = 0

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
        # For hardware applications, call next_stage at each time step to get spike data of the current time step.
        if self.new_input:
            self.get_input()
            self.new_input = False

        self.index += 1

        return self.all_spikes[self.index-1]

    def reset(self):
        # Called at the start of each epoch
        self.init_state()

    def build(self, backend):
        self._backend = backend
        self.sim_name = backend.backend_name
        self.device = backend.device0

        # if self.dt is None:
        #     self.dt = backend.dt
        if self.batch_size is not None:
            self._backend.set_batch_size(self.batch_size)

        # if self.sim_name == 'pytorch':
        #     spikes = self.torch_coding(self.source, self.device)  # (time_step, batch_size, shape)
        # else:
        #     spikes = self.numpy_coding(self.source, self.device)
        # self.all_spikes = spikes

        if self.is_encoded:
            shape = self.shape[1:]
        else:
            shape = self.shape
        # self.shape = spikes[0].shape

        key = self.id + ':' + '{'+self.coding_var_name+'}'
        self.variable_to_backend(key, shape, value=0)
        self.init_op_to_backend(None, self.init_state, [])
        backend.register_standalone(Op(key, self.next_stage, [], owner=self))


# ======================================================================================================================
# Decoders
# ======================================================================================================================

class Decoder(Node):
    '''
        Five decoding method are provided, as shown below (key: class):
        1. 'spike_counts': Spike_Counts
        2. 'first_spike': First_Spike
        3. 'time_spike_counts': TimeSpike_Counts
        4. 'time_softmax': Time_Softmax
        5. 'final_step_voltage': Final_Step_Voltage
    '''
    _node_sub_class = '<decoder>'
    _coding_subclasses = dict()
    def __init__(self, num=None, dec_target=None,  dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Decoder, self).__init__(None, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        assert num == dec_target.num, ('The num of Decoder is not consistent with num of NeuronGroup')

    def __new__(cls, num=None, dec_target=None, dt=None, coding_method='spike_counts',
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
                self.records = torch.zeros(dec_shape, device=self.device, dtype=output.dtype)
            else:
                self.records = np.zeros(dec_shape)
            self.index = 0

        self.records[self.index % self.time_step, :] = output
        self.index += 1
        if self.index >= self.time_step:
            if self.sim_name == 'pytorch':
                self.predict = self.torch_coding(self.records, self.source, self.device)
            else:
                self.predict = self.numpy_coding(self.records, self.source, self.device)
        return 0

    def reset(self):
        # Called at the start of each epoch
        self.init_state()

    def build(self, backend):
        self._backend = backend
        self.sim_name = backend.backend_name
        self.device = backend.device0
        # if self.dt is None:
        #     self.dt = backend.dt

        output_name = self.dec_target.id + ':' + '{'+self.coding_var_name+'}'
        self.init_op_to_backend(None, self.init_state, [])
        backend.register_standalone(Op(None, self.get_output, [output_name], owner=self))

# ======================================================================================================================
# Rewards
# ======================================================================================================================

class Reward(Node):
    '''
        Three reward method are provided, as shown below (key: class):
        1. 'global_reward', Global_Reward
        2. 'xor_reward': XOR_Reward
        3. 'da_reward': DA_Reward
        4. 'environment_reward': Environment_Reward
    '''
    _node_sub_class = '<reward>'
    _coding_subclasses = dict()
    def __init__(self, shape=None, num=None, dec_target=None,  dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Reward, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.dec_sample_step = kwargs.get('dec_sample_step', 1)
        self.reward_shape = kwargs.get('reward_shape', (1, ))

    def __new__(cls, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        coding_method = coding_method.lower()
        if cls is not Reward:
            return super().__new__(cls)
        elif coding_method in Reward._coding_subclasses:
            return super().__new__(Reward._coding_subclasses[coding_method])
        else:
            raise ValueError("No reward method: %s in Reward classes" % coding_method)

    @staticmethod
    def register(name, coding_class):
        '''
        Register a Reward class. Registered reward classes can be referred to
        # via their name.
        Parameters
        ----------
        name : str
            A short name for the state updater (e.g. 'step_reward')
        coding_class :
            The subclass of coding object, e.g. an 'Step_Reward'.
        '''

        # only deal with lower case names
        name = name.lower()
        if name in Reward._coding_subclasses:
            raise ValueError(('A reward class with the name "%s" has already been registered') % name)

        if not issubclass(coding_class, Reward):
            raise ValueError(
                ('Given class of type %s does not seem to be a valid reward class.' % str(type(coding_class))))

        Reward._coding_subclasses[name] = coding_class

    def init_state(self):
        self.index = 0

    # stand alone operation: get reward from output activities
    def get_reward(self, output=np.empty(0)):
        self.device = self._backend.device0
        if (self.index % self.dec_sample_step) == 0:
            self.index = 0
            shape = list(output.shape)
            dec_shape = [self.dec_sample_step] + shape
            if type(output).__name__ == 'Tensor':
                self.records = torch.zeros(dec_shape, device=self.device)
            else:
                self.records = np.zeros(dec_shape)
        reward = torch.zeros(self.reward_shape, device=self.device)
        self.records[self.index, :] = output
        self.index += 1
        if self.index >= self.dec_sample_step:
            if self.sim_name == 'pytorch':
                reward = self.torch_coding(self.records, self.source, self.device)
                self.reward = reward
            else:
                reward = self.numpy_coding(self.records, self.source, self.device)
        return reward

    def build(self, backend):
        self._backend = backend
        self.sim_name = backend.backend_name
        self.data_type = backend.data_type
        self.device = backend.device0
        # if self.dt is None:
        #     self.dt = backend.dt
        self.init_op_to_backend(None, self.init_state, [])
        reward_name = 'Output_Reward'
        self.variable_to_backend(reward_name, self.reward_shape, value=0.0)  # shape还是要让具体的子类定义吧
        if self.dec_target is not None:
            output_name = self.dec_target.id + ':' + '{'+self.coding_var_name+'}'
            backend.register_standalone(Op(reward_name, self.get_reward, [output_name], owner=self))
        else:
            backend.register_standalone(Op(reward_name, self.get_reward, [], owner=self))


class Generator(Node):
    '''
        Two generator method are provided, as shown below (key: class):
        1. 'poisson_generator': Poisson_Generator,
        2. 'cosine_generator': Cosine_Generator
    '''
    _node_sub_class = '<generator>'
    _coding_subclasses = dict()
    def  __init__(self, shape=None, num=None, dec_target=None,  dt=None,
                  coding_method=('poisson_generator', 'cc_generator', '...'),
                  coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Generator, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.gen_first = kwargs.get("gen_first", False) #only generate data for the first time, then use this date for later usage
        self.all_spikes = None
        self.build_level = 0
        self.new_input = True

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

    def init_state(self):
        self.index = 0
        self.new_input = True

    def torch_coding(self, source, device):
        '''

        Args:
            source (): It is input spike trains for encoding class and output spike trains for decoding class.
            device (): CPU or CUDA, this parameter is taken from backend

        Returns:

        '''
        raise NotImplementedError

    # initial operation: encoding input features into spike patterns
    def get_input(self):
        self.index = 0
        if (self.gen_first is True and self.all_spikes is None) or self.gen_first is False:
            if self.sim_name == 'pytorch':
                spikes = self.torch_coding(self.source, self.device)
            else:
                spikes = self.numpy_coding(self.source, self.device)
            self.all_spikes = spikes
        return self.all_spikes

    # stand alone operation: get spike pattern in every time step
    def next_stage(self):
        if self.new_input:
            self.get_input()
            self.new_input = False
        self.index += 1
        return self.all_spikes[self.index-1]

    def build(self, backend):
        self._backend = backend
        self.sim_name = backend.backend_name
        self.device = backend.device0
        # if self.dt is None:
        #     self.dt = backend.dt

        if self.sim_name == 'pytorch':
            singlestep_spikes = torch.zeros(self.shape, device=self.device)
        else:
            singlestep_spikes = np.zeros(self.shape)

        if self.dec_target is None:
            key = self.id + ':' + '{'+self.coding_var_name+'}'
            self.variable_to_backend(key, self.shape, value=singlestep_spikes)
        else:
            key = self.dec_target.id + ':' + '{'+self.coding_var_name+'}'
        self._var_names.append(key)
        self.init_op_to_backend(None, self.init_state, [])
        backend.register_standalone(Op(key, self.next_stage, [], owner=self))


# ======================================================================================================================
# Actions
# ======================================================================================================================

class Action(Node):
    '''
        Six action method are provided, as shown below (key: class):
        1. 'pop_rate_action': PopulationRate_Action
        2. 'softmax_action': Softmax_Action
        3. 'highest_spikes_action': Highest_Spikes_Action
        4. 'highest_voltage_action', Highest_Voltage_Action
        5. 'first_spike_action': First_Spike_Action
        6. 'random_action': Random_Action
    '''
    _node_sub_class = '<action>'
    _coding_subclasses = dict()
    def __init__(self, shape=None, num=None, dec_target=None,  dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Action, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.action = np.zeros((1,))


    def __new__(cls, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        coding_method = coding_method.lower()
        if cls is not Action:
            return super().__new__(cls)
        elif coding_method in Action._coding_subclasses:
            return super().__new__(Action._coding_subclasses[coding_method])
        else:
            raise ValueError("No action method: %s in Action classes" % coding_method)

    @staticmethod
    def register(name, coding_class):
        '''
        Register an action class. Registered action classes can be referred to
        # via their name.
        Parameters
        ----------
        name : str
            A short name for the state updater (e.g. 'pop_rate_action')
        coding_class :
            The subclass of coding object, e.g. an 'PopulationRate_Action'.
        '''

        # only deal with lower case names
        name = name.lower()
        if name in Action._coding_subclasses:
            raise ValueError(('A action class with the name "%s" has already been registered') % name)

        if not issubclass(coding_class, Action):
            raise ValueError(
                ('Given class of type %s does not seem to be a valid action class.' % str(type(coding_class))))

        Action._coding_subclasses[name] = coding_class

    def init_state(self):
        self.index = 0

    # stand alone operation: decoding spike patterns. Output is RL action
    def get_action(self, output):
        if (self.index % self.time_step) == 0:
            shape = list(output.shape)
            dec_shape = [self.time_step] + shape
            if type(output).__name__ == 'Tensor':
                self.records = torch.zeros(dec_shape, device=self.device)
            else:
                self.records = np.zeros(dec_shape)
        self.records[self.index % self.time_step, :] = output
        self.index += 1
        if self.index >= self.time_step:
            if self.sim_name == 'pytorch':
                self.action = self.torch_coding(self.records, self.source, self.device)
            else:
                self.action = self.numpy_coding(self.records, self.source, self.device)
        return 0

    def build(self, backend):
        self._backend = backend
        self.sim_name = backend.backend_name
        self.device = backend.device0
        # if self.dt is None:
        #     self.dt = backend.dt

        output_name = self.dec_target.id + ':' + '{'+self.coding_var_name+'}'
        self.init_op_to_backend(None, self.init_state, [])
        backend.register_standalone(Op(None, self.get_action, [output_name], owner=self))
