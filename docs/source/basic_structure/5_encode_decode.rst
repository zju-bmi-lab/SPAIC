编码解码
====================
本章节主要关注SPAIC平台中的编码、解码、信号生成、奖励以及动作生成。\
该章节主要分为五大块，编码器、生成器、解码器、奖励器以及动作器。

编码器(Encoder)
-------------------------------
:code:`Encoder` 类是 :code:`Node` 类的子类，编码器主要用于在脉冲神经网络中，将输入的数据转化为脉冲神经网络可用的时序脉冲数据。因为\
对于脉冲神经网络而言，以往人工神经网络中的数值输入不符合生理特征，通常使用二值的脉冲数\
据数据输入。并且静态的数据输入无法获取数据的时间特征，转化为具有时序的脉冲数据能够更好\
地表现数据的时序特征。在SPAIC中，我们内置了一系列较为常见的编码方式：

- sstb - SingleSpikeToBinary
- mstb - MultipleSpikeToBinary
- Poisson - Poisson
- Latency - Latency
- relative_latency - Relative_Latency
- null - NullEncoder

.. note::

    The shape of encoded date should be (batch, time_step, shape)

编码器主要在脉冲输入阶段使用，以构建具有100个神经元的Poisson编码类实例为例：

.. code-block:: python

    self.input = spaic.Encoder(num=100, coding_method='poisson', unit_conversion=1.0) # unit_conversion为缩放参数，将会对脉冲的发放频率进行缩放

生成器(Generator)
------------------------------
:code:`Generator` 类是 :code:`Node` 类的子类，生成器主要的作用在于，有时在进行神经元动力学仿真时，我们需要特殊的输入模式，因此我们需要\
有一些特殊的电压或者是电流模式的生成器。在SPAIC中，我们内置了一些模式生成器：

- poisson_generator - Poisson spike according input rate
- cc_generator - Constant current generator

.. code-block:: python

    self.input = spaic.Generator(num=1, coding_method='cc_generator')

:code:`cc_generator` 将会生成持续的脉冲输出，用于用户观察模拟各类神经元动力学。而另一种 :code:`poisson_generator` \
则会生成poisson噪声输出。

解码器(Decoder)
------------------------------
:code:`Decoder` 类是 :code:`Node` 类的子类，其主用于在脉冲神经网络中，将输出的脉冲信号进行一定程度的取舍和转换，例如根据\
:code:`spike_counts` 的规则选取发放脉冲数量最多的神经元作为预测结果，亦或是根据 :code:`first_spike` \
的规则选取第一个发放脉冲的神经元作为预测结果。在SPAIC中，我们也内置了大多数较为常见\
的解码方式：

- spike_counts - Spike_Counts
- final_step_voltage - Final_Step_Voltage
- first_spike - First_Spike
- time_spike_counts - TimeSpike_Counts
- time_softmax - Time_Softmax

解码器主要在脉冲输出阶段使用，例如当解码含有10个LIF神经元的 :code:`NeuronGroup` 对象的脉冲活动时,\
我们可以这样建立 :code:`spike_counts` 类实例:

.. code-block:: python

    self.layer1 = spaic.NeuronGroup(neuron_number=10, neuron_model='lif')
    self.output = spaic.Decoder(num=10, dec_target=self.layer1,
                    coding_method='spike_counts', coding_var_name='O')

.. note::

   这里较为需要注意的是，:code:`output` 层的数量需要与其 :code:`dec_target` 目标层的神经元数量一致。

奖励(Reward)
------------------------------
 :code:`Reward` 类是 :code:`Node` 类的子类，主要作用是在执行强化任务的时候，有时需要根据任务目的解码指定对象的活动并\
设定奖励规则来获取奖励。例如分类任务下的 :code:`global_reward` 的规则，根据脉冲发放数量\
或者最大膜电位确定预测结果，若预测结果是期望的结果，则返回正奖励；\
若不等，则返回负奖励。样本的batch_size>1时，返回取均值后的奖励作为全局奖励。\
在SPAIC中，我们内置了一些奖励类：

- global_reward - Global_Reward
- xor_reward - XOR_Reward
- da_reward - DA_Reward: get rewards in the same dimension as neurons in the dec_target
- environment_reward - Environment_Reward: get reward from environment

用户根据自己的需要指定在 :code:`Reward` 中要解码对象的神经元数量、解码对象名、奖励方法、解码对象的变量名及与其方法相关的参数等。

Global_Reward
------------------
例如当解码含有10个LIF神经元的NeuronGroup对象的脉冲活动以获得全局奖励时，我们可以这样建立Global_Reward类实例:

.. code-block:: python

    self.layer1 = spaic.NeuronGroup(neuron_number=10, neuron_model='lif')
    self.reward = spaic.Reward(num=10, dec_target=self.layer1,
                    coding_method='global_reward', coding_var_name='O')

.. note::

   这里需要注意的是，reward实例的神经元数量需要与其dec_target目标层的神经元数量一致。

一个解码self.layer1的脉冲活动以获取全局奖励的全局奖励实例就建立好了。然而许多时候我们需要按需定制不同的 :code:`Reward` \
以获得不同的奖励方案，这时候就需要在建立 :code:`Reward` 时，指定一些参数：

- pop_size - 解码神经元的群体尺寸，默认为1
- dec_sample_step - 解码采样时间步，默认为1
- reward_signal - 奖励信号，默认为1
- punish_signal - 惩罚信号，默认为-1

如果用户需要调整这些变量，可以在建立Reward的时候输入想改变的参数即可：

.. code-block:: python

    self.reward = spaic.Reward(num=10, dec_target=self.layer1, coding_method='global_reward',
                    coding_var_name='O', reward_signal=2, punish_signal=-2)

这样，一个自定义参数的Global_Reward实例就建好了。

动作(Action)
------------------------------
Action类是 :code:`Node` 类的子类，主要作用是在执行GYM强化环境中的强化任务时，需要根据指定对象的活动设定动作选择机制\
选择接下来要执行的动作。例如PopulationRate_Action规则，解码对象的神经元的群体数与动作数目个数一致，\
以每个群体的发放速率为权重来选择下一步动作，群体的发放速率越大，选中的可能性越大。\
在SPAIC中，我们内置了一些动作类：

- pop_rate_action - PopulationRate_Action
- softmax_action - Softmax_Action
- highest_spikes_action - Highest_Spikes_Action
- highest_voltage_action - Highest_Voltage_Action
- first_spike_action - First_Spike_Action
- random_action - Random_Action

用户根据自己的需要指定在Action中要解码对象的神经元数量、解码对象名、动作方法、解码对象的变量名及与其方法相关的参数等。

PopulationRate_Action
------------------
例如当解码含有5个LIF神经元的NeuronGroup对象的脉冲活动以获得下一步活动时，我们可以这样建立 :code:`PopulationRate_Action` 类实例:

.. code-block:: python

    self.layer1 = spaic.NeuronGroup(neuron_number=5, neuron_model='lif')
    self.action = spaic.Action(num=5, dec_target=self.layer1,
                    coding_method='pop_rate_action', coding_var_name='O')

一个解码 :code:`self.layer1` 的脉冲活动以获取下一步动作的群体速率动作实例就建立好了。然而许多时候我们需要按需定制不同的Action\
以获得不同的奖励方案，这时候就需要在建立 :code:`Reward` 时，指定一些参数：

- pop_size - 解码神经元的群体尺寸，默认为1

如果用户需要调整这些变量，可以在建立 :code:`Reward` 的时候输入想改变的参数即可：

.. note::

   这里需要注意的是，action实例的神经元数量需要与其dec_target目标层的神经元数量一致，且num/pop_size的结果\
   应为整数且与强化环境的动作数目相同。

