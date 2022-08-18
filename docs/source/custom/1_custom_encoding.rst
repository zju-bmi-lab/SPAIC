.. _my-custom-encoding:

编解码方法自定义
=======================
本章节主要介绍编码器、生成器、解码器、奖励器以及动作器的自定义，以便当本平台提供的内置方法无法满足用户需求时，\
用户可以方便的添加符合自己需求的编解码方案。

编码器自定义
--------------------------
编码是将输入的数据转化为脉冲神经网络可用的时序脉冲数据，是搭建神经网络要考虑的重要一步，\
不同的编码方法会生成不同的时序脉冲数据，为了满足用户的大多数应用需求，在本平台中内置了6种最常用的编码方法，\
内置的编码方法可能无法满足用户的任意需求，这时候就需要用户自己添加一些更符合其实验目的的编码方案。\
定义编码方案的这一步可以依照 :class:`spaic.Neuron.Encoders` 文件中的格式进行添加。

编码方法初始化
^^^^^^^^^^^^^^^^^^^^^
自定义的编码方法需继承 :code:`Encoder` 类，其初始化方法中的参数名需与 :code:`Encoder` 类的一致，若需要传入初始化参数以外的参数，\
可以通过 :code:`kwargs` 传入，以 :code:`PoissonEncoding` 类初始化函数为例：

.. code-block:: python

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method='poisson',
             coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(PoissonEncoding, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type,
                                          **kwargs)
        self.unit_conversion = kwargs.get('unit_conversion', 1.0)

在这个初始化方法中，:code:`unit_conversion` 是 :code:`PoissonEncoding` 类所需要的参数，我们通过从 :code:`kwargs` 中获取的\
方式来设定。

定义编码函数
^^^^^^^^^^^^^^^^^^^^^
编码函数是编码方法的实现部分，因为平台计划支持多后端（ :code:`pytorch` 、 :code:`TensorFlow` 等），不同的后端\
支持的数据类型不同，相关的数据操作也不同，所以针对不同的计算后端需要在前端编码方法中实现对应的编码函数。 \
我们以 :code:`PoissonEncoding` 编码方法的 :code:`torch_coding` 实现过程作为示例进行展示：

.. code-block:: python

    def torch_coding(self, source, device):
        # Source is raw real value data.
        # For full connection, the shape of source is [batch_size, num]
        # For convolution connection, the shape of source is [batch_size] + shape
        if source.__class__.__name__ == 'ndarray':
            source = torch.tensor(source, device=device, dtype=self._backend.data_type)
        # The shape of the encoded spike trains.
        spk_shape = [self.time_step] + list(self.shape)
        spikes = torch.rand(spk_shape, device=device).le(source * self.unit_conversion*self.dt).float()
        return spikes

在最后，需要添加 :code:`Encoder.register("poisson", PoissonEncoding)` 用于将该编码方法添加至编码方法的库中。

生成器自定义
--------------------------
生成器可用于生成服从特定分布的时空脉冲数据或者一些特殊的电流模式，在平台中内置了2种最常用的生成器方法，\
内置的生成器方法可能无法满足用户的任意需求，这时候就需要用户自己添加一些更符合其实验目的的生成器方案。\
定义生成器方案的这一步可以依照 :class:`spaic.Neuron.Generators` 文件中的格式进行添加。

生成器方法初始化
^^^^^^^^^^^^^^^^^^^^^
自定义的生成器方法需继承 :code:`Generator` 类，其初始化方法中的参数名需与 :code:`Generator` 类的一致，若需要传入初始化参数以外的参数，\
可以通过 :code:`kwargs` 传入，以恒定电流生成器 :code:`CC_Generator` 类的初始化函数为例：

.. code-block:: python

    def __init__(self, shape=None, num=None, dec_target=None,  dt=None,
                 coding_method='cc_generator', coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(CC_Generator, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type,
                                       **kwargs)


定义生成器函数
^^^^^^^^^^^^^^^^^^^^^
生成函数是生成方法的实现部分，因为平台计划支持多后端（ :code:`pytorch` 、 :code:`TensorFlow` 等），不同的后端\
支持的数据类型不同，相关的数据操作也不同，所以针对不同的计算后端需要在前端生成方法中实现对应的生成函数。 \
我们以 :code:`CC_Generator` 生成方法的 :code:`torch_coding` 实现过程作为示例进行展示：

.. code-block:: python

    def torch_coding(self, source, device):

        if not (source >= 0).all():
            import warnings
            warnings.warn('Input current shall be non-negative')
        if source.__class__.__name__ == 'ndarray':
            source = torch.tensor(source, dtype=self._backend.data_type, device=device)

        spk_shape = [self.time_step] + list(self.shape)
        spikes = source * torch.ones(spk_shape, device=device)
        return spikes


在最后，需要添加 :code:`Generator.register('cc_generator', CC_Generator)` 用于将该生成器方法添加至生成器方法的库中。

解码器自定义
--------------------------
解码是将输出的脉冲信号进行一定程度的取舍和转换，为了满足用户的大多数应用需求，平台中内置了5种常用的解码方法，\
内置的解码方法可能无法满足用户的任意需求，这时候就需要用户自己添加一些更符合其实验目的的解码方案。\
定义解码方案的这一步可以依照 :class:`spaic.Neuron.Decoders` 文件中的格式进行添加。

解码方法初始化
^^^^^^^^^^^^^^^^^^^^^
自定义的解码方法需继承 :code:`Decoder` 类，其初始化方法中的参数名需与 :code:`Decoder` 类的一致，若需要传入初始化参数以外的参数，\
可以通过 :code:`kwargs` 传入，以 :code:`Spike_Counts` 类的初始化函数为例：

.. code-block:: python

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method='spike_counts',
            coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Spike_Counts, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type,
                                      **kwargs)
        self.pop_size = kwargs.get('pop_size', 1)

在这个初始化方法中，:code:`pop_size` 是 :code:`Spike_Counts` 类实现群体脉冲数解码所需要的参数，我们通过从 :code:`kwargs` 中\
获取的方式来设定。

定义解码函数
^^^^^^^^^^^^^^^^^^^^^
解码函数是解码方法的实现部分，因为平台计划支持多后端（ :code:`pytorch` 、 :code:`TensorFlow` 等），不同的后端\
支持的数据类型不同，相关的数据操作也不同，所以针对不同的计算后端需要在前端解码方法中实现对应的解码函数。 \
我们以 :code:`Spike_Counts` 解码方法的 :code:`torch_coding` 实现过程作为示例进行展示：

.. code-block:: python

    def torch_coding(self, record, target, device):
        # record is the activity of the NeuronGroup to be decoded
        # the shape of record is (time_step, batch_size, n_neurons)
        # target is the label of the sample
        spike_rate = record.sum(0).to(device=device)
        pop_num = int(self.num / self.pop_size)
        pop_spikes_temp = (
            [
                spike_rate[:, (i * self.pop_size): (i * self.pop_size) + self.pop_size].sum(dim=1)
                for i in range(pop_num)
            ]
        )
        pop_spikes = torch.stack(pop_spikes_temp, dim=-1)
        return pop_spikes


在最后，需要添加 :code:`Decoder.register('spike_counts', Spike_Counts)` 用于将该解码方法添加至解码方法的库中。

奖励器自定义
--------------------------
奖励用于将目标对象的活动转化为奖励信号。为了满足用户的大多数应用需求，平台中内置了4种常用的奖励方法，\
内置的奖励方法可能无法满足用户的任意需求，这时候就需要用户自己添加一些更符合其实验目的的奖励方案。\
定义奖励方案的这一步可以依照 :class:`spaic.Neuron.Rewards` 文件中的格式进行添加。

奖励方法初始化
^^^^^^^^^^^^^^^^^^^^^
自定义的奖励方法需继承 :code:`Reward` 类，其初始化方法中的参数名需与 :code:`Reward` 类的一致，若需要传入初始化参数以外的参数，\
可以通过 :code:`kwargs` 传入，以 :code:`Global_Reward` 类的初始化函数为例：

.. code-block:: python

    def __init__(self,shape=None, num=None, dec_target=None, dt=None, coding_method='global_reward', coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Global_Reward, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.pop_size = kwargs.get('pop_size', 1)
        self.reward_signal = kwargs.get('reward_signal', 1)
        self.punish_signal = kwargs.get('punish_signal', -1)

在这个初始化方法中，**pop_size**, **reward_signal**, **punish_signal** 是 :code:`Global_Reward` 类需要的参数，我们通过从 :code:`kwargs` 中\
获取的方式来设定。

定义奖励函数
^^^^^^^^^^^^^^^^^^^^^
奖励函数是奖励方法的实现部分，因为平台计划支持多后端（ :code:`pytorch` 、 :code:`TensorFlow` 等），不同的后端\
支持的数据类型不同，相关的数据操作也不同，所以针对不同的计算后端需要在前端奖励方法中实现对应的奖励函数。 \
我们以 :code:`Global_Reward` 奖励方法的 :code:`torch_coding` 实现过程作为示例进行展示：

.. code-block:: python

    def torch_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        spike_rate = record.sum(0)
        pop_num = int(self.num / self.pop_size)
        pop_spikes_temp = (
            [
                spike_rate[:, (i * self.pop_size): (i * self.pop_size) + self.pop_size].sum(dim=1)
                for i in range(pop_num)
            ]
        )
        pop_spikes = torch.stack(pop_spikes_temp, dim=-1)
        predict = torch.argmax(pop_spikes, dim=1)  # return the indices of the maximum values of a tensor across columns.
        reward = self.punish_signal * torch.ones(predict.shape, device=device)
        flag = torch.tensor([predict[i] == target[i] for i in range(predict.size(0))])
        reward[flag] = self.reward_signal
        if len(reward) > 1:
            reward = reward.mean()
        return reward


在最后，需要添加 :code:`Reward.register('global_reward', Global_Reward)` 用于将该奖励方法添加至奖励方法的库中.

动作器自定义
--------------------------
动作用于将目标对象的活动转化为下一步的动作。为了满足用户的大多数应用需求，平台中内置了6种常用的动作方法，\
内置的动作方法可能无法满足用户的任意需求，这时候就需要用户自己添加一些更符合其实验目的的动作方案。\
定义动作方案的这一步可以依照 :class:`spaic.Neuron.Actions` 文件中的格式进行添加。

动作方法初始化
^^^^^^^^^^^^^^^^^^^^^
自定义的动作方法需继承 :code:`Action` 类，其初始化方法中的参数名需与 :code:`Action` 类的一致，若需要传入初始化参数以外的参数，\
可以通过 :code:`kwargs` 传入，以 :code:`Softmax_Action` 类的初始化函数为例：

.. code-block:: python

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method='softmax_action', coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Softmax_Action, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)


定义动作函数
^^^^^^^^^^^^^^^^^^^^^
动作函数是动作方法的实现部分，因为平台计划支持多后端（ :code:`pytorch` 、 :code:`TensorFlow` 等），不同的后端\
支持的数据类型不同，相关的数据操作也不同，所以针对不同的计算后端需要在前端动作方法中实现对应的动作函数。 \
我们以 :code:`Softmax_Action` 奖励方法的 :code:`torch_coding` 实现过程作为示例进行展示：

.. code-block:: python

    def torch_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        assert (
            record.shape[2] == self.num
        ), "Output layer size is not equal to the size of the action space."
        spikes = torch.sum(record, dim=0)
        probabilities = torch.softmax(spikes, dim=0)
        return torch.multinomial(probabilities, num_samples=1).item()


在最后，需要添加 :code:`Action.register('softmax_action', Softmax_Action)` 用于将该动作方法添加至动作方法的库中。