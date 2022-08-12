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
定义编码方案的这一步可以依照 :code:`Neuron.Encoders` 文件中的格式进行添加。

编码方法初始化
--------------------------
自定义的编码方法需继承 :code:`Encoder` 类，其初始化方法中的参数名需与 :code:`Encoder` 类的一致，若需要传入初始化参数以外的参数，\
可以通过 :code:`kwargs` 传入，以 :code:`PoissonEncoding` 类初始化函数为例：

.. code-block:: python

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
             coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(PoissonEncoding, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type,
                                          **kwargs)
        self.unit_conversion = kwargs.get('unit_conversion', 0.1)

在这个初始化方法中，:code:`unit_conversion` 是 :code:`PoissonEncoding` 类所需要的参数，我们通过从 :code:`kwargs` 中获取的\
方式来设定。

定义编码函数
--------------------
编码函数是编码方法的实现部分，因为平台支持多后端（ :code:`pytorch` 、 :code:`TensorFlow` 以及 :code:`numpy` ），不同的后端\
支持的数据类型不同，相关的数据操作也不同，所以针对不同的计算后端需要在前端编码方法中实现对应的编码函数，根据平台支持的后端，\
内置的编码方法主要实现了三种编码函数，分别是 :code:`torch_coding` 、 :code:`tensorflow_coding` 以及 :code:`numpy_coding` ，\
用户可以根据想要使用的后端实现其中任意一种编码函数。我们以 :code:`PoissonEncoding` 编码方法的 :code:`torch_coding` 实现过程作为\
示例进行展示：

.. code-block:: python

    def torch_coding(self, source, device):
        # Source is raw real value data.
        # For full connection, the shape of source is [batch_size, num]
        # For convolution connection, the shape of source is [batch_size] + shape
        if source.__class__.__name__ == 'ndarray':
            source = torch.tensor(source, device=device, dtype=torch.float32)
        shape = source.shape
        # The shape of the encoded spike trains.
        spk_shape = [self.time_step] + list(shape)
        spikes = torch.rand(spk_shape, device=device).le(source * self.unit_conversion).float()
        return spikes

    Encoder.register("poisson", PoissonEncoding)

在代码的最后，需要添加 :code:`Encoder.register("poisson", PoissonEncoding)` 用于将该编码方法添加至编码方法的库中，\
以便前端的调用。

生成器自定义
--------------------------
生成器可用于生成服从特定分布的时空脉冲数据或者一些特殊的电流模式，在平台中内置了2种最常用的生成器方法，\
内置的生成器方法可能无法满足用户的任意需求，这时候就需要用户自己添加一些更符合其实验目的的生成器方案。\
定义生成器方案的这一步可以依照 :code:`Neuron.Generators` 文件中的格式进行添加。

生成器方法初始化
--------------------------
自定义的生成器方法需继承 :code:`Generator` 类，其初始化方法中的参数名需与 :code:`Generator` 类的一致，若需要传入初始化参数以外的参数，\
可以通过 :code:`kwargs` 传入，以恒定电流生成器 :code:`CC_Generator` 类的初始化函数为例：

.. code-block:: python

    def __init__(self, shape=None, num=None, dec_target=None,  dt=None,
                 coding_method=('poisson_generator', 'cc_generator', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(CC_Generator, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type,
                                       **kwargs)
        self.num = num

在这个初始化方法中，

定义生成器函数
--------------------
生成函数是生成方法的实现部分，因为平台支持多后端( :code:`pytorch` 、 :code:`TensorFlow` 以及 :code:`numpy` )，不同的后端\
支持的数据类型不同，相关的数据操作也不同，所以针对不同的计算后端需要在前端生成方法中实现对应的生成函数，根据平台支持的后端，\
内置的生成方法主要实现了三种生成函数，分别是 :code:`torch_coding` 、 :code:`tensorflow_coding` 以及 :code:`numpy_coding` ，\
用户可以根据想要使用的后端实现其中任意一种生成函数。我们以 :code:`CC_Generator` 生成方法的 :code:`torch_coding` 实现过程作为\
示例进行展示：

.. code-block:: python

    def torch_coding(self, source, device):
        if source.__class__.__name__ == 'ndarray':
            source = torch.tensor(source, dtype=torch.float, device=device)

        if source.ndim == 0:
            batch = 1
        else:
            batch = source.shape[0]

        shape = [batch, self.num]
        spk_shape = [self.time_step] + list(shape)
        spikes = source * torch.ones(spk_shape, device=device)
        return spikes


在代码的最后，需要添加 :code:`Generator.register('cc_generator', CC_Generator)` 用于将该生成器方法添加至生成器方法的库中，\
以便前端的调用。

解码器自定义
--------------------------
解码是将输出的脉冲信号进行一定程度的取舍和转换，为了满足用户的大多数应用需求，平台中内置了5种常用的解码方法，\
内置的解码方法可能无法满足用户的任意需求，这时候就需要用户自己添加一些更符合其实验目的的解码方案。\
定义解码方案的这一步可以依照 :code:`Neuron.Decoders` 文件中的格式进行添加。

解码方法初始化
--------------------------
自定义的解码方法需继承 :code:`Decoder` 类，其初始化方法中的参数名需与 :code:`Decoder` 类的一致，若需要传入初始化参数以外的参数，\
可以通过 :code:`kwargs` 传入，以 :code:`Spike_Counts` 类的初始化函数为例：

.. code-block:: python

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
            coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Spike_Counts, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type,
                                      **kwargs)
        self.pop_size = kwargs.get('pop_size', 1)

在这个初始化方法中，:code:`pop_size` 是 :code:`Spike_Counts` 类实现群体脉冲数解码所需要的参数，我们通过从 :code:`kwargs` 中\
获取的方式来设定。

定义解码函数
--------------------
解码函数是解码方法的实现部分，因为平台支持多后端（ :code:`pytorch` 、 :code:`TensorFlow` 以及 :code:`numpy` ），不同的后端\
支持的数据类型不同，相关的数据操作也不同，所以针对不同的计算后端需要在前端解码方法中实现对应的解码函数，根据平台支持的后端，\
内置的解码方法主要实现了三种解码函数，分别是 :code:`torch_coding` 、 :code:`tensorflow_coding` 以及 :code:`numpy_coding` ，\
用户可以根据想要使用的后端实现其中任意一种解码函数。我们以 :code:`Spike_Counts` 解码方法的 :code:`torch_coding` 实现过程作为\
示例进行展示：

.. code-block:: python

    def torch_coding(self, record, target, device):
        # record is the activities of the NeuronGroup to be decoded
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


在代码的最后，需要添加 :code:`Decoder.register('spike_counts', Spike_Counts)` 用于将该解码方法添加至解码方法的库中，\
以便前端的调用。