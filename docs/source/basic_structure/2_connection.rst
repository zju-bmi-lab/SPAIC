连接
===========

本章节主要介绍能够在 **SPAIC** 平台上使用的连接方式，包含了全连接，卷积连接，稀疏连接，一对一连接等。
作为脉冲神经网络最为基本的组成结构之一，连接中包含了网络最为重要的权重信息。与此同时，作为类脑计算平\
台， **SPAIC** 平台中的连接支持仿生连接的形式，即支持反馈连接与连接延迟以及突触连接等具有一定生理特征\
的连接方式。

连接参数
--------------

.. code-block:: python

    def __init__(self, pre: Assembly, post: Assembly, name=None,
            link_type=('full', 'sparse_connect', 'conv', '...'), syn_type=['basic'],
            max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn',
            syn_kwargs=None, **kwargs):

在连接的初始化参数中，我们可以看到，在建立连接时，必须给定的参数为 :code:`pre` , \
:code:`post` 以及 :code:`link_type` 。

- **pre** - 突触前神经元，或是突触前神经元组，亦可视为连接的起点，上一层
- **post** - 突触后神经元，或是突触后神经元组，亦可视为连接的终点，下一层
- **name** - 连接的姓名，用于建立连接时更易区分，建议用户给定有意义的名称
- **link_type** - 连接类型，可选的有全连接、稀疏连接、卷积连接等
- **syn_type** - 突触类型，将会在突触部分进行更为详细的讲解
- **max_delay** - 突触延迟，即突触前神经元的信号将延迟几个时间步之后再传递给突触后神经元
- **sparse_with_mask** - 稀疏矩阵所用过滤器的开启与否
- **pre_var_name** - 突触前神经元对突触的输出，即该连接接收到的信号，默认接受到突触前神经元发放的‘output’脉冲信号，即默认为'O'
- **post_var_name** - 突触对突触后神经元的输出，即输出的信号，默认为突触电流’Isyn‘
- **syn_kwargs** - 突触的自定义参数，将在突触介绍部分做进一步讲解
- **\**kwargs** - 在自定义参数中包含了某些连接所需要的特定参数，这些参数将在下文提及这些连接时谈到

除了这些参数以外，还有一些与权重相关的重要参数，例如:

- **w_mean** - 权重的平均值
- **w_std** - 权重的标准差
- **w_max** - 权重的最大值
- **w_min** - 权重的最小值
- **weight** - 权重值

在没有给定权重值，也就是用户没有传入 ``weight`` 的情况下，我们会进行权重的随机生成，这个时候就需要\
借用到 ``w_mean`` 与 ``w_std`` ，根据标准差与均值生成随机数之后，若用户设定了 ``w_min`` 与 ``w_max`` 则截取在 ``w_min`` \
与 ``w_max`` 之间的值作为权重，否则直接将生成的随机数作为权重。

例如在连接 ``conn1_example`` 中，该连接在建立时将会根据均值为1，标准差为5生成随机权重，并且将小于0.0的权重归为0.0，\
将大于2.0的权重归为2.0。

.. code-block:: python

    self.conn1_example = spaic.Connection(self.layer1, self.layer2, link_type='full',
                                    w_mean=1.0, w_std=5.0, w_min=0.0, w_max=2.0)

全连接
-----------
全连接是连接中最为基本的一种形式，

.. code-block:: python

    self.conn1_full = spaic.Connection(self.layer1, self.layer2, link_type='full')

全连接包含的重要关键字参数为：

.. code-block:: python

    weight = kwargs.get('weight', None) # 权重，如果不给定权重，连接将采取生成随机权重
    self.w_std = kwargs.get('w_std', 0.05) # 权重的标准差，用于生成随机权重
    self.w_mean = kwargs.get('w_mean', 0.005) # 权重的均值，用于生成随机权重
    self.w_max = kwargs.get('w_max', None) # 权重的最大值，
    self.w_min = kwargs.get('w_min', None) # 权重的最小值，

    bias = kwargs.get('bias', None) # 默认不使用bias，如果想要使用，可以传入Initializer对象或者与输出通道同维自定义向量对bias进行初始化
一对一连接
-----------------------
一对一连接在 **SPAIC** 中分为两种，基本的 ``one_to_one`` 以及稀疏形式的 ``one_to_one_sparse`` ，

.. code-block:: python

    self.conn_1to1 = spaic.Connection(self.layer1, self.layer2, link_type='one_to_one')
    self.conn_1to1s = spaic.Connection(self.layer1, self.layer2, link_type='one_to_one_sparse')

一对一连接主要包含的重要关键字参数为：

.. code-block:: python
    weight = kwargs.get('weight', None) # 权重，如果不给定权重，连接将采取生成随机权重
    self.w_mean = kwargs.get('w_mean', 0.005) # 权重的均值，用于生成随机权重
    self.w_max = kwargs.get('w_max', None) # 权重的最大值，
    self.w_min = kwargs.get('w_min', None) # 权重的最小值，

    bias = kwargs.get('bias', None) # 默认不使用bias，如果想要使用，可以传入Initializer对象或者与输出通道同维自定义向量对bias进行初始化
卷积连接
-----------------------
常见的卷积连接，池化方法可选择的有 :code:`avgpool` 以及 :code:`maxpool` ，这两个池化方法需要在突触类型中传入方可启用。

.. note::
    为了更好地提供对计算的支持，目前卷积连接需要与卷积突触一同使用。

卷积连接中主要包含的连接参数有：

.. code-block:: python

        self.out_channels = kwargs.get('out_channels', None)  # 输出通道
        self.in_channels = kwargs.get('in_channels', None)    # 输入通道
        self.kernel_size = kwargs.get('kernel_size', [3, 3])# 卷积核
        self.w_std = kwargs.get('w_std', 0.05) # 权重的标准差，用于生成随机权重
        self.w_mean = kwargs.get('w_mean', 0.05) # 权重的均值，用于生成随机权重
        weight = kwargs.get('weight', None) # 权重，如果不给定权重，连接将采取生成随机权重

        self.stride = kwargs.get('stride', 1)
        self.padding = kwargs.get('padding', 0)
        self.dilation = kwargs.get('dilation', 1)
        self.groups = kwargs.get('groups', 1)
        self.upscale = kwargs.get('upscale', None)

        bias = kwargs.get('bias', None) # 默认不使用bias，如果想要使用，可以传入Initializer对象或者与输出通道同维自定义向量对bias进行初始化


卷积连接的示例1：

.. code-block:: python

        # 通过Initializer对象初始化 weight 和 bias
        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='conv', syn_type=['conv'],
                                                in_channels=1, out_channels=4,
                                                kernel_size=(3, 3),
                                                weight=kaiming_uniform(a=math.sqrt(5)),
                                                bias=uniform(a=-math.sqrt(1 / 9), b=math.sqrt(1 / 9))
                                                )
        # 传入自定义值初始化 weight 和 bias
        self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='conv', syn_type=['conv'],
                                              in_channels=4, out_channels=8, kernel_size=(3, 3),
                                              weight=w_std * np.random.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]) + self.w_mean,
                                              bias=np.empty(out_channels)
                                              )
        # 根据默认的w_std和w_mean随机生成初始化权重
        self.connection3 = spaic.Connection(self.layer2, self.layer3, link_type='conv', syn_type=['conv'],
                                              in_channels=8, out_channels=8, kernel_size=(3, 3)
                                              )
        # 通过Initializer对象初始化 weight 和 bias
        self.connection4 = spaic.Connection(self.layer3, self.layer4, link_type='full',
                                              syn_type=['flatten', 'basic'],
                                              weight=kaiming_uniform(a=math.sqrt(5)),
                                              bias=uniform(a=-math.sqrt(1 / layer3_num), b=math.sqrt(1 / layer3_num))
                                              )
        # 传入自定义值初始化 weight 和 bias
        self.connection5 = spaic.Connection(self.layer4, self.layer5, link_type='full',
                                              weight=w_std * np.random.randn(layer4_num, layer3_num) + self.w_mean,
                                              bias=np.empty(layer5_num)
                                              )


卷积连接的示例2：

.. code-block:: python

        self.conv2 = spaic.Connection(self.layer1, self.layer2, link_type='conv',
                                        syn_type=['conv', 'dropout'], in_channels=128, out_channels=256,
                                        kernel_size=(3, 3), stride=args.stride, padding=args.padding,
                                        weight=kaiming_uniform(a=math.sqrt(5)),
                                        bias=uniform(a=-math.sqrt(1 / 1152), b=math.sqrt(1 / 1152))
                                        )
        self.conv3 = spaic.Connection(self.layer2, self.layer3, link_type='conv',
                                        syn_type=['conv', 'maxpool', 'dropout'], in_channels=256, out_channels=512,
                                        kernel_size=(3, 3), stride=args.stride, padding=args.padding,
                                        pool_stride=2, pool_padding=0,
                                        weight=kaiming_uniform(a=math.sqrt(5)),
                                        bias=uniform(a=-math.sqrt(1 / 2304), b=math.sqrt(1 / 2304))
                                        )
        self.conv4 = spaic.Connection(self.layer3, self.layer4, link_type='conv',
                                        syn_type=['conv', 'maxpool', 'dropout'], in_channels=512, out_channels=1024,
                                        kernel_size=(3, 3), stride=args.stride, padding=args.padding,
                                        pool_stride=2, pool_padding=0,
                                        weight=kaiming_uniform(a=math.sqrt(5)),
                                        bias=uniform(a=-math.sqrt(1 / 4608), b=math.sqrt(1 / 4608))
                                        syn_kwargs={'p': 0.6})


稀疏连接
----------------------
常见的稀疏连接，通过传入参数 :code:`density` 来设置稀疏连接的连接稠密程度

随机连接
---------------------------
常见的随机连接，通过传入参数 :code:`probability` 来设置随机连接的连接概率








