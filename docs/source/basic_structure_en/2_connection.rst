连接
===========

本章节主要介绍能够在SPAIC平台上使用的连接方式，包含了全连接，卷积连接，稀疏连接，一对一连接等。
作为脉冲神经网络最为基本的组成结构之一，连接中包含了网络最为重要的权重信息。与此同时，作为类脑计算平\
台，SPAIC平台中的连接支持仿生连接的形式，即支持反馈连接与连接延迟以及突触连接等具有一定生理特征\
的连接方式。

连接参数
--------------

.. code-block:: python

    def __init__(self, pre_assembly: Assembly, post_assembly: Assembly,
            name=None, link_type=('full', 'sparse', 'conv', '...'),
            policies=[], max_delay=0, sparse_with_mask=False, pre_var_name='O',
            post_var_name='WgtSum', **kwargs):

在连接的初始化参数中，我们可以看到，在建立连接时，必须给定的参数为突触前神经元、突触后神经元以及连接类型。

- pre_assembly - 突触前神经元，或是突触前神经元组，亦可视为连接的起点，上一层
- post_assembly - 突触后神经元，或是突触后神经元组，亦可视为连接的终点，下一层
- name - 连接的姓名，用于建立连接时更易区分，建议用户给定有意义的名称
- link_type - 连接类型，可选的有全连接、稀疏连接、卷积连接等
- policies - 连接的策略，
- max_delay - 突触延迟，即突触前神经元的信号将延迟几个时间步之后再传递给突触后神经元
- sparse_with_mask - 稀疏矩阵所用过滤器的开启与否
- pre_var_name - 突触前神经元对突触的输出，即该连接接收到的信号，默认接受到突触前神经元发放的‘output’脉冲信号，即默认\
为'O'
- post_var_name - 突触对突触后神经元的输出，即输出的信号，默认为输出权重和’WgtSum‘
- \**kwargs 在自定义参数中包含了某些连接所需要的特定参数，这些参数将在下文提及这些连接时谈到

除了这些参数以外，还有一些与权重相关的重要参数，例如:

- w_mean - 权重的平均值
- w_std - 权重的标准差
- w_max - 权重的最大值
- w_min - 权重的最小值
- weight - 权重值

在没有给定权重值，也就是用户没有传入weight的情况下，我们会进行权重的随机生成，这个时候就需要\
借用到w_mean与w_std，根据方差与均值生成随机数之后，若用户设定了w_min与w_max则截取在w_min\
与w_max之间的值作为权重，否则则直接将生成的随机数作为权重。

例如在连接conn1_example中，该连接在建立时将会根据均值为1，方差为5生成随机权重，并且将小于0.0的权重归为0.0，\
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

    self.weight = kwargs.get('weight', None) # 权重，如果不给定权重，连接将采取生成随机权重
    self.mask = kwargs.get('mask', None) #
    self.w_std = kwargs.get('w_std', 0.05) # 权重的方差，用于生成随机权重
    self.w_mean = kwargs.get('w_mean', 0.005) # 权重的均值，用于生成随机权重
    self.w_max = kwargs.get('w_max', None) # 权重的最大值，
    self.w_min = kwargs.get('w_min', None) # 权重的最小值，
    self.flatten_on = kwargs.get('flatten', False) # 拉平的操作，用于卷积层后

一对一连接
-----------------------
一对一连接在SPAIC中分为两种，基本的one_to_one以及稀疏形式的one_to_one_sparse，

.. code-block:: python

    self.conn_1to1 = spaic.Connection(self.layer1, self.layer2, link_type='one_to_one')
    self.conn_1to1s = spaic.Connection(self.layer1, self.layer2, link_type='one_to_one_sparse')

一对一连接主要包含的重要关键字参数为：

.. code-block:: python

    self.w_std = kwargs.get('w_std', 0.05) # 权重的方差，用于生成随机权重


卷积连接
-----------------------

稀疏连接
----------------------








