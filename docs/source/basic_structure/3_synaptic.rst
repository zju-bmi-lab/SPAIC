突触
===========

本章节将会主要提及脉冲神经网络中使用的不同的突触模型，主要介绍最为常用的化学突触与电突触两种。\
突触类型主要在建立连接时传入 :code:`syn_type` 进行设置，而突触的参数则由 :code:`syn_kwargs` 传入。

化学突触
---------------
化学突触是突触的正常表现形式，神经元与神经元之间通过突触递质进行信息传递，从而导致了某些\
离子浓度的上升亦或是下降。在计算神经科学中我们将生理上的突触转化为权重，从而使得兴奋递质\
与抑制性递质转化为正负权值的形式来作用到神经元上。

在 **SPAIC** 中，我们的连接默认使用化学突触的形式进行连接，神经元模型中也默认接收到的电流\
来自于突触传递的电流，因此我们将化学突触称之为基础突触，即 :code:`basic` 。

.. code-block:: python

    self.connection = spaic.Connection(self.layer1, self.layer2, link_type='full',
                                        syn_type=['basic'],
                                        w_std=0.0, w_mean=0.1)

电突触(Gap junction)
---------------------------------
电突触，也就是我们常称之为 ``Gap junction`` 的一种突触形式，其原理是由于突触前与突触后神经元\
相隔距离尤其地近，以至于产生了带电的离子互相交换。电突触的特点在于通常情况下电突触是双向的\
（即突触的作用同时作用在突触前神经元以及突触后神经元，使得双方的电压不断接近）

电突触的数学公式为 :code:`Igap = w_gap(Vpre - Vpost)`

若要使用电突触，则需要在连接的参数中将突触类型设置为电突触:

.. code-block:: python

    self.connection = spaic.Connection(self.layer1, self.layer2,
                                              link_type='full', syn_type=['electrical'],
                                              w_std=0.0, w_mean=0.1,
                                              )


全部突触
-----------------------
在突触中，我们还实现了一些其他种类的突触，并且将池化与展平操作都放入了突触中。

- **Basic_synapse** -- 通过 :code:`basic` 调用
- **conv_synapse** -- 通过 :code:`conv` 调用，配合卷积连接使用
- **DirectPass_synapse** -- 通过 :code:`directpass` 调用，该突触类型将会使得输出等同于输入部分，即输出的Isyn将会等同于连接的突触前神经元的输出的值
- **Dropout_synapse** -- 通过 :code:`dropout` 调用
- **AvgPool_synapse** -- 通过 :code:`avgpool` 调用
- **MaxPool_synapse** -- 通过 :code:`maxpool` 调用
- **BatchNorm2d_synapse** -- 通过 :code:`batchnorm2d` 调用
- **Flatten** -- 通过 :code:`flatten` 调用
- **First_order_chemical_synapse** -- 通过 :code:`1_order_synapse` 调用，化学突触中的一阶衰减突触
- **Second_order_chemical_synapse** -- 通过 :code:`2_order_synapse` 调用，化学突触中的二阶衰减突触
- **Mix_order_chemical_synapse** -- 通过 :code:`mix_order_synapse` 调用，化学突触中的混合衰减突触
