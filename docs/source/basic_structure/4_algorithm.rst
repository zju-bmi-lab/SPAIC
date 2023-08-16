算法
=====================

本章节主要介绍在 **SPAIC** 平台中内置的算法，目前我们已经在平台中添加了 **STCA** 、**STBP** 、**STDP** 与 **R-STDP**  算法。\
其中， **STCA** 与 **STBP** 都是采用了替代梯度的梯度反传算法，而 **STDP** 则是 **SNN** 中经典的无监督\
突触可塑性算法， **R-STDP** 在 **STDP** 的基础上添加了 :code:`reward` 机制，更好的适用于强化学习。

示例一
^^^^^^
.. code-block:: python

        #STCA learner
        self.learner = spaic.Learner(trainable=self, algorithm='STCA', alpha=0.5)
        self.learner.set_optimizer('Adam', 0.001)
        self.learner.set_schedule('StepLR', 0.01)

在示例一中，采用了 **STCA** 算法，用户在 :code:`trainable` 参数中传入需要训练的对象， :code:`self` \
代指整个网络。如果用户有针对性训练的需要，可以在 :code:`trainable` 的地方传入指定的层，例如 :code:`self.layer1` \
等，若需要传入多个指定层，则采用列表的 方式: :code:`[self.layer1, self.layer2]` 。如果用户制定了部分对象为可训练的，\
则需要启用 :code:`pathway` 参数，用于辅助梯度在全局的传递。需要将剩下不需要训练的对象添加至 :code:`pathway` 中，从而使其可以\
传递梯度。而最后的 :code:`alpha=0.5` 则是传入 **STCA** 的一个参数。 在 **SPAIC** 中，算法自带参数都在末尾进行传参。



此处还使用了 :code:`Adam` 优化算法与 :code:`StepLR` 学习率调整机制，在平台中我们\
设置了诸多可供使用的优化算法:

    'Adam', 'AdamW', 'SparseAdam', 'Adamx', 'ASGD', 'LBFGS', 'RMSprop', 'Rpop', 'SGD',\
    'Adadelta', 'Adagrad'

以及学习率调整机制：

    'LambdaLR', 'StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
    'CyclicLR', 'CosineAnnealingWarmRestarts'
示例二
^^^^^^
.. code-block:: python

        #STDP learner
        self._learner = spaic.Learner(trainable=self.connection1, algorithm='full_online_STDP', w_norm=3276.8)

在示例二中，采用了STDP算法，用户在 :code:`trainable` 参数中传入需要训练的对象，一般为神经网络的指定层。最后的 :code:`w_norm` 是传入 STDP 的参数，具体传入的参数名根据特定算法而定。

示例三
^^^^^^
.. code-block:: python

        # global reward
        self.reward = spaic.Reward(num=label_num, dec_target=self.layer3, coding_time=run_time,
                                     coding_method='global_reward', pop_size=pop_size, dec_sample_step=time_step)
        # RSTDP
        self._learner = spaic.Learner(trainable=[self.connection3], algorithm='RSTDP',
                                 lr=1, A_plus=1e-1, A_minus=-1e-2)

在示例三中，采用了RSTDP算法，该算法为有监督算法，用户需要用 :code:`Reward` 传入reward作为惩罚、奖励信号。 :code:`Reward` 和 :code:`RSTDP` 需要传入的参数分别在 :code:`spaic.Neuron.Rewards` 和 :code:`spaic.Learning.STDP_Learner` 中查看。
在训练过程中，区别于其他算法，需要用 :code:`Reward` 传递监督信号并使用 :code:`optim_step` 函数将后端的 :code:`parameters_dict` 和 :code:`variables` 的参数同步，如下所示。

.. code-block:: python

        Net.input(data)
        Net.reward(label)
        Net.run(run_time)
        output = Net.output.predict
        Net._learner.optim_step()

.. note::
    对于 ``RSTDPET`` 学习算法， ``batch_size`` 应设为1
