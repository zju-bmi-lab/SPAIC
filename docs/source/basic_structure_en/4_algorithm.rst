Learner
=====================

This chapter will introduce the :code:`learner` in **SPAIC**. Recently, **SPAIC** supports **STCA**, **STBP**, **STDP** and **R-STDP** \
algorithms. **STCA** and **STBP** use **BPTT** by surrogate gradient. **STDP** is the classical unsupervised algorithms that use synaptic \
plasticity. **R-STDP** has a reward on **STDP** that suitable for reinforcement learning.

Example 1
^^^^^^
.. code-block:: python

        self.learner = spaic.Learner(trainable=self, algorithm='STCA', alpha=0.5)
        self.learner.set_optimizer('Adam', 0.001)
        self.learner.set_schedule('StepLR', 0.01)


In the sample code, we use **STCA** learning algorithm, users need to use :code:`trainable` to specify the training target. \
:code:`self` represent the whole network. If user doesn't want to train the whole network, can specify the target such as \
:code:`self.layer1` or :code:`[self.layer1, self.layer2]` . And the last :code:`alpha=0.5` is a parameters of STCA learning \
algorithm. In **SPAIC**, all the parameters of algorithms should be provided at the end of the function.

In the sample code, we also use :code:`Adam` optimization algorithm and :code:`StepLR` learning rate scheduler. **SPAIC** \
also has some other optimization algorithms:

    'Adam', 'AdamW', 'SparseAdam', 'Adamx', 'ASGD', 'LBFGS', 'RMSprop', 'Rpop', 'SGD',\
    'Adadelta', 'Adagrad'

and learning rate schedulers:

    'LambdaLR', 'StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
    'CyclicLR', 'CosineAnnealingWarmRestarts'

Example 2
^^^^^^
.. code-block:: python

        #STDP learner
        self._learner = spaic.Learner(trainable=self.connection1, algorithm='full_online_STDP', w_norm=3276.8)

In example 2, we use the STDP Algorithm. Users pass in stuff to be trained in :code:`trainable` , usually a certain layer of the SNN. The final :code:`w_norm` is the parameter of the STDP Algorithm. The parameters to pass in is determined by certain Algorithm.

Example 3
^^^^^^
.. code-block:: python

        # global reward
        self.reward = spaic.Reward(num=label_num, dec_target=self.layer3, coding_time=run_time,
                                     coding_method='global_reward', pop_size=pop_size, dec_sample_step=time_step)
        # RSTDP
        self._learner = spaic.Learner(trainable=[self.connection3], algorithm='RSTDP',
                                 lr=1, A_plus=1e-1, A_minus=-1e-2)

In Example 3, the RSTDP algorithm is used, which is a supervised algorithm.The user needs to use :code:`Reward` to pass in reward as a penalty or reward signal. Parameters of :code:`Reward` and :code:`RSTDP` can be checked in :code:`spaic.Neuron.Rewards` and :code:`spaic.Learning.STDP_Learner` respectively.
In the training process, different from other algorithms, users need to use :code:`Reward` to pass the supervision signal and use the :code:`optim_step` function to synchronize the :code:`parameters_dict` and :code:`variables` parameters of the backend, as follows.

.. code-block:: python

        Net.input(data)
        Net.reward(label)
        Net.run(run_time)
        output = Net.output.predict
        Net._learner.optim_step()
.. note::
    To ``RSTDPET`` learning algorithm, the ``batch_size`` should be 1.