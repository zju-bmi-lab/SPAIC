Learner
=====================

This chapter will introduce the :code:`learner` in **SPAIC**. Recently, **SPAIC** supports **STCA**, **STBP**, **STDP** and **R-STDP** \
algorithms. **STCA** and **STBP** use **BPTT** by surrogate gradient. **STDP** is the classical unsupervised algorithms that use synaptic \
plasticity. **R-STDP** has a reward on **STDP** that suitable for reinforcement learning.

.. code-block:: python

        self.learner = spaic.Learner(trainable=self, algorithm='STCA', alpha=0.5)
        self.learner.set_optimizer('Adam', 0.001)
        self.learner.set_schedule('StepLR', 0.01)


In the sample code, we use **STCA** learning algorithm, users need to use :code:`trainable` to specify the training target. \
:code:`self` represent the whole network. If user doesn't want to train the whole network, can specify the target such as \
:code:`self.layer1` or :code:`[self.layer`, self.layer2]` . And the last :code:`alpha=0.5` is a parameters of STCA learning \
algorithm. In **SPAIC**, all the parameters of algorithms should be provided at the end of the function.

In the sample code, we also use :code:`Adam` optimization algorithm and :code:`StepLR` learning rate scheduler. **SPAIC** \
also has some other optimization algorithms:

    'Adam', 'AdamW', 'SparseAdam', 'Adamx', 'ASGD', 'LBFGS', 'RMSprop', 'Rpop', 'SGD',\
    'Adadelta', 'Adagrad'

and learning rate schedulers:

    'LambdaLR', 'StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
    'CyclicLR', 'CosineAnnealingWarmRestarts'

.. note::
    To ``RSTDPET`` learning algorithm, the ``batch_size`` should be 1.