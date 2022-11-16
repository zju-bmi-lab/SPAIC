Synapse
===========

This chapter will introduce synaptic models in **SPAIC** .

Chemical Synapse
---------------------
``Chemical synapse`` is a common form of synapse, information transmitted between neurons by synaptic transmitters, \
which causes some concentration of certain ions to change. In computational neuroscience, we use weight and calculate \
form to simulate the physiological synapse that excitatory and inhibitory transmitters are simulated with plus or minus weight.

In **SPAIC** , by default, the synapse use chemical synapse and neurons use 'Isyn' as input. \
So, we call the basic chemical synapse as :code:`basic` .

.. code-block:: python

    self.connection = spaic.Connection(self.layer1, self.layer2, link_type='full',
                                        syn_type=['basic'],
                                        w_std=0.0, w_mean=0.1)

Gap Junction
---------------------------------
``Gap junction`` , another common form of synapse. The presynaptic and postsynaptic neurons are so closely that \
charged ions exchange with each other. The characteristic of gap junction is that they are usually bidirection \
(i.e. the action of the synapse acts on both the presynaptic neuron and the postsynaptic neuron, bringing the \
voltage of the two neurons closer together).

The calculate form of gap junction: ``Igap = w_gap(Vpre - Vpost)``

If users want to use gap junction, need to set the synapse_type as :code:`electrical` .

.. code-block:: python

    self.connection = spaic.Connection(self.layer1, self.layer2,
                                              link_type='full', syn_type=['electrical'],
                                              w_std=0.0, w_mean=0.1,
                                              )


All Synapses
-----------------------
In ``Synapse`` , we also construct some other synapse, including pooling and flatten.


- **Basic_synapse** -- :code:`basic`
- **conv_synapse** -- :code:`conv` combine with convolution connection.
- **DirectPass_synapse** -- :code:`directpass`  , choose this synapse will let output equal to the input, which means the output :code:`Isyn` will equal to the output value of presynapse neurons.
- **Dropout_synapse** -- :code:`dropout`
- **AvgPool_synapse** -- :code:`avgpool`
- **MaxPool_synapse** -- :code:`maxpool`
- **BatchNorm2d_synapse** -- :code:`batchnorm2d`
- **Flatten** -- :code:`flatten`
- **First_order_chemical_synapse** -- :code:`1_order_synapse` , first order attenuated synapses in chemical synapses
- **Second_order_chemical_synapse** -- :code:`2_order_synapse` , second order attenuated synapses in chemical synapses
- **Mix_order_chemical_synapse** -- :code:`mix_order_synapse` , mix order attenuated synapses in chemical synapses

- **Max pooling** -- :code:`maxpool`
- **Average pooling** -- :code:`avgpool`
- **Flatten** -- :code:`flatten`
- **Dropout** --  :code:`dropout`
- **Direct pass** -- :code:`directpass` , choose this synapse will let output equal to the input, which means the output :code:`Isyn` will equal to the output value of presynapse neurons.
