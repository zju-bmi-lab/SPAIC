Encoder & Decoder
====================
This chapter introduces five classes, including ``Encoder`` , ``Generator`` , ``Decoder`` , ``Reward`` and ``Action`` . ``Encoder`` \
and ``Generator`` are used to encode input data into spike trains. ``Decoder`` , ``Reward`` and ``Action`` are used \
to decode output spike trains to obtain predict label, reward signal and action.

Encoder
-------------------------------
The class :code:`Encoder` is a subclass of the class :code:`Node` . \
The ``Encoder`` is mainly used to convert the input data into spike trains available in the spiking neural network. \
For the spiking neural network, the numerical input of the previous artificial neural network does not conform to the \
physiological characteristics, and the binary spike data is usually used as the input.

In **SPAIC** , we have built in some common encoding methods:

- SingleSpikeToBinary ('sstb')
- MultipleSpikeToBinary ('mstb')
- PoissonEncoding ('poisson')
- Latency ('latency')
- NullEncoder ('null')

When instantiating an encoded class, users need to specify the **shape** or **num** , **coding_method** and other related paramters.

SingleSpikeToBinary ('sstb')
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If the input data is a vector of the firing times of neurons, and one neuron corresponds to one firing time. \
We can use the ``SingleSpikeToBinary`` encoding method to convert the firing time into a binary matrix. \
The firing time corresponds to the index of time window.

For example, converting firing times [0.9, 0.5, 0.2, 0.7, 0.1] of one input sample into a binary matrix.

.. code-block:: python

    self.input = spaic.Encoder(num=node_num, coding_method='sstb')
    # or
    self.input = spaic.Encoder(shape=[node_num], coding_method='sstb')

.. note::
    The **num** of node is equal to the size of one input sample, namely 5, and the network runtime is equal to the maximum firing time of dataset plus dt, namely 1.0.


MultipleSpikeToBinary ('mstb')
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If the input data contains the index of the firing neuron and its firing time, we can use the ``MultipleSpikeToBinary`` \
encoding method to convert the firing index and firing time into a binary matrix.

For example, converting [[[0.9, 0.5, 0.2, 0.7, 0.1], [1, 1, 3, 2, 6]]] into a binary matrix.

.. code-block:: python

    self.input = spaic.Encoder(num=node_num, shape=[2, node_num], coding_method='mstb')

.. note::
    - Since one neuron corresponds to zero or multiple spike times, the size of each samples may be different.
    - The **num** of node is equal to the maximum id of neuron plus 1, namely 7.
    - The **shape** of one sample should be [2, node_num], where the first row is the vector of firing times, and the second row is the vector of neuron ids.
    - For MultipleSpikeToBinary encoding method, initialization parameters **num** and **shape** need to be specified simultaneously.


PoissonEncoding ('poisson')
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``Poisson coding`` method coding method belongs to rate coding. The stronger the stimulus, the higher the firing rate. \
Taking the image input as an example, the pixel intensity is first mapped to the instantaneous firing rate of the input neuron. \
Then at each time step, an uniform random number between 0 and 1 is generated and compared with the instantaneous firing rate. \
If the random number is less than the instantaneous rate, a spike is generated.

The following code defines a PoissonEncoder object which transforms the input into poisson spike trains.

.. code-block:: python

    self.input = spaic.Encoder(num=node_num, coding_method='poisson')

.. note::
    - For ``full connection`` , the initialization parameter **shape** may not be specified.
    - For ``convolution connection`` , the initialization parameter **shape** should be specified as [channel, width, height]. In this case, the initialization parameter **num** may not be specified.
    - For :code:`PoissonEncoding` , sometimes we need to scale the input intensity, which can be done by specifying the **unit_conversion** parameters:

        **unit_conversion** - a constant parameter that scales the input rate, default as 1.0


Latency ('latency')
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The stronger the external stimulus, the earlier the neurons fire. \
Taking the image input as an example, the larger the gray value in the image, the more important the information is and \
the earlier the firing time of the neuron.

The following code defines a ``Latency`` object which transforms the input into spike trains.

.. code-block:: python

    self.input = spaic.Encoder(num=node_num, coding_method='latency')

.. note::
    - For ``full connection`` , the initialization parameter **shape** may not be specified.
    - For ``convolution connection`` , the initialization parameter **shape** should be specified as [channel, width, height]. In this case, the initialization parameter **num** may not be specified.

NullEncoder ('null')
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If no encoding method is required, we can use ``NullEncoder`` .

The following code defines a ``NullEncoder`` object.

.. code-block:: python

    self.input = spaic.Encoder(num=node_num, coding_method='null')

.. note::
    - For ``full connection`` , the initialization parameter **shape** may not be specified.
    - For ``convolution connection`` , the initialization parameter **shape** should be specified as [channel, width, height]. In this case, the initialization parameter **num** may not be specified.
    - For ``full connection`` , the shape of external input should be [batch_size, time_step, node_num].
    - For ``convolution connection`` , the shape of external input should be [batch_size, time_step, channel, width, height].


Generator
------------------------------
The :code:`Generator` class is a subclass of the :code:`Node` class. \
It is a special encoder that will generate spike trains or current without dataset. \
For example, in some computational neuroscience studies, users need special input like poisson spikes to model background cortical activities.

To meet requirements, some common pattern generators are provided in **SPAIC** .

- **Poisson_Generator ('poisson_generator')** -- generate poisson spike trains according input rate
- **CC_Generator ('cc_generator')** -- generate constant current input

When instantiating an encoded class, users need to specify the **shape** or **num** , **coding_method** and other related paramters.

Poisson_Generator ('poisson_generator')
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``Poisson_Generator`` method generate spike trains according to input rate. \
At each time step, an uniform random number between 0 and 1 is generated and compared with input rate. \
If the random number is less than input rate, a spike is generated.

The following code defines a ``Poisson_Generator`` object which transforms the input rate into poisson spike trains.

.. code-block:: python

    self.input = spaic.Generator(num=node_num, coding_method='poisson_generator')

.. note::
    - For ``full connection`` , the initialization parameter **shape** may not be specified.
    - For ``convolution connection`` , the initialization parameter **shape** should be specified as [channel, width, height].
    - In this case, the initialization parameter **num** may not be specified.
    - If external input is a constant value, the input rate is the same for all nodes by default.
    - If each node needs a different input rate, you should pass in an input matrix corresponding to the shape of the node.
    - Sometimes we need to scale the input rate, which can be done by specifying the **unit_conversion** parameters:

        **unit_conversion** - a constant parameter that scales the input rate, default as 1.0.

CC_Generator ('cc_generator')
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``CC_Generator`` can generate constant current input, which is helpful for users to observe and simulate various neuronal dynamics. \
The ``CC_Generator`` is used similarly to ``Poisson_Generator`` , with **coding_method='cc_generator'** .

The following code defines a ``CC_Generator`` object which transforms the input rate into spike trains.

.. code-block:: python

    self.input = spaic.Generator(num=node_num, coding_method='cc_generator')

.. note::

    CC_Generator's precautions are similar to Poisson_Generator's.


Decoder
------------------------------
The :code:`Decoder` class is a subclass of the :code:`Node` class. \
The main usage of :code:`Decoder` is to convert the output spikes or voltages to a numerical signal. \
In **SPAIC** , we have built in some common decoding methods:

- **Spike_Counts ('spike_counts')** -- get the mean spike count of each neuron in the target layer.
- **First_Spike ('first_spike')** -- get the first firing time of each neuron in the target layer.
- **Final_Step_Voltage ('final_step_voltage')** -- get the final step voltage of each neuron in the target layer.
- **Voltage_Sum ('voltage_sum')** -- get the voltage sum of each neuron in the target layer.

The :code:`Decoder` class is mainly used in the output layer of the network. \
When instantiating an decoded class, users need to specify the **num** ,  **dec_target** , **coding_method** and related parameters.

For example, when decoding the spiking activity of a :code:`NeuronGroup` object with 10 LIF neurons, we can create an \
instance of the :code:`Spike_Counts` class:

.. code-block:: python

    self.target = spaic.NeuronGroup(num=10, model='lif')
    self.output = spaic.Decoder(num=10, dec_target=self.target, coding_method='spike_counts')

.. note::
    - The value of parameter **dec_target** is the layer to be decoded.
    - The value of parameter **num** in :code:`Decoder` class should be the same as the value of **num** in the target layer.
    - If you want to instantiate other decoding classes, simply assign str name of corresponding class to **coding_method** parameter.
    - The value of parameter **coding_var_name** is the variable to be decoded, such as 'O' or 'V'. 'O' represents spike and 'V' represents voltage.
    - For :code:`Spike_Counts` and :code:`First_Spike` , the default value of parameter **coding_var_name** is 'O'.
    - For :code:`Final_Step_Voltage` and :code:`Voltage_Sum` , the default value of parameter **coding_var_name** is 'V'.

For :code:`Spike_Counts`, we can specify **pop_size** parameter,

    - **pop_size** - population size of decoded neurons, default as 1 (each category is represented by one neuron)

Reward
------------------------------
The :code:`Reward` class is a subclass of the :code:`Node` class. \
It can be seen as a different type of decoder. \
During the execution of a reinforcement learning task, :code:`Reward` is needed to decode the activity of the target object according to the task purpose.

In **SPAIC** , we have built in some reward methods:

- **Global_Reward ('global_reward')** -- get a global reward. For the classification task, the predict label is determined according to the number of spikes or the maximum membrane potential. If the predict label is the same as the expected one, the positive reward will be returned. On the contrary, negative rewards will be returned.
- **XOR_Reward ('xor_reward')** -- get reward for xor task. When the expected result is 1, if the number of output spikes is greater than 0, a positive reward will be obtained. When the expected result is 0, if the number of output pulses is greater than 0, the penalty is obtained
- **DA_Reward ('da_reward')** -- get rewards in the same dimension as neurons in the dec_target
- **Environment_Reward ('environment_reward')** -- get reward from RL environment

The :code:`Reward` class is mainly used in the output layer of the network. \
When instantiating an reward class, users need to specify the **num** , **dec_target** , **coding_method** and other related parameters.

For example, when decoding the spiking activity of a :code:`NeuronGroup` object with 10 LIF neurons to obtain a \
global reward, we can create an instance of the :code:`Global_Reward` class as follows:

.. code-block:: python

    self.target = spaic.NeuronGroup(num=10, model='lif')
    self.reward = spaic.Reward(num=10, dec_target=self.target, coding_method='global_reward')

.. note::
    - The value of parameter **dec_target** is the layer to be decoded.
    - The value of parameter **num** in :code:`Reward` class should be the same as the value of **num** in the target layer.
    - If you want to instantiate other reward classes, simply assign str name of corresponding class to **coding_method** parameter.
    - The value of parameter **coding_var_name** is the variable to be decoded, such as 'O' or 'V'. 'O' represents spike and 'V' represents voltage.
    - The default value  is 'O'.

For :code:`Global_Reward` , :code:`XOR_Reward` and :code:`DA_reward` , we can specify some parameters:

    - **pop_size** - population size of decoded neurons, default as 1 (each category is represented by one neuron)
    - **dec_sample_step** - decoding sampling time step, default as 1 (get reward each time step)
    - **reward_signal** - reward, default as 1.0
    - **punish_signal** - punish, default as -1.0

Action
------------------------------
The :code:`Action` class is a subclass of the :code:`Node` class.\
It is also a special decoder that will transform the output to an action. The main usage \
of ``Action`` is to choose the next action according to the action selection mechanism of the target object \
during reinforcement learning tasks.

In **SPAIC** , we have built in some action methods:

- **Softmax_Action ('softmax_action')** -- action sampled from softmax over spiking activity of target layer.
- **PopulationRate_Action ('pop_rate_action')** -- take the label of the neuron group with largest spiking frequency as action.
- **Highest_Spikes_Action ('highest_spikes_action')** -- action sampled from highest activities of target layer.
- **Highest_Voltage_Action ('highest_voltage_action')** -- action sampled from highest voltage of target layer.
- **First_Spike_Action ('first_spike_action')** -- action sampled from first spike of target layer.
- **Random_Action ('random_action')** -- action sampled from action space randomly.

The :code:`Action` class is mainly used in the output layer of the network. \
When instantiating an action class, users need to specify the **num** , **dec_target** , **coding_method** and other related paramters.

For example, when decoding the spiking activity of a :code:`NeuronGroup` object with 10 LIF neurons to obtain next \
action, we can create an instance of the :code:`Softmax_Action` class as follows:

.. code-block:: python

    self.target = spaic.NeuronGroup(num=10, model='lif')
    self.reward = spaic.Action(num=10, dec_target=self.target, coding_method='softmax_action')

.. note::
    - The value of parameter **dec_target** is the layer to be decoded.
    - The value of parameter **num** in :code:`Action` class should be the same as the value of **num** in the target layer.
    - If you want to instantiate other action classes, simply assign str name of corresponding class to **coding_method** parameter.
    - The value of parameter **coding_var_name** is the variable to be decoded, such as 'O' or 'V'. 'O' represents spike and 'V' represents voltage.

For :code:`PopulationRate_Action`, we can specify **pop_size** parameters:

    - **pop_size** - population size of decoded neurons, default as 1 (each category is represented by one neuron)

