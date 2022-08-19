.. _my-custom-encoding:

Custom encoding or decoding
=================================
This chapter will introduce how to customize :code:`Encoder` , :code:`Generator` , :code:`Decoder` , :code:`Action` and :code:`Reward` .

Customize Encoder
---------------------------
Encoder is used to transmit the input data to temporal spiking data. It is one of the important step in building spiking \
neural network. Different encoding method will generate different data. To meet most of the application situation, **SPAIC** \
has already provided some common encoding methods. And customize encoding method can add as the format in :class:`spaic.Neuron.Encoders` .

Initialize Encoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The user-defined encoding method should inherit the class :code:`Encoder` , and the parameter name in the initialization method \
should be the same as that of the class :code:`Encoder` . Other parameters can be passed in by **kwargs** . \
Take :code:`PoissonEncoding` class initialization function as an example:

.. code-block:: python

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method='poisson',
             coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(PoissonEncoding, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type,
                                          **kwargs)
        self.unit_conversion = kwargs.get('unit_conversion', 1.0)


In this initialization method, **unit_conversion** is the required parameter for the :code:`PoissonEncoding` class,
which can get from **kwargs** .

Define Encoder Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The encoding function is the implementation part of the encoding method. \
Because the platform supports multiple backends ( :code:`Pytorch` , :code:`TensorFlow` etc.), different backends \
support different data types and data operations. \
Therefore, the corresponding coding function needs to be implemented in the front-end coding method for different computing \
back-end. \
We take the implementation of :code:`torch_coding` for :code:`PoissonEncoding` as an example:

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

At the end of this code, don't forget add :code:`Encoder.register("poisson", PoissonEncoding)` to add the usage linked to \
this function.

Customize Generator
--------------------------
Generator can be used to generate specific distributed spike trains or some special current mode. **SPAIC** \
has already provided some common generating methods. \
And customize generating method can add as the format in :class:`spaic.Neuron.Generators` file.

Initialize Generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The user-defined generating method should inherit the class :code:`Generator` , and the parameter name in the initialization method \
should be the same as that of the class :code:`Generator` . Other parameters can be passed in by **kwargs** . \
Take :code:`CC_Generator` class initialization function as an example:

.. code-block:: python

    def __init__(self, shape=None, num=None, dec_target=None,  dt=None,
                 coding_method='cc_generator', coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(CC_Generator, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type,
                                       **kwargs)



Define Generator Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A coding function is the implementation part of a generating method. \
Because the platform supports multiple backends ( :code:`Pytorch` , :code:`TensorFlow` etc.), different backends \
support different data types and data operations. \
Therefore, the corresponding coding function needs to be implemented in the front-end coding method for different computing \
back-end. \
We take the implementation of :code:`torch_coding` for :code:`CC_Generator` as an example:

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

:code:`Generator.register('cc_generator', CC_Generator)` also needed here for front-end use.

Customize Decoder
--------------------------
Decoder is used to convert the output spikes or voltages to a numerical signal. **SPAIC** \
has already provided some common decoding methods. \
And decoding method can add as the format in :class:`spaic.Neuron.Decoders` file.

Initialize Decoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The user-defined decoding method should inherit the class :code:`Decoder`, and the parameter name in the initialization method \
should be the same as that of the class :code:`Decoder`. Other parameters can be passed in by **kwargs** . \
Take :code:`Spike_Counts` class initialization function as an example:

.. code-block:: python

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method='spike_counts',
            coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Spike_Counts, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type,
                                      **kwargs)
        self.pop_size = kwargs.get('pop_size', 1)

In this initialization method, **pop_size** is the required parameter for the :code:`Spike_Counts` class, \
which can get from **kwargs** .

Define Decoder Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A coding function is the implementation part of a decoding method. \
Because the platform supports multiple backends ( :code:`Pytorch` , :code:`TensorFlow` etc.), different backends \
support different data types and data operations. \
Therefore, the corresponding coding function needs to be implemented in the front-end coding method for different computing \
back-end. \
We take the implementation of :code:`torch_coding` for :code:`Spike_Counts` as an example:

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

:code:`Decoder.register('spike_counts', Spike_Counts)` also needed here for front-end use.


Customize Reward
--------------------------
Reward is used to convert the activity of the target object into reward signal. **SPAIC** \
has already provided some common reward methods. \
And reward method can add as the format in :class:`spaic.Neuron.Rewards` file.

Initialize Reward
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The user-defined reward method should inherit the class :code:`Reward` , and the parameter name in the initialization method \
should be the same as that of the class :code:`Reward` . Other parameters can be passed in by **kwargs** . \
Take :code:`Global_Reward` class initialization function as an example:

.. code-block:: python

    def __init__(self,shape=None, num=None, dec_target=None, dt=None, coding_method='global_reward', coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Global_Reward, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.pop_size = kwargs.get('pop_size', 1)
        self.reward_signal = kwargs.get('reward_signal', 1)
        self.punish_signal = kwargs.get('punish_signal', -1)

In this initialization method, **pop_size** , **reward_signal** , **punish_signal** are required parameters for the :code:`Global_Reward` class, \
which can get from **kwargs** .

Define Reward Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A coding function is the implementation part of a reward method. \
Because the platform supports multiple backends ( :code:`Pytorch` , :code:`TensorFlow` etc.), different backends \
support different data types and data operations. \
Therefore, the corresponding coding function needs to be implemented in the front-end coding method for different computing \
back-end. \
We take the implementation of :code:`torch_coding` for :code:`Global_Reward` as an example:

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

:code:`Reward.register('global_reward', Global_Reward)` also needed here for front-end use.

Customize Action
--------------------------
Action is used to convert the activity of the target object into next action. **SPAIC** \
has already provided some common action methods. \
And action method can add as the format in :class:`spaic.Neuron.Actions` file.

Initialize Action
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The user-defined action method should inherit the class :code:`Action` , and the parameter name in the initialization method \
should be the same as that of the class :code:`Action` . Other parameters can be passed in by **kwargs** . \
Take :code:`Softmax_Action` class initialization function as an example:

.. code-block:: python

    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method='softmax_action', coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(Softmax_Action, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)

Define Action Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A coding function is the implementation part of a action method. \
Because the platform supports multiple backends ( :code:`Pytorch` , :code:`TensorFlow` etc.), different backends \
support different data types and data operations. \
Therefore, the corresponding coding function needs to be implemented in the front-end coding method for different computing \
back-end. \
We take the implementation of :code:`torch_coding` for :code:`Softmax_Action` as an example:

.. code-block:: python

    def torch_coding(self, record, target, device):
        # the shape of record is (time_step, batch_size, n_neurons)
        assert (
            record.shape[2] == self.num
        ), "Output layer size is not equal to the size of the action space."
        spikes = torch.sum(record, dim=0)
        probabilities = torch.softmax(spikes, dim=0)
        return torch.multinomial(probabilities, num_samples=1).item()


:code:`Action.register('softmax_action', Softmax_Action)` also needed here for front-end use.