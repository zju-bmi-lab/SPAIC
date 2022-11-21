Connection
================

This chapter will introduce :code:`spaic.Connection` and how to use different connections in **SPAIC** . \
As the most basic component of spiking neuron network, :code:`spaic.Connection` contains the most important weight information of \
the model. At the same time, as a brain-inspired platform, **SPAIC** supports bionic link which means supports feedback \
connections, synaptic delay or other connections with physiological properties.

Connect Parameters
------------------------------

.. code-block:: python

    def __init__(self, pre: Assembly, post: Assembly, name=None,
            link_type=('full', 'sparse_connect', 'conv', '...'), syn_type=['basic'],
            max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn',
            syn_kwargs=None, **kwargs):

In the initial parameters of connection, we can see that when we construct connections, :code:`pre` , \
:code:`post` and :code:`link_type` are requisite.

- **pre** - presynaptic neuron
- **post** - postsynaptic neuron
- **name** - name of the connection, make connections easier to distinguish
- **link_type** - link type, 'full connection', 'sparse connection' or 'convolution connection', etc.
- **syn_type** - synapse type, it will be further explanation in the synaptic section
- **max_delay** - the maximum sypatic delay
- **sparse_with_mask** - whether use mask in sparse connection
- **pre_var_name** - the signal variable name of presynaptic neuron, default as ``O`` , means output spike
- **post_var_name** - the signal variable name of postsynaptic neuron, default as ``Isyn`` , means synaptic current
- **syn_kwargs** - the custom parameters of synapse, it will be further explanation in the synaptic section
- **\**kwargs** - some typical parameters are included in ``kwargs`` .

Despite these initial parameters, there are still some important parameters about weight:

- **w_mean** - mean value of weight
- **w_std** - standard deviation of weight
- **w_max** - maximum value of weight
- **w_min** - minimum value of weight
- **weight** - weight

**SPAIC** will generate weight randomly if users don't provide weight. :code:`w_mean` and :code:`w_std` will be used \
to generate the weight. **SPAIC** will clamp the weight if :code:`w_min` or :code:`w_max` is offered.

For example, in :code:`conn1_example`, the connection will generate weight with mean 1 and standard deviation of 5, \
and clip weights between 0 and 2.

.. code-block:: python

    self.conn1_example = spaic.Connection(self.layer1, self.layer2, link_type='full',
                                    w_mean=1.0, w_std=5.0, w_min=0.0, w_max=2.0)

Full Connection
---------------------
``Full connection`` is one of the basic connection type.

.. code-block:: python

    self.conn1_full = spaic.Connection(self.layer1, self.layer2, link_type='full')

Important key parameters of full connection:

.. code-block:: python

    self.weight = kwargs.get('weight', None) # weight, if not given, it will generate randomly
    self.mask = kwargs.get('mask', None) #
    self.w_std = kwargs.get('w_std', 0.05) # standard deviation of weight, used to generate weight
    self.w_mean = kwargs.get('w_mean', 0.005) # mean value of weight, used to generate weight
    self.w_max = kwargs.get('w_max', None) # maximum value of weight
    self.w_min = kwargs.get('w_min', None) # minimum value of weight

One-to-one Connection
--------------------------------
There are two kinds of one to one connection in **SPAIC**, the basic ``one_to_one`` and the sparse ``one_to_one_sparse``

.. code-block:: python

    self.conn_1to1 = spaic.Connection(self.layer1, self.layer2, link_type='one_to_one')
    self.conn_1to1s = spaic.Connection(self.layer1, self.layer2, link_type='one_to_one_sparse')

Important key parameters of one to one connection:

.. code-block:: python

    self.w_std = kwargs.get('w_std', 0.05) # standard deviation of weight, used to generate weight


Convolution Connection
--------------------------------
Common ``convolution connection``, pooling method can choose :code:`avgpool` or :code:`maxpool` in synapse type.

.. note::
    In order to provide better computational support, convolution connections need to be used with convolution synapses.


Main connection parameters in convolution connection:

.. code-block:: python

        self.out_channels = kwargs.get('out_channels', 4)  # input channel
        self.in_channels = kwargs.get('in_channels', 1)    # output channel
        self.kernel_size = kwargs.get('kernel_size', [3, 3]) # convolution kernel
        self.w_std = kwargs.get('w_std', 0.05) # standard deviation of weight, used to generate weight
        self.w_mean = kwargs.get('w_mean', 0.05) # mean value of weight, used to generate weight
        weight = kwargs.get('weight', None) # weight, if not given, connection will generate randomly

        self.is_parameter = kwargs.get('is_parameter', True)
        self.is_sparse = kwargs.get('is_sparse', False)
        self.mask = kwargs.get('mask', None)
        self.stride = kwargs.get('stride', 1)
        self.padding = kwargs.get('padding', 0)
        self.dilation = kwargs.get('dilation', 1)
        self.groups = kwargs.get('groups', 1)

Convolution connection example 1:

.. code-block:: python

        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='conv', syn_type=['conv'],
                                                in_channels=1, out_channels=4,
                                                kernel_size=(3, 3),
                                                init='uniform',
                                                init_param={'a':-math.sqrt(1/(9)), 'b':math.sqrt(1/(9))})

        self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='conv', syn_type=['conv'],
                                              in_channels=4, out_channels=8, kernel_size=(3, 3),
                                              init='uniform', init_param={'a':-math.sqrt(1/(8*9)), 'b':math.sqrt(1/(8*9))})

        self.connection3 = spaic.Connection(self.layer2, self.layer3, link_type='full',
                                              syn_type=['flatten', 'basic'],
                                              init='kaiming_normal', init_param={'a': math.sqrt(5)})


Convolution connection example 2:

.. code-block:: python

        self.conv2 = spaic.Connection(self.layer1, self.layer2, link_type='conv',
                                        syn_type=['conv', 'dropout'], in_channels=128, out_channels=256,
                                        kernel_size=(3, 3), stride=args.stride, padding=args.padding, init='uniform',
                                        init_param=(-math.sqrt(1/(128*3*3)), math.sqrt(1/(128*9))), bias=args.bias)
        self.conv3 = spaic.Connection(self.layer2, self.layer3, link_type='conv',
                                        syn_type=['conv', 'maxpool', 'dropout'], in_channels=256, out_channels=512,
                                        kernel_size=(3, 3), stride=args.stride, padding=args.padding,
                                        pool_stride=2, pool_padding=0, init='uniform',
                                        init_param=(-math.sqrt(1/(256*9)), math.sqrt(1/(256*9))), bias=args.bias)
        self.conv4 = spaic.Connection(self.layer3, self.layer4, link_type='conv',
                                        syn_type=['conv', 'maxpool', 'dropout'], in_channels=512, out_channels=1024,
                                        kernel_size=(3, 3), stride=args.stride, padding=args.padding,
                                        pool_stride=2, pool_padding=0, init='uniform',
                                        init_param=(-math.sqrt(1/(512*9)), math.sqrt(1/(512*9))),
                                        syn_kwargs=[], bias=args.bias)


Sparse Connection
----------------------
Common ``sparse connection``, set the density of connection with parameter :code:`density` .

Random Connection
---------------------------
Common ``random connection``, set the connection probability with parameter :code:`probability` .








