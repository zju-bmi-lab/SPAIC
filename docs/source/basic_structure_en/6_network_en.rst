Network
=====================

This section mainly introduces the methods for building and running networks based on the ``Network`` class in the **SPAIC** platform.

Network Construction
-----------------------------------
Model construction can use three methods: The first is similar to Pytorch's module class inheritance, in the form of building in the _init_ function; the second is similar to Nengo's method of construction using the with statement; the third way, \
new network modules can also be added to the existing network during the modeling process through the function interfaces.

Model Construction Method 1: Class Inheritance Form

.. code-block:: python

   class SampleNet(spaic.Network):
      def __init__(self):
         super(SampleNet, self).__init__()

         self.layer1 = spaic.NeuronGroup(100, neuron_model='clif')
         ......

   Net = SampleNet()


Model Construction Method 2: with Form

.. code-block:: python

     Net = SampleNet()
     with Net:

        layer1 = spaic.NeuronGroup(100, neuron_model='clif')
        ......

Model Construction Method 3: Build or Modify Network Through Function Interface

.. code-block:: python

    Net = SampleNet()
    layer1 = spaic.NeuronGroup(100, neuron_model='clif')
    ....

    Net.add_assembly('layer1', layer1)

The current ``Network`` provides function interfaces for constructing or modifying a network, including:

- **add_assembly(name, assembly)** - Add a neural assembly class (including ``Node``, ``NeuronGroup``, etc.), where the parameter name represents the variable name in the network, and assembly represents the neural assembly object to be added.
- **copy_assembly(name, assembly)** - Copy a neural assembly class (including ``Node``, ``NeuronGroup``, etc.). Unlike add_assembly, this interface clones the assembly object before adding it to the network.
- **add_connection(name, connection)** - Add a connection object, where the parameter name represents the variable name in the network, and connection represents the connection object to be added.
- **add_projection(name, projection)** - Add a topology projection object, where the parameter name represents the variable name in the network, and projection represents the topology mapping object to be added.
- **add_learner(name, learner)** - Add a learning algorithm, where the parameter name represents the variable name in the network, and learner represents the learning algorithm object to be added.
- **add_moitor(name, moitor)** - Add a monitor, where the parameter name represents the variable name in the network, and moitor represents the monitor object to be added.


Network Execution and Execution Parameter Settings
------------------------------
The ``Network`` object provides function interfaces for running and setting execution parameters, including:

- **run(backend_time)** - The function interface for running the network, where the backend_time parameter is the network runtime.
- **run_continue(backend_time)** - The function interface for continuing to run the network. Unlike run, run_continue does not reset the initial values of variables but continues to run based on the original initial values.
- **set_backend(backend, device, partition)** - Set the network runtime backend, where the parameters backend represents the backend object or backend name, device is the hardware used for backend computation, and partition represents whether to distribute the model across different devices.
- **set_backend_dt(backend, dt)** - Set the network runtime timestep, where dt is the timestep.
- **set_random_seed(seed)** - Set the network runtime random seed.