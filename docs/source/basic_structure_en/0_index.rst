===================
Basic Structure
===================



Basic components
===================

:code:`Assembly` is the most important basic class in **SPAIC** . :code:`spaic.Assembly` contains three part: ``Network`` ,\
``NeuronGroup`` and ``Node`` . :code:`spaic.Network` is the basic class of the whole model. :code:`spaic.NeuronGroup` \
contains neurons. :code:`spaic.Node` is the basic class of the input and output nodes.

Front-end structure：

.. image:: ../_static/SPAIC_FRONTEND.jpg
    :width: 75%

Assembly
--------------------------
:code:`Assembly` is an abstract class of neural network structure, representing any network structure, and other network modules are \
subclasses of the ``Assembly`` . ``Assembly`` has two properties named :code:`_groups` and\
:code:`_connections` that save the neurons and connections. As the main interface for network model, it contains \
the following main functions:

    - **add_assembly(name, assembly)** -- Add new assembly members into the neuron assembly
    - **del_assembly(assembly=None, name=None)** -- Delete exist members from the neuron assembly
    - **copy_assembly(name, assembly)** -- copy an exist assembly and add it into this assemlby
    - **replace_assembly(old_assembly, new_assembly)** -- replace an exist member with a new assembly
    - **merge_assembly(assembly)** -- merge the given assembly into another assembly
    - **select_assembly(assemblies, name=None)** -- choose part of the target assembly as a new assembly
    - **add_connection(name, connection)** -- add a new connection into the assembly
    - **del_connection(connection=None, name=None)** -- delete the target connection
    - **assembly_hide()** -- hide this assembly
    - **assembly_show()** -- show this assembly



NeuronGroup
--------------------------
:code:`spaic.NeuronGroup` is the class with some neurons, usually, we call it a layer of neuron with same neuron model and connection ways. For more details, please look up :doc:`./1_neuron`


Node
---------------------------
:code:`spaic.Node` is the transform node of model, it contains a lot of encoding and decoding methods. :code:`Encoder` , :code:`Decoder` , \
:code:`Generator` , :code:`Action` and :code:`Reward` are all inherited from :code:`Node`.  For more details, please look up :doc:`./5_encode_decode`

Network
--------------------------
:code:`spaic.Network` is the most top structure in **SPAIC** , all modules like :code:`NeuronGroup` or :code:`Connection` should be contained in it. \
:code:`spaic.Network` also controls the training, simulation and some data interaction process. :code:`spaic.Network` supports some \
useful interface as follow:

    - **run(run_time)** -- run the model, run_time is the time window
    - **save_state** -- save the weight of model
    - **state_from_dict** -- load weight of model

Projection
-------------------------
:code:`spaic.Projection` is a high-level abstract class of topology, instead of :code:`Connection` , :code:`Project` represent \
the connections between :code:`Assembly` s. :code:`Connection` is subclass of :code:`Projection` , when user build :code:`Projection` \
between :code:`Assembly` s, the construct function will build the corresponding connections according to the :code:`policies` of \
:code:`Projection`

Connection
--------------------------
:code:`spaic.Connection` is used to build connections between :code:`NeuronGroup` s, it also used to construct different synapses. For more details, please look up  :doc:`./2_connection`

Backend
--------------------------
The core of backend that construct variables and generate computational graph. For more details, please look up  :doc:`../backend_en` 。

More details
======================
.. toctree::
   :maxdepth: 1
   :titlesonly:

   1_neuron
   2_connection
   3_synaptic
   4_algorithm
   5_encode_decode
   6_network_en