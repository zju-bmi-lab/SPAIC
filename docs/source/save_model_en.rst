save or load model
=====================

This section will describe two ways of saving network information in detail.

pre-defined function in Network
---------------------------------------------------------
Use pre-defined functions :code:`save_state` and :code:`state_from_dict` of ``spaic.Network`` to save or load the weight of the model directly.

The optional parameters are :code:`filename` , :code:`direct` and :code:`save`. If users use :code:`save_state` without \
giving any parameters, the function will use default name :code:`autoname` with random number as the direct name and save \
the weight into the ``'./autoname/parameters/_parameters_dict.pt'`` . If given :code:`filename`, or :code:`direct` , it will \
save the weight into ``'direct/filename/parameters/_parameters_dict.pt'`` . Parameter :code:`save` is default as ``True`` , which \
means it will save the weight. If users choose ``False`` , this function will return the :code:`parameter_dict` of the model \
directly.

The parameters of :code:`state_from_dict` is same as :code:`save_state` but have two more parameters: :code:`state` and :code:`direct` ,\
and :code:`save` parameters is unneeded. If users provide :code:`state` , this function will use given parameters to replace the parameter dict \
of the backend. If :code:`state` is None, this function will decide the saving path according to :code:`filename` and :code:`direct`. The \
:code:`device` will decide where to storage the parameters.

.. code-block:: python

    Net.save_state('Test1', True)
    ...
    Net.state_from_dict(filename='Test1', device=device)


network_save and network_load
---------------------------------------------------------------------------------------------------------------------------------------
The network save module :code:`spaic.Network_saver.network_save` and :code:`spaic.Network_loader.network_load` in `spaic.Library` \
will save the whole network structure of the model and the weight information separately. This method requires a filename \
``filename`` when used, and then the platform will create a new file ``./filename/filename.json`` in the running directory \
of the current program to save the network structure. At the same time, when using :code:`network_save` , users also can choose the \
save format between ``json`` or ``yaml`` .

.. code-block:: python

    network_dir = network_save(Net=Net, filename='TestNet',
                                            trans_format='json', combine=False, save=True)

    # network_dir = 'TestNet'
    Net2 = network_load(network_dir, device=device)

In :code:`network_save` :

- **Net** -- the specific network object in **SPAIC**
- **filename** -- filename, ``network_save`` will save the ``Net`` with this name
- **path** -- file storage path, a new folder will be created based on the filename if target path doesn't have such folder
- **trans_format** -- save format, can choose ``json`` or ``yaml`` , default as ``json``
- **combine** -- this parameters decides whether save the weight and network structure in one file, default as ``False``
- **save** -- this parameters decides whether save the structure locally, if choose ``True`` , this function will save locally and return the file name. If choose ``False`` , it will only return the structure as a dict.
- **save_weight** -- this parameters decides whether save the backend information and weights of the model

During the process of storing the parameters of parts of the network, if the parameters of the neurons are passed in as Tensor, the names of these parameters are stored in the storage file and the actual parameters are stored in the diff_para_dict.pt file in the same directory as the weights.

Then, I will give some example to explain the meaning of saved file:

.. code-block:: python

    # information about Nodes
    -   input:
            _class_label: <nod> # Indicate this object is node
            _dt: 0.1 # Length of every time step
            _time: null #
            coding_method: poisson # Encode method
            coding_var_name: O # Output target of this node
            dec_target: null # Decode target of this node, since this is input node, it doesn't have decode target
            name: input # name of this node
            num: 784 # element number of this node
            shape: # shape
            - 784

    # information about NeuronGroups
    -   layer1:
            _class_label: <neg> # Indicate this object is NeuronGroup
            id: autoname1<net>_layer1<neg> # ID of this NeuronGroup, it is NeuronGroup 'layer1' of the network 'autoname1'
            model_name: clif # neuron model of this NeuronGroup, it's CLIF
            name: layer1 # name of this NeuronGroup
            num: 10 # neuron number of this NeuronGroup
            parameters: {} # parameters of kwargs, like some parameters of neuron model
            shape: # shape
            - 10
            type: null # type of this NeuronGroup, it is just like a label for Projection

    -   layer3:
        -   layer1:
                _class_label: <neg> # Indicate this object is NeuronGroup
                id: autoname1<net>_layer3<asb>_layer1<neg>  # ID of this NeuronGroup，it is NeuronGroup 'layer1' of the Assembly 'layer3' of the network 'autoname1'
                model_name: clif # neuron model of this NeuronGroup, it's CLIF
                name: layer1 # name of this NeuronGroup
                num: 10 # neuron number of this NeuronGroup
                parameters: {} # parameters of kwargs, like some parameters of neuron model
                shape: # shape
                - 10
                type: null # type of this NeuronGroup, it is just like a label for Projection

        -   connection0:
                _class_label: <con> # Indicate this object is Connection
                link_type: full # link type of this Connection, it is full connection
                max_delay: 0 # the maximum delay step of this Connection
                name: connection0 # name of this Connection
                parameters: {} # parameters of kwargs, like some parameters of convolution connection
                post: layer3   # postsynaptic neuron, here is point to Assembly layer3
                post_var_name: Isyn   # the output of this synapse, here is 'Isyn', a default value
                pre: layer2    # presynaptic neuron, here is point to layer2
                pre_var_name: O         # input of this synapse, here is 'O', a default value
                sparse_with_mask: false # whether use mask, details will be explained in chapter 'Basic Structure.Connection'
                weight: # weight matrix
                    autoname1<net>_layer3<asb>_connection0<con>:autoname1<net>_layer3<asb>_layer3<neg><-autoname1<net>_layer3<asb>_layer2<neg>:{weight}: # here is the ID of this weight
                    -   - 0.05063159018754959

    # information about Connections
    -   connection1:
            _class_label: <con> # Indicate this object is Connection
            link_type: full # link type of this Connection, it is full connection
            max_delay: 0 # the maximum delay step of this Connection
            name: connection1 # name of this Connection
            parameters:  # parameters of kwargs, like some parameters of convolution connection, here is the parameter for randomly initializing the weight
                w_mean: 0.02
                w_std: 0.05
            post: layer1   # postsynaptic neuron, here is point to layer1
            post_var_name: Isyn   # the output of this synapse, here is 'Isyn', a default value
            pre: input     # presynaptic neuron, here is point to input node
            pre_var_name: O         # input of this synapse, here is 'O', a default value
            sparse_with_mask: false # whether use mask, details will be explained in chapter 'Basic Structure.Connection'
            weight: # weight matrix
                autoname1<net>_connection1<con>:autoname1<net>_layer1<neg><-autoname1<net>_input<nod>:{weight}:
                -   - 0.05063159018754959
                    ......

    # information about Learners
    -   learner2:
            _class_label: <learner> # Indicate this object is Learner
            algorithm: full_online_STDP # the algorithms of this Learner, here is full_online_STDP
            lr_schedule_name: null # the learning rate scheduler of this Learner, here is unused
            name: _learner2 # name of this Learner
            optim_name: null # the optimizer of this Learner, here is unused
            parameters: {} # parameters of kwargs
            trainable: # the training target of this Learner
            - connection1
            - connection2

