.. _my-custom-connection:



Custom synapse or connection model
=======================================
This chapter will introduce how to customize connections and synapse.

Customize connection
----------------------------------
``Connection`` is the basic structure of neuron network, it contains weight information. Different connection way will generate different \
spatially structure. To meet users' requirements, **SPAIC** has constructed many common connection methods. If users want to add some \
personalize connection, can follow the document or the format in :class:`spaic.Network.Connection`.


Initialize connection method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Custom connection method need to inherit :code:`Connection` class and modify the corresponding parameters. Use :code:`FullConnection` as example:

.. code-block:: python

    def __init__(self, pre, post, name=None, link_type=('full', 'sparse_connect', 'conv','...'),
                 syn_type=['basic_synapse'], max_delay=0, sparse_with_mask=False, pre_var_name='O', post_var_name='Isyn',
                 syn_kwargs=None, **kwargs):
        super(FullConnection, self).__init__(pre=pre, post=post, name=name,
                                             link_type=link_type, syn_type=syn_type, max_delay=max_delay,
                                             sparse_with_mask=sparse_with_mask,
                                             pre_var_name=pre_var_name, post_var_name=post_var_name, syn_kwargs=syn_kwargs, **kwargs)
        self.weight = kwargs.get('weight', None)
        self.w_std = kwargs.get('w_std', 0.05)
        self.w_mean = kwargs.get('w_mean', 0.005)
        self.w_max = kwargs.get('w_max', None)
        self.w_min = kwargs.get('w_min', None)

        self.is_parameter = kwargs.get('is_parameter', True) # is_parameter以及is_sparse为后端使用的参数，用于确认该连接是否为可训练的以及是否为稀疏化存储的
        self.is_sparse = kwargs.get('is_sparse', False)

In this initial way, the extra parameters should get from :code:`kwargs` .

Customize synapse model
----------------------------
To meet users requirements, **SPAIC** has constructed some common synapse model. But if users want to add some \
personalized model, they need to define synapse model as the format of :code:`Network.Synapse` .


Define parameters that can be obtained externally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the initial part of defining the neuron model, we need to define some parameters that the neuron model \
can change, which can be changed by passing parameters. For example,  in the first-order decay model of \
chemical synapses, the original formula can be obtained after transformation:

.. code-block:: python

    class First_order_chemical_synapse(SynapseModel):
        """
        .. math:: Isyn(t) = weight * e^{-t/tau}
        """

In this formula, :code:`self.tau` is changeable, so we can change it by :code:`kwargs` .

.. code-block:: python

    self._syn_tau_variables['tau[link]'] = kwargs.get('tau', 5.0)

Define variables
^^^^^^^^^^^^^^^^^^^^^^^^^^
In the variable definition stage, we need to understand several variable forms of synapses:

- **_syn_tau_constant_variables** -- Exponential decay constant
- **_syn_variables** -- Normal variable

To :code:`_syn_tau_constant_variables` , we will transmit it as :code:`value = np.exp(-self.dt / var)` ,

When defining variables, initial values need to be set at the same time. After each run of the network, \
the parameters of neurons will be reset to the initial values set at this point.


.. code-block:: python

    self._syn_variables[I] = 0
    self._syn_variables[WgtSum] = 0
    self._syn_tau_constant_variables[tauP] = self.tau_p


Define calculation operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The calculation operation is the most important part of the synaptic model. The calculation operation \
determines how the parameters will undergo some changes during the simulation.

There are a few rules to follow when adding computations. First, each row can only evaluate one specific \
operator, so you need to decompose the original formula into independent operators.  The current built-in \
operator in the platform can be found in :code:`backend.basic_operation` :

- add, minus, div
- var_mult, mat_mult, mat_mult_pre, sparse_mat_mult, reshape_mat_mult
- var_linear, mat_linear
- reduce_sum, mult_sum
- threshold
- cat
- exp
- stack
- conv_2d, conv_max_pool2d

Use the process of computing chemical current in chemical synapse as an example:

.. code-block:: python

    # Isyn = O * weight
    # The first is the result, conn.post_var_name
    # Compute operator `mat_mult_weight` at the second index
    # The third is the factor of the calculation, input_name and weight[link]
    # '[updated]' means the updated value of current calculation, temporary variables don't need
    self._syn_operations.append(
        [conn.post_var_name + '[post]', 'mat_mult_weight', self.input_name,
         'weight[link]'])

