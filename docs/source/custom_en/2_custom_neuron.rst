.. _my-custom-neuron-en:


Custom Neuron Model
=======================
Neuron model is the most important part in neural dynamics simulation. \
Different models and parameters will produce different phenomena. **SPAIC** includes \
many common neuron models in order to meet the needs of different applications. However, **SPAIC** \
is sometimes out of reach and users need to add their own personalized neurons \
that are more appropriate for their experiments. The neuron definition step can follow the \
the format in :code:`spaic.Neuron` .


Define Variables
-----------------------------
At first, we need to introduce some typical variable type in **SPAIC** :

- **_variables** -- normal variable
- **_tau_variables** -- exponential decay time constant
- **_membrane_variables** -- decay time constant
- **_parameter_variables** -- parameters
- **_constant_variables** -- constant

To :code:`_tau_variables` , we will transmit it as :code:`tau_var = np.exp(-dt/tau_var)` .
To :code:`_membrane_variables` , we will transmit it as :code:`membrane_tau_var = dt/membrane_tau_var` ,

When defining variables, initial value also should be given, since all the neuron parameters will be reset to the initial value \
after each model run. Some parameters can be change based on the parameters received.  We use :code:`lif` neuron model as the example:

.. code-block:: python

    """
    LIF model:
    # V(t) = tuaM * V^n[t-1] + Isyn[t]   # tauM: constant membrane time (tauM=RmCm)
    O^n[t] = spike_func(V^n[t-1])
    """

In the formula of :code:`lif` model, the original formula should be transmitted to the differential equation by users themselves. \
Then, :code:`tauM` and the threshold :code:`v_th` are changeable, so we get parameters from :code:`kwargs`:

.. code-block:: python

    # The complete add code
    self._variables['V'] = 0.0
    self._variables['O'] = 0.0
    self._variables['Isyn'] = 0.0

    self._parameter_variables['Vth'] = kwargs.get('v_th', 1)
    self._constant_variables['Vreset'] = kwargs.get('v_reset', 0.0)

    self._tau_variables['tauM'] = kwargs.get('tau_m', 20.0)



Define Calculation
-----------------------
Compute operation is the most important part of Neuron Model. These operations decide the change of elements during simulation. \
When add compute operations, there are some rules to follow. At first, every operation can only do one compute process, so users \
need to decomposition formula to independent operations. The whole build-in calculate operator can be found \
in :class:`spaic.backend.backend` , and here is the example about :code:`LIF` model:

.. code-block:: python

    # Recently, [updated] represent that here it needs the updated value instead of the old value from last round of
    # calculation. Temporary variable don't need [updated].
    # Vtemp = V * tauM + I, to be mentioned, tauM is a '_tau_variables' variable, which means it was not the initial value.
    self._operations.append(('Vtemp', 'var_linear', 'tauM', 'V', 'Isyn[updated]'))

    # O = 1 if Vtemp >= Vth else 0， 'threshold' used to check whether 'Vtemp' reaches the threshold 'Vth'
    self._operations.append(('O', 'threshold', 'Vtemp', 'Vth'))

    # Used to reset voltage after spike is sent
    self._operations.append(('V', 'reset', 'Vtemp',  'O[updated]'))

Also, we need to add :code:`NeuronModel.register("lif", LIFModel)` to combine the name with the model for front-end use.