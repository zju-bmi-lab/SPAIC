Neuron
=====================

This chapter introduces how to choose neuron model and change some important parameters of the model.

neuron model
----------------
Neuron model is one of the most important component of the model. Different neuron model will have different \
neuron dynamics. In spiking neuron network, people always convert the change of membrane potential of neuron model \
into different equation and approximate it by difference equation. Finally, obtain the differential neuron model \
that can be computed by computer. In SPAIC, we contains most of the common neuron models:

- LIF - Leaky Integrate-and-Fire model
- CLIF - Current Leaky Integrate-and-Fire model
- GLIF - Generalized Leaky Integrate-and-Fire model
- aEIF - Adaptive Exponential Integrate-and-Fire model
- IZH - Izhikevich model
- HH - Hodgkin-Huxley model

In SPAIC, :code:`NeuronGroup` is like nodes of the network model. Like layers in PyTorch, in SPAIC, \
NeuronGroup is the layer. Users need to specify the neuron numbers, neuron model or other related paramters. \

.. code-block:: python

    from spaic import NeuronGroup


LIF neuron model
-----------------------
For example, we build a layer with 100 LIF neurons:

.. code-block:: python

    self.layer1 = NeuronGroup(neuron_number=100, neuron_model='lif')


A layer with 100 standard LIF neurons has been constructed. While, sometimes we need to specify the \
LIF neuron to get different neuron dynamics, that we will need to specify some parameters:

- tau_p, tau_q - time constants of synapse, default as 4.0 and 1.0
- tau_m - time constant of neuron membrane potential, default as 6.0
- v_th - the threshold voltage of a neuron, default as 1.0
- v_reset - the reset voltage of the neuron, which defaults to 0.0

If users need to change these parameters, they can enter the parameters when construct NeuronGroups.

.. code-block:: python

    self.layer2 = NeuronGroup(neuron_number=100, neuron_model='lif',
                    tau_p=1.0, tau_q=1.0, tau_m=10.0, v_th=10, v_reset=0.2)


CLIF neuron model
-------------------------
CLIF(Current Leaky Integrated-and-Fire Model) neuron paramters:

- tau_p, tau_q - time constants of synapse, default as 12.0 and 8.0
- tau_m - time constant of neuron membrane potential, default as 20.0
- v_th - the threshold voltage of a neuron, default as 1.0

GLIF neuron model
-------------------------
GLIF(Generalized Leaky Integrate-and-Fire Model) [#f1]_ neuron paramters:

- R, C, E_L
- Theta_inf
- f_v
- delta_v
- b_s
- delta_Theta_s
- k_1, k_2
- delta_I1, delta_i2
- a_v, b_v
- tau_p, tau_q

aEIF neuron model
-------------------------
aEIF(Adaptive Exponential Integrated-and-Fire Model) [#f2]_ neuron paramters:

- tau_p, tau_q, tau_w, tau_m
- a, b
- delta_t, delta_t2
- EL

IZH neuron model
--------------------------
IZH(Izhikevich Model) neuron paramters:
- tau_p, tau_q
- a, b
- Vrest, Ureset

HH neuron model
--------------------------
HH(Hodgkin-Huxley Model) neuron paramters
- dt
- g_NA, g_K, g_L
- E_NA, E_K, E_L
- alpha_m1, alpha_m2, alpha_m3
- beta_m1, beta_m2, beta_m3
- alpha_n1, alpha_n2, alpha_n3
- beta_n1, beta_n2, beta_n3
- alpha_h1, alpha_h2, alpha_h3
- beta_1, beta_h2, beta_h3
- V65
- m, n, h
- V, vth


customize
----------------
In the following chapter called  :ref:`my-custom-neuron` , we will talke about how to add custom neuron model \
into SPAIC with more details.



.. [#f1] GLIF model. Mihalaş S, Niebur E. A generalized linear integrate-and-fire neural model produces diverse spiking behaviors. Neural Comput. 2009 Mar;21(3):704-18.` doi:10.1162/neco.2008.12-07-680. <https://doi.org/10.1162/neco.2008.12-07-680>`_ . PMID: 18928368; PMCID: PMC2954058.
.. [#f2] AEIF model. Brette, Romain & Gerstner, Wulfram. (2005). Adaptive Exponential Integrate-And-Fire Model As An Effective Description Of Neuronal Activity. Journal of neurophysiology. 94. 3637-42.` doi:10.1152/jn.00686.2005. <https://doi.org/10.1152/jn.00686.2005>`_


