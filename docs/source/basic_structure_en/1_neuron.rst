Neuron
=====================

本章节主要介绍在训练以及仿真中如何选择神经元模型，以及如何根据需求在原模型的基础上更改一些重要的参数。

脉冲神经元模型
----------------
神经元模型是脉冲神经网络中极为重要的一个组成部分，不同的神经元模型通常代表了对不同的神\
经元动力学的仿真与模拟。在脉冲神经网络中，我们通常将神经元模型关于电压的变化特征化为微\
分方程，再由差分方程来对其进行逼近，最后获得了计算机可以进行运算的神经元模型。在\
SPAIC中，我们包含了大多数较为常见的神经元模型：

- LIF - Leaky Integrate-and-Fire models
- CLIF - Current Leaky Integrate-and-Fire model
- GLIF - Generalized Leaky Integrate-and-Fire model
- aEIF - Adaptive Exponential Integrate-and-Fire model
- IZH - Izhikevich model
- HH - Hodgkin-Huxley model

在SPAIC中，NeuronGroup是作为网络节点的组成，如同Pytorch中的layer，SPAIC\
中的每个layer都是一个NeuronGroup，用户需要根据自己的需要指定在这个NeuronGroup中\
所包含的神经元数量、神经元类型、神经元的位置、神经元类型及与其类型相关的参数等。首先需\
要的就是导入NeuronGroup库：

.. code-block:: python

    from spaic import NeuronGroup


LIF神经元
------------------
以建立一层含有100个LIF神经元的layer为例:

.. code-block:: python

    self.layer1 = NeuronGroup(neuron_number=100, neuron_model='lif')


一个含有100个标准LIF神经元的layer就建立好了。然而许多时候我们需要按需定制不同的LIF\
神经元以获得不同的神经元的表现，这时候就需要在建立NeuronGroup时，指定一些参数：

- tau_p, tau_q - 突触的时间常量，默认为4.0和1.0
- tau_m - 神经元膜电位的时间常量，默认为6.0
- v_th - 神经元的阈值电压，默认为1.0
- v_reset - 神经元的重置电压，默认为0.0，因为平台内置的LIF模型的电压稳定点为0.0

如果用户需要调整这些变量，可以在建立NeuronGroup的时候输入想改变的参数即可：

.. code-block:: python

    self.layer2 = NeuronGroup(neuron_number=100, neuron_model='lif',
                    tau_p=1.0, tau_q=1.0, tau_m=10.0, v_th=10, v_reset=0.2)


这样，一个自定义参数的LIF神经元就建好了。

CLIF神经元
-------------------------
CLIF(Current Leaky Integrated-and-Fire Model)神经元的参数:

- tau_p, tau_q - 突触的时间常量，默认为12.0和8.0
- tau_m - 神经元膜电位的时间常量，默认为20.0
- v_th - 神经元的阈值电压，默认为1.0

GLIF神经元
-------------------------
GLIF(Generalized Leaky Integrate-and-Fire Model) [#f1]_ 神经元参数:

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

aEIF神经元
-------------------------
aEIF(Adaptive Exponential Integrated-and-Fire Model) [#f2]_ 神经元参数:

- tau_p, tau_q, tau_w, tau_m
- a, b
- delta_t, delta_t2
- EL

IZH神经元
--------------------------
IZH(Izhikevich Model) 神经元参数:
- tau_p, tau_q
- a, b
- Vrest, Ureset

HH神经元
--------------------------

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


自定义
----------------
在稍后的 :ref:`my-custom-neuron` 这一章节中，我们会更加详细具体地讲述该如何在我们平台上添加自定义的神\
经元模型。



.. [#f1] GLIF model. Mihalaş S, Niebur E. A generalized linear integrate-and-fire neural model produces diverse spiking behaviors. Neural Comput. 2009 Mar;21(3):704-18.` doi:10.1162/neco.2008.12-07-680. <https://doi.org/10.1162/neco.2008.12-07-680>`_ . PMID: 18928368; PMCID: PMC2954058.
.. [#f2] AEIF model. Brette, Romain & Gerstner, Wulfram. (2005). Adaptive Exponential Integrate-And-Fire Model As An Effective Description Of Neuronal Activity. Journal of neurophysiology. 94. 3637-42.` doi:10.1152/jn.00686.2005. <https://doi.org/10.1152/jn.00686.2005>`_


