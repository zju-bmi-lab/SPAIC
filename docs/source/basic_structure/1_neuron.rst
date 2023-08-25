神经元
=====================

本章节主要介绍在训练以及仿真中如何选择神经元模型，以及如何根据需求在原模型的基础上更改一些重要的参数。

神经元模型是脉冲神经网络中极为重要的一个组成部分，不同的神经元模型通常代表了对不同的神\
经元动力学的仿真与模拟。在脉冲神经网络中，我们通常将神经元模型关于电压的变化特征化为微\
分方程，再由差分方程来对其进行逼近，最后获得了计算机可以进行运算的神经元模型。在\
**SPAIC** 中，我们包含了大多数较为常见的神经元模型：

- **IF** - Integrate-and-Fire model
- **LIF** - Leaky Integrate-and-Fire model
- **CLIF** - Current Leaky Integrate-and-Fire model
- **GLIF** - Generalized Leaky Integrate-and-Fire model
- **aEIF** - Adaptive Exponential Integrate-and-Fire model
- **IZH** - Izhikevich model
- **HH** - Hodgkin-Huxley model

在 **SPAIC** 中， ``NeuronGroup`` 是作为网络节点的组成，如同 **PyTorch** 中的layer， **SPAIC** \
中的每个layer都是一个 ``NeuronGroup`` ，用户需要根据自己的需要指定在这个 ``NeuronGroup`` 中\
所包含的神经元数量、神经元类型及与其类型相关的参数等。首先需\
要的就是导入 ``NeuronGroup`` 库：

.. code-block:: python

    from spaic import NeuronGroup


LIF神经元
------------------
**LIF(Leaky Integrated-and-Fire Model)** 神经元的公式以及参数：

.. math::
    V & = tua\_m * V + I \\
    O & = spike\_func(V^n)


以建立一层含有100个 **LIF** 神经元的layer为例:

.. code-block:: python

    self.layer1 = NeuronGroup(num=100, model='lif')


一个含有100个标准 **LIF** 神经元的layer就建立好了。然而许多时候我们需要按需定制不同的 **LIF** \
神经元以获得不同的神经元的表现，这时候就需要在建立 ``NeuronGroup`` 时，指定一些参数：

- **tau_m** - 神经元膜电位的时间常量，默认为6.0
- **v_th** - 神经元的阈值电压，默认为1.0
- **v_reset** - 神经元的重置电压，默认为0.0

如果用户需要调整这些变量，可以在建立 ``NeuronGroup`` 的时候输入想改变的参数即可：

.. code-block:: python

    self.layer2 = NeuronGroup(num=100, model='lif',
                    tau_m=10.0, v_th=10, v_reset=0.2)


这样，一个自定义参数的LIF神经元就建好了。

.. image:: ../_static/LIF_Appearance.png

CLIF神经元
-------------------------
**CLIF(Current Leaky Integrated-and-Fire Model)** 神经元公式以及参数:

.. math::

    V(t) & = M(t) - S(t) - E(t) \\
    I & = V0 * I \\
    M & = tau\_p * M + I \\
    S & = tau\_q * S + I \\
    E & = tau\_p * E + Vth * O \\
    O & = spike\_func(V)


- **tau_p, tau_q** - 突触的时间常量，默认为12.0和8.0
- **tau_m** - 神经元膜电位的时间常量，默认为20.0
- **v_th** - 神经元的阈值电压，默认为1.0

.. image:: ../_static/CLIF_Appearance.png

GLIF神经元
-------------------------
**GLIF(Generalized Leaky Integrate-and-Fire Model)** [#f1]_ 神经元参数:

- **R, C, E_L**
- **Theta_inf**
- **f_v**
- **delta_v**
- **b_s**
- **delta_Theta_s**
- **k_1, k_2**
- **delta_I1, delta_I2**
- **a_v, b_v**

aEIF神经元
-------------------------
**aEIF(Adaptive Exponential Integrated-and-Fire Model)** [#f2]_ 神经元公式以及参数:

.. math::
    V & = V + dt / C * (gL * (EL - V + EXP) - w + I) \\
    w & = w + dt / tau\_w * (a * (V - EL) - w) \\
    EXP & = delta\_t * exp(dv\_th/delta\_t) \\
    dv & = V - EL \\
    dv\_th & = V - Vth \\
    O & = spike\_func(V) \\
    if\quad V & > 20: \\
    then\quad V & = EL, w = w + b

- **C, gL** - 膜电容与泄漏电导系数
- **tau_w** - 自适应时间常量
- **a.** - 阈下自适应系数
- **b.** - 脉冲激发自适应系数
- **delta_t** - 速率因子
- **EL** - 泄漏反转电位

.. image:: ../_static/AEIF_Appearance.png

IZH神经元
--------------------------
**IZH(Izhikevich Model)** [#f3]_  神经元公式以及参数:

.. math::
    V &= V + dt / tau\_M * (C1 * V * V + C2 * V + C3 - U + I)  \\
    V &= V + dt / tau\_M * (V* (C1 * V + C2) + C3 - U + I) \\
    U &= U + a. * (b. * V - U) \\
    O &= spike\_func(V^n) \\
    if\quad V &> Vth, \\
    then\quad V &= Vreset, U = U + d

- **tau_m**
- **C1, C2, C3**
- **a, b, d**
- **Vreset** - 电压重置位

.. image:: ../_static/IZH_Appearance.png

HH神经元
--------------------------
**HH(Hodgkin-Huxley Model)**  [#f4]_ 神经元模型及参数:

.. math::
    V & = V + dt/tau\_v * (I - Ik) \\
    Ik & = NA + K + L \\
    NA & = g\_NA * m^3 * h * (V - V_NA) \\
    K & = g\_K * n^4 * (V - V_K) \\
    L & = g\_L * (V - V_L) \\
    K\quad activation: \\
    n & = n + dt/tau\_n * (alpha\_n * (1-n) - beta\_n * n) \\
    Na\quad activation: \\
    m & = m + dt/tau\_m * (alpha\_m * (1-m) - beta\_m * m) \\
    Na\quad inactivation: \\
    h & = h + dt/tau\_h * (alpha\_h * (1-h) - beta\_h * h) \\
    alpha\_m & = 0.1 * (-V + 25) / (exp((-V+25)/10) - 1) \\
    beta\_m & = 4 * exp(-V/18) \\
    alpha\_n & = 0.01 * (-V + 10) / (exp((-V+10)/10) - 1) \\
    beta\_n & = 0.125 * exp(-V/80) \\
    alpha\_h & = 0.07 * exp(-V/20) \\
    beta\_h & = 1/(exp((-V+30)/10) + 1) \\
    O & = spike\_func(V)


- **dt**
- **g_NA, g_K, g_L**
- **E_NA, E_K, E_L**
- **alpha_m1, alpha_m2, alpha_m3**
- **beta_m1, beta_m2, beta_m3**
- **alpha_n1, alpha_n2, alpha_n3**
- **beta_n1, beta_n2, beta_n3**
- **alpha_h1, alpha_h2, alpha_h3**
- **beta_1, beta_h2, beta_h3**
- **Vreset**
- **m, n, h**
- **V, v_th**

.. image:: ../_static/HH_Appearance.png

自定义
----------------
在稍后的 :ref:`my-custom-neuron` 这一章节中，我们会更加详细具体地讲述该如何在我们平台上添加自定义的神\
经元模型。



.. [#f1] **GLIF model** : Teeter, C., Iyer, R., Menon, V., Gouwens, N., Feng, D., Berg, J., ... & Mihalas, S. (2018). Generalized leaky integrate-and-fire models classify multiple neuron types. Nature communications, 9(1), 1-15.
.. [#f2] **AEIF model** : Brette, Romain & Gerstner, Wulfram. (2005). Adaptive Exponential Integrate-And-Fire Model As An Effective Description Of Neuronal Activity. Journal of neurophysiology. 94. 3637-42.` doi:10.1152/jn.00686.2005. <https://doi.org/10.1152/jn.00686.2005>`_
.. [#f3] **IZH model** : Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on neural networks, 14(6), 1569-1572.
.. [#f4] **HH model** : Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. The Journal of physiology, 117(4), 500.
