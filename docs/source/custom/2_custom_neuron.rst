.. _my-custom-neuron:



神经元模型自定义
=======================
神经元模型是进行神经动力学仿真环节中最为重要的一步，不同的模型与不同的参数都会产生不同的现象。\
为了应对用户不同的应用需求， **SPAIC** 内置了许多最为常用的神经元模型，但是偶尔还是会有力所不能及，\
这时候就需要用户自己添加一些更符合其实验的个性化神经元。定义神经元的这一步可以依照 :class:`spaic.Neuron.Neuron` \
文件中的格式进行添加。

定义变量以及外部参数
--------------------------
在定义变量的阶段，我们需要先了解平台中设定的几个变量的形式：

- **_tau_variables** -- 指数衰减常数
- **_membrane_variables** -- 衰减常数
- **_variables** -- 普通变量
- **_parameter_variables** -- 参数变量
- **_constant_variables** -- 固定变量

对于 :code:`_tau_variables` 会进行变换 :code:`tau_var = np.exp(-dt/tau_var)`,
对于 :code:`_membrane_variables` 会进行变换 :code:`membrane_tau_var = dt/membrane_tau_var`,

在定义变量时，同时需要设定初始值，在网络的每一次运行后，神经元的参数都会被重置为此处设定的初始值。\
在定义神经元模型的最初部分，我们需要先定义该神经元模型可以变更的一些参数，这些参数可由传参来改变。\
例如在 :code:`lif` 神经元中，我们将其原本的公式经过变换后可得：

.. code-block:: python

    """
    LIF model:
    # V(t) = tuaM * V^n[t-1] + Isyn[t]   # tauM: constant membrane time (tauM=RmCm)
    O^n[t] = spike_func(V^n[t-1])
    """

在这个公式中，:code:`tauM` 以及阈值 :code:`v_th` 都是可变的参数，所以\
我们通过从 :code:`kwargs` 中获取的方式来改变，完整的变量定义如下：

.. code-block:: python

    self._variables['V'] = 0.0
    self._variables['O'] = 0.0
    self._variables['Isyn'] = 0.0

    self._parameter_variables['Vth'] = kwargs.get('v_th', 1)
    self._constant_variables['Vreset'] = kwargs.get('v_reset', 0.0)

    self._tau_variables['tauM'] = kwargs.get('tau_m', 20.0)



定义计算式
--------------------
计算式是神经元模型最为重要的部分，一行一行的计算式决定了神经元的各个参数在模拟过程中将会经过一些什么样的变化。

在添加计算式时，有一些需要遵守的规则。首先，每一行只能计算一个特定的计算符，所以需要将原公式\
进行分解，分解为独立的计算符。目前在平台中内置的计算符可以参考 :class:`spaic.backend.backend` 中对各个计算符具体的介绍:

- add, minus, div -- 简单的加减除的操作
- var_mult, mat_mult, mat_mult_pre, sparse_mat_mult, reshape_mat_mult  -- 变量乘法，矩阵乘法，对第一个因子进行维度转换的矩阵乘法，稀疏矩阵乘法，对第二个因子进行维度转换的矩阵乘法
- var_linear, mat_linear -- result=ax+b 变量的一阶线性乘法加和与矩阵的一阶线性乘法加和
- threshold -- 阈值函数
- cat
- exp
- stack
- conv_2d, conv_max_pool2d


在使用这些计算符时的格式，我们以 :code:`LIF` 模型中计算化学电流的过程作为示例：

.. code-block:: python

    # [updated]符号目前代表该数值取的是本轮计算中计算出的新值，临时变量无需添加，
    # Vtemp = V * tauM + I, 此处的tauM需要注意，因为tauM为 _tau_variables
    self._operations.append(('Vtemp', 'var_linear', 'tauM', 'V', 'Isyn[updated]'))

    # O = 1 if Vtemp >= Vth else 0， threshold起的作用为判断Vtemp是否达到阈值Vth
    self._operations.append(('O', 'threshold', 'Vtemp', 'Vth'))

    # 此处作用为在脉冲发放之后重置电压V
    self._operations.append(('V', 'reset', 'Vtemp',  'O[updated]'))


在最后，需要添加 :code:`NeuronModel.register("lif", LIFModel)` 用于将该神经元模型添加至神经元模型的库中。