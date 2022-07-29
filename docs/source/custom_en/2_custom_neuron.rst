.. _my-custom-neuron:



Custom neuron model
=======================
神经元模型是进行神经动力学仿真环节中最为重要的一步，不同的模型与不同的参数都会产生不同的现象。\
为了应对用户不同的应用需求，SPAIC内置了许多最为常用的神经元模型，但是偶尔还是会有力所不能及，\
这时候就需要用户自己添加一些更符合其实验的个性化神经元。定义神经元的这一步可以依照 :code:`Neuron.Neuron` \
文件中的格式进行添加。

定义可从外部获取的参数
--------------------------
在定义神经元模型的最初部分，我们需要先定义该神经元模型可以变更的一些参数，这些参数可由传参来改变。\
例如在 :code:`lif` 神经元中，我们将其原本的公式经过变换后可得：

.. code-block:: python

    # LIF model:
    # I = tauP*I + WgtSum^n[t-1] + b^n                         # sum(w * O^(n-1)[t])
    # F = tauM * exp(-O^n[t-1] / tauM)
    # V(t) = V^n[t-1] * F + I
    # O^(n)[t] = spike_func(V^n(t))

在这个公式中，:code:`tauP` 、:code:`tauM` 以及阈值 :code:`v_th` 都是可变的参数，所以\
我们通过从 :code:`kwargs` 中获取的方式来改变：

.. code-block:: python

    self.neuron_parameters['tau_p'] = kwargs.get('tau_p', 1.0)
    self.neuron_parameters['tau_m'] = kwargs.get('tau_m', 10.0)
    self.neuron_parameters['v_th']  = kwargs.get('v_th', 1.0)

定义变量
----------
在定义变量的阶段，我们需要先了解平台中设定的几个变量的形式：

- _tau_constant_variables: 指数衰减常数
- _membrane_variables: 衰减常数
- _variables: 普通变量
- _constant_variables: 固定变量

对于 :code:`_tau_constant_variables` 我们会进行一个变换 :code:`tau_var = np.exp(-dt/tau_var)`,
对于 :code:`_membrane_variables` 我们会进行一个变换 :code:`membrane_tau_var = dt/membrane_tau_var`,

在定义变量时，同时需要设定初始值，在网络的每一次运行后，神经元的参数都会被重置为此处设定的初始值。

.. code-block:: python

    self._variables['V'] = 0.0
    self._variables['O'] = 0.0
    self._variables['WgtSum'] = 0.0
    self._variables['b'] = 0.0
    self._variables['I_che'] = 0.0
    self._variables['I_ele'] = 0.0
    self._variables['I'] = 0.0

    self._constant_variables['Vth'] = self.neuron_parameters['v_th']

    self._tau_constant_variables['tauM'] = self.neuron_parameters['tau_m']
    self._tau_constant_variables['tauP'] = self.neuron_parameters['tau_p']


定义计算式
--------------------
计算式是神经元模型最为重要的部分，一行一行的计算式决定了神经元的各个参数在模拟过程中将会经过一些什么样的变化。

在添加计算式时，有一些需要遵守的规则。首先，每一行只能计算一个特定的计算符，所以需要将原公式\
进行分解，分解为独立的计算符。目前在平台中内置的计算符可以参考 :code:`backend.basic_operation`:

- add, minus, div
- var_mult, mat_mult, mat_mult_pre, sparse_mat_mult, reshape_mat_mult
- var_linear, mat_linear
- reduce_sum, mult_sum
- threshold
- cat
- exp
- stack
- conv_2d, conv_max_pool2d

在使用这些计算符时的格式，我们以 :code:`LIF` 模型中计算化学电流的过程作为示例：

.. code-block:: python

    # PSP = WgtSum + b的公式转化为以下计算式并添加至self._operations中，PSP作为计算结果放置在第一位，计算符add放置在第二位
    # [updated]符号目前代表该数值取的是本轮计算中计算出的新值，临时变量无需添加，
    self._operations.append(('PSP', 'add', 'WgtSum[updated]', 'b'))

    # I_che = tauP * I_che + PSP的公式转化为以下计算式，由于PSP是临时变量，无需添加[updated]
    self._operations.append(('I_che', 'var_linear', 'tauP', 'I_che', 'PSP'))

    # I = I_che + I_ele， 此处的I_che需要使用上一步计算出的I_che时，就需要添加上I_che
    self._operations.append(('I', 'add', 'I_che[updated]', 'I_ele[updated]'))

    # Vtemp = V * tauM + I, 此处的tauM需要注意，因为tauM为 _tau_constant_variables
    self._operations.append(('Vtemp', 'var_linear', 'V', 'tauM', 'I[updated]'))

    # O = 1 if Vtemp >= Vth else 0， threshold起的作用为判断Vtemp是否达到阈值Vth
    self._operations.append(('O', 'threshold', 'Vtemp', 'Vth'))

    # 此处作用为在脉冲发放之后重置电压V
    self._operations.append(('Vreset', 'var_mult', 'Vtemp', 'O[updated]'))
    self._operations.append(('V', 'minus', 'Vtemp', 'Vreset'))


在代码的最后，需要添加 :code:`NeuronModel.register("lif", LIFModel)` 用于将该神经元模型添加至神经元模型的库中，以便前端的调用。