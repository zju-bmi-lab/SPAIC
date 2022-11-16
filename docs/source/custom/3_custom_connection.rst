.. _my-custom-connection:



突触、连接模型自定义
=======================
本章节主要介绍连接模型的自定义，以便当本平台提供的内置方法无法满足用户需求时，用户可以方便的添加符合自己需求的连接方案。


连接模型自定义
--------------------------
连接作为脉冲神经网络最为基本的组成结构之一，包含了网络最为重要的权重信息。不同的连接方法会生成不同的空间连接结构，为了满足用户的\
大多数应用需求，在本平台中内置了10种最常用的连接模型，包含了全连接，卷积连接，一对一连接，稀疏连接等。与此同时，作为类脑计算平\
台，本平台中的连接支持仿生连接的形式，即支持反馈连接与连接延迟以及突触连接等具有一定生理特征的连接方式。内置的连接方法可能无法\
满足用户的任意需求，这时候就需要用户自己添加一些更符合其实验目的的连接模型。\
定义连接模型的这一步可以依照 :class:`spaic.Network.Connection` 文件中的格式进行添加。

连接方法初始化
^^^^^^^^^^^^^^^^^^^^^
自定义的连接方法需继承 :code:`Connection` 类，其初始化方法中的参数名需与 :code:`Connection` 类的一致，若需要传入初始化参数\
以外的参数，可以通过 :code:`kwargs` 传入，以 :code:`FullConnection` 类初始化函数为例：

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

在这个初始化方法中， :code:`FullConnection` 类所额外需要的参数，通过从 :code:`kwargs` 中获取的方式来设定。


突触模型自定义部分
-----------------------
突触模型是进行神经动力学仿真环节中非常重要的一步，不同的模型与不同的参数都会产生不同的现象。\
为了应对用户不同的应用需求， **SPAIC** 内置了两种最常用的突触模型（化学突触和电突触），但是偶尔还是会有力所不能及，\
这时候就需要用户自己添加一些更符合其实验的个性化突触模型。定义突触的这一步可以参考 :code:`Network.Synapse` \
文件依照格式进行添加。

定义可从外部获取的参数
^^^^^^^^^^^^^^^^^^^^^
在定义神经元模型的最初部分，我们需要先定义该神经元模型可以变更的一些参数，\
这些参数可由传参来改变。例如在化学突触的一阶衰减模型中，我们将其原本的公式经过变换后可得：

.. code-block:: python

    class First_order_chemical_synapse(SynapseModel):
        """
        .. math:: Isyn(t) = weight * e^{-t/tau}
        """

在这个公式中，:code:`self.tau` 是可变参数，所以我们通过 :code:`kwargs` 中获取的方式来改变：

.. code-block:: python

    self._syn_tau_variables['tau[link]'] = kwargs.get('tau', 5.0)
定义变量
^^^^^^^^^^^^^^^^^^^^^
在定义变量阶段，我们要先了解突触的几个变量形式：

- **_syn_tau_constant_variables** -- 指数衰减常数
- **_syn_variables** -- 普通变量

对于 :code:`_syn_tau_constant_variables` 我们会进行一个变换 :code:`value = np.exp(-self.dt / var)` ,

在定义变量时，同时需要设定初始值，在网络的每一次运行后，神经元的参数都会被重置为此处设定的初始值。

.. code-block:: python

    self._syn_variables[I] = 0
    self._syn_variables[WgtSum] = 0
    self._syn_tau_constant_variables[tauP] = self.tau_p


定义计算式
^^^^^^^^^^^^^^^^^^^^^
计算式是突触模型最为重要的部分，一行一行的计算式决定了各个参数在模拟过程中将会经过一些什么样的变化。

在添加计算式时，有一些需要遵守的规则。首先，每一行只能计算一个特定的计算符，所以需要将原公式\
进行分解，分解为独立的计算符。目前在平台中内置的计算符可以参考 :code:`backend.basic_operation` :

- add, minus, div
- var_mult, mat_mult, mat_mult_pre, sparse_mat_mult, reshape_mat_mult
- var_linear, mat_linear
- reduce_sum, mult_sum
- threshold
- cat
- exp
- stack
- conv_2d, conv_max_pool2d

在使用这些计算符时的格式，我们以化学突触模型中计算化学电流的过程作为示例：

.. code-block:: python

    # Isyn = O * weight 的公式转化为以下计算式并添加至self._syn_operations中，
    # conn.post_var_name作为计算结果放置在第一位，
    # 计算符mat_mult_weight放置在第二位，
    # input_name以及weight[link]代表着计算的因子，放置于第三位及以后，
    # [updated]符号目前代表该数值取的是本轮计算中计算出的新值，临时变量无需添加，
    self._syn_operations.append(
        [conn.post_var_name + '[post]', 'mat_mult_weight', self.input_name,
         'weight[link]'])

