保存与读取模型
=====================

该部分将详细描述两种存取网络信息的方式。

Network中预先定义的函数
---------------------------------------------------------
采用 ``Network`` 中预定义的 :code:`save_state` 与 :code:`state_from_dict` 函数将权重直接进行存取。

:code:`save_state` 函数可选的参数有 :code:`filename` 、\
:code:`direct` 以及 :code:`save` 。用户如果直接调用 :code:`save_state` 函数时，将会以默认的随机名称 :code:`autoname` 将后端中的权重变量直接存储于当前目录下的\
``'./autoname/parameters'`` 文件夹下的 ``'_parameters_dict.pt'`` 文件中。启用 :code:`filename` 时，将会以用户给予的 :code:`filename` 替换 :code:`autoname` 。
启用 :code:`direct` 参数则用于指定存储的目录。 :code:`save` 参数默认为 ``True`` ，即启用保存，若为 ``False`` ，则该函数会直接返回后端中存储的权重信息。

:code:`state_from_dict` 函数的参数与 :code:`save_state` 类似，不同点在于多了 :code:`state` 与 :code:`device` 参数而少了 :code:`save` 参数。 \
:code:`state` 参数如果传入参数，则该函数会直接使用传入的参数来替代后端的权重参数，在该参数为None的情况下，则会根据 :code:`filename` 与 :code:`direct` 来决定文件\
的读取路径。 使用此函数时选用的 :code:`device` 则会将读取出来的权重参数存储于对应的设备上。


.. code-block:: python

    Net.save_state('Test1', True)
    ...
    Net.state_from_dict(filename='Test1', device=device)


network_save 与 network_load
---------------------------------------------------------------------------------------------------------------------------------------
``Library`` 中的网络存储模块 :code:`spaic.Network_saver.network_save` 函数与 :code:`spaic.Network_loader.network_load` 函数\
将会将完整的网络结构以及权重信息分别存储下来，该方式在使用时需要一个文件名 ``filename`` ，然后平台会在用户提供的目录或是默认的当前目录下新\
建 ``'filename/filename.json'`` 用于保存网络结构，权重的存储路径与 :code:`net.save_state` 相同，都会在目标目录下进行存储。 \
其次，用户在使用 :code:`network_save` 时，还可以选择存储的文件格式， ``json`` 或是 ``yaml`` 。

.. code-block:: python

    network_dir = network_save(Net=Net, filename='TestNet',
                                            trans_format='json', combine=False, save=True)

    # network_dir = 'TestNet'
    Net2 = network_load(network_dir, device=device)

在 :code:`network_save` 中，

- **Net** -- 具体 **SPAIC** 网络中的网络对象
- **filename** -- 文件名称， ``network_save`` 将会将 ``Net`` 以该名称进行存储，若不提供则会根据网络名存储
- **path** -- 文件的存储路径，将会在目标路径下根据文件名新建文件夹
- **trans_format** -- 存储格式，此处可以选择的是‘json’或是’yaml‘，默认为‘json’结构。
- **combine** -- 该参数制定了权重是否与网络结构存储在一起，默认为 ``False`` ，分开存储网络结构与权重信息。
- **save** -- 该参数决定了平台是否会将网络结构存储下来，若为 ``True`` ，则最后会返回存储的名称以及网络信息，若为 ``False`` ，则不会存储网络，仅仅只会将网络结构以字典的形式返回
- **save_weight** -- 该参数决定了平台是否会存储权重部分（后端部分），若为 ``True`` 则会存储权重。

在存储网络各部分参数过程中，如果神经元的参数采用Tensor的形式传入，则在存储文件中将会存储这些参数的名称，并将实际参数存储于与权重同一目录下的diff_para_dict.pt文件中。

下面，举例说明保存下来的网络结构中各个参数所代表的意义：

.. code-block:: python

    # 输入节点的存储信息
    -   input:
            _class_label: <nod> # 表示该对象为节点类型
            _dt: 0.1 # 每个时间步的长度
            _time: null #
            coding_method: poisson # 编码方式
            coding_var_name: O # 该节点输出的对象
            dec_target: null # 解码对象，由于是input节点，没有解码对象
            name: input # 节点名称
            num: 784 # 节点中的元素个数
            shape: # 维度
            - 784

    # 神经元层的存储信息
    -   layer1:
            _class_label: <neg> # 表示该对象为NeuronGroup类型
            id: autoname1<net>_layer1<neg> # 表示该NeuronGroup的id，具体含义为，该对象是在名为autoname1的网络下的名为layer1的神经元组
            model_name: clif # 采用的神经元模型的类型
            name: layer1 # 该NeuronGroup的姓名
            num: 10 # 该NeuronGroup中Neuron的数量
            parameters: {} # 额外输入的kwargs中的parameters，在神经元中为各类神经元模型的参数
            shape: # 维度
            - 10
            type: null # 该type表示的是神经元是兴奋还是抑制，用于Projection中policy功能

    -   layer3:
        -   layer1:
                _class_label: <neg> # 表示该对象为NeuronGroup类型
                id: autoname1<net>_layer3<asb>_layer1<neg>  # 表示该NeuronGroup的id，具体含义为，该对象是在名为autoname1的网络下的名为layer3的组合中的名为layer1的神经元组
                model_name: clif # 采用的神经元模型的类型
                name: layer1 # 该NeuronGroup的姓名，由于是在layer3内部，所以不会出现与上述layer1重名的现象
                num: 10 # 该NeuronGroup中Neuron的数量
                parameters: {} # 额外输入的kwargs中的parameters，在神经元中为各类神经元模型的参数
                shape: # 维度
                - 10
                type: null # 该type表示的是神经元是兴奋还是抑制，暂未启用该参数

        -   connection0:
                _class_label: <con> # 表示该对象为Connection类型
                link_type: full # 连接形式为全链接
                max_delay: 0 # 连接的最大延迟
                name: connection0 # 连接的姓名
                parameters: {}
                post: layer3   # 突触后神经元为layer3层, 此处为特殊情况，layer3其实为一个assembly
                post_var_name: WgtSum   # 该连接对突触后神经元的输出为WgtSum
                pre: layer2    # 突触前神经元为layer2层
                pre_var_name: O         # 该连接接受突触前神经元的输入为‘O’
                sparse_with_mask: false # 是否启用mask，该设定为平台对于系数矩阵所设置，具体可移步connection中查看具体说明
                weight: # 权重矩阵
                    autoname1<net>_layer3<asb>_connection0<con>:autoname1<net>_layer3<asb>_layer3<neg><-autoname1<net>_layer3<asb>_layer2<neg>:{weight}: # 此处为该权重的id，在平台后端变量库中可以获取
                    -   - 0.05063159018754959
                    # 该权重的id的格式解读为：这是一个属于网络autoname1的组合layer3中的名为connection0的连接，该链接由'<-'标识后方的autoname1中的layer3下的layer2层连接向autoname1中的layer3中的layer3
                    # 即， layer3为autoname1中的一个组合层，该连接为组合层layer3中的layer2连向了layer3

    # 连接的存储信息
    -   connection1:
            _class_label: <con> # 表示该对象为Connection类型
            link_type: full # 连接形式为全链接
            max_delay: 0 # 连接的最大延迟
            name: connection1 # 连接的姓名
            parameters: # 连接的参数，此处为连接初始化时所用的参数，有给定权值时将会采用给定的权值
                w_mean: 0.02
                w_std: 0.05
            post: layer1   # 突触后神经元为layer1层
            post_var_name: WgtSum   # 该连接对突触后神经元的输出为WgtSum
            pre: input     # 突触前神经元为input层
            pre_var_name: O         # 该连接接受突触前神经元的输入为‘O’
            sparse_with_mask: false # 是否启用mask，该设定为平台对于系数矩阵所设置，具体可移步connection中查看具体说明
            weight: # 权重矩阵
                autoname1<net>_connection1<con>:autoname1<net>_layer1<neg><-autoname1<net>_input<nod>:{weight}:
                -   - 0.05063159018754959
                    ......

    # 学习算法的存储信息
    -   learner2:
            _class_label: <learner> # 表示该对象为Learner类型，为学习算法
            algorithm: full_online_STDP # 表示Learner对象采用的学习算法是 full_online_STDP
            lr_schedule_name: null # 表示该Learner对象采用的 lr_schedule优化算法，null为未采用
            name: _learner2 # 该Learner对象的名称
            optim_name: null # 表示该Learner对象采用的optimizer优化算法，null为未采用
            parameters: {} # 表示该Learner对象的额外参数，例如在STCA中需要设定一个alpha值
            trainable: # 表示该Learner对象作用的范围，此处即学习算法针对connection1与connection2起作用
            - connection1
            - connection2

