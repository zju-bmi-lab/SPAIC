save or load model
=====================

This section will describe two ways of saving network information in detail.

pre-defined function in Network
---------------------------------------------------------
Use pre-defined functions :code:`save_state` and :code:`state_from_dict` to save the weight of the model directly. \

The optional parameters are :code:`filename` , :code:`direct` and :code:`save`. If users use :code:`save_state` without \
giving any parameters, the function will use default name :code:`autoname` with random number as the direct name and save \
the weight into the './autoname/parameters/_parameters_dict.pt'. If given :code:`filename`, or :code:`direct` , it will \
save the weight into 'direct/filename/parameters/_parameters_dict.pt'. Parameter :code:`save` is default as True, which \
means it will save the weight. If users choose False, this function will return the :code:`parameter_dict` of the model \
directly.

The parameters of :code:`state_from_dict` is same as :code:`save_state` but have two more parameters: :code:`state` and :code:`direct` ,\
and :code:`save` parameters is unneeded. If users provide :code:`state` , this function will use given parameters to replace the parameter dict \
of the backend. If :code:`state` is None, this function will decide the saving path according to :code:`filename` and :code:`direct`. The \
:code:`device` will decide where to storage the parameters.

.. code-block:: python

    Net.save_state('Test1', True)
    ...
    Net.state_from_dict('Test1', device)

network_save 与 network_load
---------------------------------------------------------------------------------------------------------------------------------------
The network save module with :code:`spaic.Network_saver.network_save` and :code:`spaic.Network_loader.network_load` \
will save the whole network structure of the model and the weight information separately.

Library中的网络存储模块 :code:`spaic.Network_saver.network_save` 函数与 :code:`spaic.Network_loader.network_load` 函数\
将会将完整的网络结构以及权重信息分别存储下来，该方式在使用时需要一个文件名dir_name，然后平台会在当前程序的运行目录下新\
建'NetData/dir_name/dir_name.json'用于保存网络结构，权重的存储路径与 :code:`net.save_state` 相同，都会在用户当前目录下新建NetData文件夹，然后存于\
NetData/。其次，用户在使用 :code:`network_save` 时，还可以选择存储的文件格式，是采用json文件的格式或是yaml。

.. code-block:: python

    network_dir = network_save(Net=Net, filename='TestNet',
                                            trans_format='json', combine=False, save=True)

    # network_dir = 'TestNet'
    Net2 = network_load(network_dir, device=device)

在 :code:`network_save` 中，

- Net: 具体SPAIC网络中的网络对象
- filename: 文件名称，network_save将会将Net以该名称进行存储
- trans_format 存储格式，此处可以选择的是‘json’或是’yaml‘，默认为‘json’结构。
- combine: 该参数制定了权重是否与网络结构存储在一起，默认为False，分开存储网络结构与权重信息。
- save: 该参数决定了平台是否会将网络结构存储下来，若为True，则最后会返回存储的名称以及网络信息，若为False，则不会存储网络，仅仅只会将网络结构以字典的形式返回

下面，我举例说明保存下来的网络结构中各个参数所代表的意义：

.. code-block:: python

    # 输入输出节点的存储格式
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

    # 神经元层的存储格式
    -   layer1:
            _class_label: <neg> # 表示该对象为NeuronGroup类型
            id: autoname1<net>_layer1<neg> # 表示该NeuronGroup的id，具体含义为，该对象是在名为autoname1的网络下的名为layer1的神经元组
            model_name: clif # 采用的神经元模型的类型
            name: layer1 # 该NeuronGroup的姓名
            num: 10 # 该NeuronGroup中Neuron的数量
            parameters: {} # 额外输入的kwargs中的parameters，在神经元中为各类神经元模型的参数
            shape: # 维度
            - 10
            type: null # 该type表示的是神经元是兴奋还是抑制，暂未启用该参数

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
                post_assembly: layer3   # 突触后神经元为layer3层, 此处为特殊情况，layer3其实为一个assembly
                post_var_name: WgtSum   # 该连接对突触后神经元的输出为WgtSum
                pre_assembly: layer2    # 突触前神经元为layer2层
                pre_var_name: O         # 该连接接受突触前神经元的输入为‘O’
                sparse_with_mask: false # 是否启用mask，该设定为平台对于系数矩阵所设置，具体可移步connection中查看具体说明
                weight: # 权重矩阵
                    autoname1<net>_layer3<asb>_connection0<con>:autoname1<net>_layer3<asb>_layer3<neg><-autoname1<net>_layer3<asb>_layer2<neg>:{weight}: # 此处为该权重的id，在平台后端变量库中可以获取
                    -   - 0.05063159018754959
                    # 该权重的id的格式解读为：这是一个属于网络autoname1的组合layer3中的名为connection0的连接，该链接由'<-'标识后方的autoname1中的layer3下的layer2层连接向autoname1中的layer3中的layer3
                    # 即， layer3为autoname1中的一个组合层，该连接为组合层layer3中的layer2连向了layer3

    # 连接的存储格式
    -   connection1:
            _class_label: <con> # 表示该对象为Connection类型
            link_type: full # 连接形式为全链接
            max_delay: 0 # 连接的最大延迟
            name: connection1 # 连接的姓名
            parameters: # 连接的参数，此处为连接初始化时所用的参数，有给定权值时将会采用给定的权值
                w_mean: 0.02
                w_std: 0.05
            post_assembly: layer1   # 突触后神经元为layer1层
            post_var_name: WgtSum   # 该连接对突触后神经元的输出为WgtSum
            pre_assembly: input     # 突触前神经元为input层
            pre_var_name: O         # 该连接接受突触前神经元的输入为‘O’
            sparse_with_mask: false # 是否启用mask，该设定为平台对于系数矩阵所设置，具体可移步connection中查看具体说明
            weight: # 权重矩阵
                autoname1<net>_connection1<con>:autoname1<net>_layer1<neg><-autoname1<net>_input<nod>:{weight}:
                -   - 0.05063159018754959
                    ......

    # 学习算法的存储格式
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

