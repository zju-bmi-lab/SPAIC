网络
=====================

本章节主要介绍 **SPAIC** 平台中基于 ``Network`` 类构建网络和运行网络的方法。

网络构建
-----------------------------------
模型构建可以采用三种构建方法：第一种类似Pytorch的module类继承，在_init_函数中构建的形式，第二种是类似Nengo的通过with语句的构建方式。第三种，\
也可以在建模过程中在现有网络中通过添加函数接口添加新的网络模块。

模型构建方法1： 类继承形式

.. code-block:: python

   class SampleNet(spaic.Network):
      def __init__(self):
         super(SampleNet, self).__init__()

         self.layer1 = spaic.NeuronGroup(100, neuron_model='clif')
         ......

   Net = SampleNet()


模型构建方法2： with形式

.. code-block:: python

     Net = SampleNet()
     with Net:

        layer1 = spaic.NeuronGroup(100, neuron_model='clif')
        ......

模型构建方法3： 通过函数接口构建或更改网络

.. code-block:: python

    Net = SampleNet()
    layer1 = spaic.NeuronGroup(100, neuron_model='clif')
    ....

    Net.add_assembly('layer1', layer1)

当前 ``Network`` 提供的构建或修改网络的函数接口包括：

- **add_assembly(name, assembly)** - 添加神经集群类（包括 ``Node``， ``NeuronGroup`` 等），其中参数name代表网络中变量名，assembly代表被添加的神经集群对象
- **copy_assembly(name, assembly)** - 复制神经集群类（包括 ``Node``， ``NeuronGroup`` 等），与add_assembly不同，此接口将assembly对象进行克隆后添加到网络中
- **add_connection(name, connection)** - 添加连接对象，其中参数name代表网络中变量名，connection代表被添加的连接对象
- **add_projection(name, projection)** - 添加拓扑映射对象，其中参数name代表网络中变量名，projection代表被添加的拓扑映射对象
- **add_learner(name, learner)** - 添加学习算法，其中参数name代表网络中变量名，learner代表被添加的学习算法对象
- **add_moitor(name, moitor)** - 添加监视器，其中参数name代表网络中变量名，moitor代表被添加的监视器对象


网络运行及运行参数设置
------------------------------
``Network`` 对象提供运行和设置运行参数的函数接口，包括：

- **run(backend_time)** - 运行网络的函数接口，其中backend_time参数是网络运行时间
- **run_continue(backend_time)** - 继续运行网络的函数接口，与run不同，run_continue不会重置各变量初始值而是根据原初始值继续运行
- **set_backend(backend, device, partition)** - 设置网络运行的后端， 其中参数backend是运行后端对象或后端名称，device是后端计算的硬件， partition代表是否将模型分布不同device上
- **set_backend_dt(backend, dt)** - 设置网络运行时间步长， dt为步长
- **set_random_seed(seed)** - 设置网络运行随机种子


