Basic Structure
**********************
.. toctree::
   :maxdepth: 1
   :titlesonly:

   1_neuron
   2_connection
   3_synaptic
   4_algorithm
   5_encode_decode


Basic components
===================

Assembly is the most important basic class in SPAIC. :code:`spaic.assembly` contains three part: Network,\
NeuronGroup and Node. :code:`spaic.Network` is the basic class of the whole model. :code:`spaic.Neurongroup` \
contains neurons and :code:`spaic.Node` is the basic class of the input and output nodes.

Assembly
--------------------------
It is an abstract class of neural network structure, representing any network structure, and other network modules are \
subclasses of the Assembly class. The Assembly object has two properties named :code:`_groups` and\
:code:`_connections`


是神经网络结构的抽象类，代表任意网络结构，其它网络模块都是Assembly类的子类。Assembly对象具有名为 :code:`_groups` ,\
:code:`_connections` 两个dict属性，保存神经集合内部的神经集群以及连接等。同时具有名为 :code:`_supers` , :code:`_input_connections` , \
:code:`_output_connections` 的list属性，分别代表包含此神经集合的上层神经集合以及与此神经集合进行的连接。作为网络建模的主要接口，包含如下主要建模函数：

    - add_assembly(name, assembly): 向神经集合中加入新的集合成员
    - del_assembly(assembly=None, name=None): 删除神经集合中已经存在的某集合成员
    - copy_assembly(name, assembly): 复制一个已有的assembly结构，并将新建的assembly加入到此神经集合
    - replace_assembly(old_assembly, new_assembly): 将集合内部已有的一个神经集合替换为一个新的神经集合
    - merge_assembly(assembly): 将此神经集合与另一个神经集合进行合并，得到一个新的神经集合
    - select_assembly(assemblies, name=None): 将此神经集合中的部分集合成员以及它们间的连接取出来组成一个新的神经集合，原神经集合保持不变
    - add_connection(name, connection): 连接神经集合内部两个神经集群
    - del_connection(connection=None, name=None): 删除神经集合内部的某个连接
    - assembly_hide(): 将此神经集合隐藏，不参与此次训练、仿真或展示。
    - assembly_show(): 将此神经集合从隐藏状态转换为正常状态。



NeuronGroup (神经元集群)
--------------------------
是一定数量神经元集群的类，通常称为一层神经元，具有相同的神经元模型、连接形式等，虽然继承自Assembly类，但其内部的 :code:`_groups` 和 :code:`_connections` 属性为空。

Node (节点)
--------------------------
神经网络输入输出的中转节点，包含编解码机制，将输入转化为放电或将放电转会为输出。与 :code:`NeuronGroup` 一样，内部的 :code:`_groups` \
和 :code:`_connections` 属性都为空。

Network (网络)
--------------------------
Assembly子类中的最上层结构，每个构建的神经网络的所有模块都包含到一个Network对象中，同时负责网络训练、仿真、数据交互等网络建模外的工作。
除了Assemby对象的 :code:`_groups` 和 :code:`_connections` 等属性外，还具有 :code:`_monitors` , :code:`_learners` ,\
:code:`_optimizers` , :code:`_backend` , :code:`_pipeline` 等属性，同时 :code:`_supers` , \
:code:`_input_connections` ,  :code:`_output_connections` 等属性为空。为训练等功能提供如下接口：

    - set_runtime: 设置仿真时间
    - run: 进行一次仿真
    - save_state: 保存网络权重
    - state_from_dict: 读取网络权重

Projection (拓扑连接)
-------------------------
是神经网络结构拓扑的抽象类，代表着Assembly与Assembly之间的连接，connection是Projection的子类。在不同的Assembly之间建立Projection时，可以选定\
需要的连接类型，并且设定 :code:`policies` ，然后Projection会根据用户设定的类型和策略，生成对应的连接。


Connection (连接)
--------------------------
建立各神经集合间连接的类，包含了不同类型突触连接的生成、管理的功能。

平台前端结构图：

.. image:: ../_static/SPAIC_FRONTEND.png
    :width: 100%

Backend
===================
后端核心，负责实际构建变量以及生成计算图。
