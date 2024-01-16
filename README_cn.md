# SNN仿真训练平台的简单使用介绍
# SPAIC
Spike-based artificial intelligence computing platform

spaic平台仿真训练平台是针对脉冲神经网络开发的网络构建、前向模拟与学习训练平台，主要包括前端网络建模、后端仿真及训练、模型算法库、数据显示与分析等模块

依赖包：pytorch, numpy, 

# 如何安装

最近，SPAIC使用PyTorch作为后端进行计算。如果你想使用CUDA，请确保你已经安装了CUDA版本的PyTorch。

**SPAIC平台的教程文档：** https://spaic.readthedocs.io/en/latest/index.html

**从** [**PyPI**](https://pypi.org/project/SPAIC/) **安装最新的稳定版本:**

```bash
pip install SPAIC
```

**从** [**GitHub**](https://github.com/ZhejianglabNCRC/SPAIC) **安装:**

```bash
git clone https://github.com/ZhejianglabNCRC/SPAIC.git
cd SPAIC
python setup.py install
```

如果在阅读教程文档之后仍旧抱有疑问，欢迎通过邮件与我们取得联系：  
Chaofei Hong <a href="mailto:hongchf@zhejainglab.com"> hongchf@zhejianglab.com</a>  
Mengwen Yuan <a href="mailto:yuanmw@zhejianglab.com"> yuanmw@zhejianglab.com</a>  
Mengxiao Zhang <a href="mailto:mxzhangice@zju.edu.com"> mxzhangice@zju.edu.com</a>  



# 前端网络建模主要结构模块及函数

​	平台主要通过Assembly, Connection, NeuronGroup, Node, Network等五类结构模块构建网络，其具体功能叙述如下，建模结构关系如下图所示。

<img src="./docs/source/_static/front-end network components.png" style="zoom: 67%;" />



- **Assembly（神经集合）**：是神经网络结构拓扑的抽象类，代表任意网络结构，其它网络模块都是Assembly类的子类。Assembly对象具有名为_groups, _connections两个dict属性，保存神经集合内部的神经集群以及连接等。同时具有名为 _supers, _input_connections,  _output_connections的list属性，分别代表包含此神经集合的上层神经集合以及与此神经集合进行的连接。作为网络建模的主要接口，包含如下主要建模函数：
    - add_assembly(name, assembly): 向神经集合中加入新的集合成员
    - del_assembly(assembly=None, name=None): 删除神经集合中已经存在的某集合成员
    - copy_assembly(name, assembly): 复制一个已有的assembly结构，并将新建的assembly加入到此神经集合
    - replace_assembly(old_assembly, new_assembly):将集合内部已有的一个神经集合替换为一个新的神经集合
    - merge_assembly( assembly): 将此神经集合与另一个神经集合进行合并，得到一个新的神经集合
    - select_assembly(assemblies, name=None):将此神经集合中的部分集合成员以及它们间的连接取出来组成一个新的神经集合，原神经集合保持不变
    - add_connection( name, connection): 连接神经集合内部两个神经集群
    - del_connection(connection=None, name=None): 删除神经集合内部的某个连接
    - assembly_hide():将此神经集合隐藏，不参与此次训练、仿真或展示。
    - assembly_show():将此神经集合从隐藏状态转换为正常状态。
- **Connection(连接)**：建立各神经集合间连接的类，包含了不同类型突触连接的生成、管理的功能。
- **NeuronGroup (神经元集群)**：是一定数量神经元集群的类，通常称为一层神经元，具有相同的神经元模型、连接形式等，虽然继承自Assembly类，但其内部的 _groups和 _connections属性为空。
- **Node (节点)**：神经网络输入输出的中转节点，包含编解码机制，将输入转化为放电或将放电转会为输出。与NeuronGroup一样，内部的_groups和 _connections属性都为空。
- **Network(网络)**: Assembly子类中的最上层结构，每个构建的神经网络的所有模块都包含到一个Network对象中，同时负责网络训练、仿真、数据交互等网络建模外的工作。除了Assemby对象的 _groups和 _connections等属性外，还具有 _monitors, _learners, _optimizers, _backend, _pipeline等属性，同时 _supers, _input_connections,  _output_connections等属性为空。为训练等功能提供如下接口：
    - set_runtime：设置仿真时间
    - run: 进行一次仿真
    - save_state: 保存网络权重
    - state_from_dict: 读取网络权重



# 典型用例

采用spaic平台仿真训练主要包括：数据或环境导入、训练器训练流程相关参数选择、模型构建（包括输入输出节点构建、神经元集群、网络连接、网络拓扑、学习算法、数据记录器等单元）、神经元仿真或训练、模型数据分析及保存等步骤

### 导入spaic库


```python
import spaic
import torch #有些功能还没有写，需要借用pytorch的介绍
```

### 设置训练仿真参数


```python
run_time = 200.0
bat_size = 100
```

### 导入训练数据集


```python
# 创建数据集

root = 'D:\Datasets\MNIST'
dataset = spaic.MNIST(root, is_train=False)

# 创建DataLoader迭代器
dataloader = spaic.Dataloader(dataset, batch_size=bat_size, shuffle=True, drop_last=True)
n_batch = dataloader.batch_num
```

    >> Dataset loaded


## 网络模型构建
模型构建可以采用两种构建方法：第一种类似Pytorch的module类继承，在_init_函数中构建的形式，第二种是类似Nengo的通过with语句的构建方式。另外，也可以在建模过程中引入模型算法库中已经存在的模型结构

### 模型构建方法1： 类继承形式


```python
class ExampleNet(spaic.Network):
     def __init__(self):
        super(ExampleNet, self).__init__()
        
        # 建立输入节点并选择输入编码形式
        self.input = spaic.Encoder(num=784, encoding='latency')
              
        # 建立神经元集群，选择神经元类型，并可以设置 放电阈值、膜电压时间常数等神经元参数值
        self.layer1 = spaic.NeuronGroup(100, neuron_model='clif')
        self.layer2 = spaic.NeuronGroup(10, neuron_model='clif')
        
        # 建立神经集群间的连接
        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full')
        self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='full')
        
        # 建立输出节点，并选择输出解码形式
        self.output = spaic.Decoder(decoding='spike_counts',target=self.layer2)

        # 建立状态检测器，可以Monitor神经元、输入输出节点、连接等多种单元的状态量
        self.monitor1 = spaic.StateMonitor(self.layer1, 'V')

        # 加入学习算法，并选择需要训练的网络结构，（self代表全体ExampleNet结构）
        self.learner1 = spaic.STCA(0.5, self)
        
        # 加入优化算法
        self.optim = spaic.Adam(lr=0.01, schedule='StepLR', maxstep=1000)

# 初始化ExampleNet网络对象
Net = ExampleNet()
```

### 模型构建方法2：with形式


```python
# 初始化基本网络类的对象
Net = spaic.Network()

# 通过把网络单元在with内定义，建立网络结构
with Net:
    # 建立输入节点并选择输入编码形式
    input = spaic.Encoder(num=784, encoding='latency')


    # 建立神经元集群，选择神经元类型，并可以设置 放电阈值、膜电压时间常数等神经元参数值
    layer1 = spaic.NeuronGroup(100, neuron_model='clif')
    layer2 = spaic.NeuronGroup(10, neuron_model='clif')

    # 建立神经集群间的连接
    connection1 = spaic.Connection(input1, layer1, link_type='full')
    connection2 = spaic.Connection(layer1, layer2, link_type='full')

    # 建立输出节点，并选择输出解码形式
    output = spaic.Decoder(decoding='spike_counts',target=layer2)

    # 建立状态检测器，可以Monitor神经元、输入输出节点、连接等多种单元的状态量
    monitor1 = spaic.StateMonitor(layer1, 'V')

    # 加入学习算法，并选择需要训练的网络结构，（self代表全体ExampleNet结构）
    learner1 = spaic.STCA(0.5, Net)
    
    # 加入优化算法
    optim = spaic.Adam(lr=0.01, schedule='StepLR', maxstep=1000)
    
```

### 模型构建方法3：通过引入模型库模型并进行修改的方式构建网络


```python
from spaic.Library import ExampleNet
Net = ExampleNet()
# 神经元参数
neuron_param = {
    'tau_m': 8.0,
    'V_th': 1.5,
}
# 新建神经元集群
layer3 = spaic.NeuronGroup(100, neuron_model='lif', param=neuron_param)
layer4 = spaic.NeuronGroup(100, neuron_model='lif', param=neuron_param)

# 向神经集合中加入新的集合成员
Net.add_assembly('layer3', layer3)
# 删除神经集合中已经存在集合成员
Net.del_assembly(Net.layer3)
# 复制一个已有的assembly结构，并将新建的assembly加入到此神经集合
Net.copy_assembly('net_layer', ExampleNet())
# 将集合内部已有的一个神经集合替换为一个新的神经集合
Net.replace_assembly(Net.layer1, layer3)
# 将此神经集合与另一个神经集合进行合并，得到一个新的神经集合
Net2 = ExampleNet()
Net.merge_assembly(Net2)
#连接神经集合内部两个神经集群
con = spaic.Connection(Net.layer2, Net.net_layer, link_type='full')
Net.add_connection('con3', con)
#将此神经集合中的部分集合成员以及它们间的连接取出来组成一个新的神经集合，原神经集合保持不变
Net3 = Net.select_assembly([Net.layer2, net_layer])
```

### 选择后端仿真器并编译网络结构


```python
backend = spaic.Torch_Backend()
sim_name = backend.backend_name
Net.build(backend)
```

### *定义优化算法、训练调度器（暂时可以用pytorch模块）


```python
# 暂时使用pytorch的优化器和调度器
param = Net.get_aprameters()
optim = torch.optim.Adam(param, lr=0.01)
sheduler = torch.optim.lr_scheduler.StepLR(optim, 100)
```

### 进行训练


```python
Net.set_runtime(run_time)
Net.train(epochs=100)
#Net.train(optim=sheduler, epochs = 100, run_time=run_time)
```

### 进行测试


```python
Net.test()
```

### 单次仿真


```python
Net.run(run_time=run_time)
```

### 展示结果


```python
from matplotlib import pyplot as plt
plt.plot(Net.monitor1.times, Net.monitor1.values[0,0,:])
plt.show()
```

### 保存网络


```python
Net.save(filename='TestNet')
```
