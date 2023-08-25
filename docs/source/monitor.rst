监视器
============================

监视器主要的作用是监控网络运行过程中各类变量的变化过程，在SPAIC中，我们内置了两种形式的监视器，分别是 :code:`StateMonitor` \
与 :code:`SpikeMonitor` 。

:code:`StateMonitor` 与 :code:`SpikeMonitor` 的建立方式相同， :code:`StateMonitor` 是神经元及网络连接等的一般状态量\
的监视，而 :code:`SpikeMonitor` 是针对脉冲发放频率的监视：

.. code-block:: python

    self.mon_V = spaic.StateMonitor(self.layer1, 'V')
    self.mon_O = spaic.StateMonitor(self.input, 'O')
    self.spk_O = spaic.SpikeMonitor(self.layer1, 'O')

在监视器初始化中，我们可以指定如下参数：

- **target** -- 需要监视的对象，对于StateMonitor可以是NeuronGroup、Connection等任何包含变量的网络模块，对于SpikeMonitor一般是NeuronGroup、Encoder等具有脉冲发放的模块
- **var_name** -- 需要监视的变量名，需要是监视对象具有的变量，比如神经元的膜电压 'V'
- **index** -- 检测变量的索引值，例如一层神经集群中选择某几个神经元进行记录，可以使用 index=[1,3,4,...]，默认为整个变量全部记录
- **dt** -- 监视器的采样时间间隔，默认与仿真步长相同
- **get_grad** -- 是否需要记录梯度，True为需要梯度，False为不需要，默认为False
- **nbatch** -- 是否需要记录多个Batch的数据，True则会保存多次run的数据，False则每次run覆盖数据，默认为False

在:code:`StateMonitor`和:code:`SpikeMonitor`两种监视器中具有如下通用的函数接口：
- **monitor_on** -- 将监视器设置为开启记录状态。监视器创建后默认是开启状态。
- **monitor_off** -- 将监视器设置为关闭记录状态。
- **clear** -- 清除监视器当前记录的所有数据。

这两个监视器的区别在于，:code:`StateMonitor` 中存储了五个数据：

- **nbatch_times** -- 将会存储所有批次的时间步信息，数据的shape结构为(第几批次，第几个时间步)
- **nbatch_values** -- 将会存储所有批次的目标层的监视参数的情况，数据的shape结构为(第几批次，第几个神经元，第几个时间步，batch中的第几个样本)
- **times** -- 将会存储当前批次的时间步信息，数据的shape结构为(第几个时间步)
- **values** -- 将会存储当前批次的目标层的监视参数的变量，数据的shape结构为(本batch中第几个样本，第几个神经元，第几个时间步)
- **tensor_values** -- 将会存储当前批次的目标层的监视的原Tensor变量，数据的shape结构为(本batch中第几个样本，第几个神经元，第几个时间步)
- **grad** -- 将会存储当前批次的目标变量的梯度情况，数据的shape与values的shape结构相同

而 :code:`SpikeMonitor` 中存储着另外四个数据：

- **spk_index** -- 存储着当前批次脉冲发放的神经元的编号
- **spk_times** -- 存储着当前批次脉冲发放的时刻信息
- **time** -- 存储着当前批次的时间步的信息
- **time_spk_rate** -- 存储着当前批次的目标层的瞬时发放频率
- **spk_rate** -- 储存当前批次的目标层的平均发放率
- **spk_count** -- 储存当前批次的目标层每个神经元的脉冲个数，数据结构为(本batch中第几个样本，第几个神经元）
使用示例:

.. code-block:: python

    time_line = Net.mon_V.times  # 取layer1层时间窗的坐标序号
    value_line = Net.mon_V.values[0][0]  # 取本batch中layer1层第一个样本的第一个神经元的整个时间窗内的电压变化数据
    input_line = Net.mon_O.values[0][0]  # 取本batch中input层第一个样本的第一个神经元的整个时间窗内的脉冲发放情况

    # 由于初始化时nbatch为False，默认只有单个批次
    output_line_index = Net.spk_O.spk_index[0] + 1.2  # 取本batch中第一个样本的脉冲发放的index信息，由于只有单个神经元，增加数值1.2调整脉冲点的位置
    output_line_time = Net.spk_O.spk_times[0]  # 取本batch中第一个样本的脉冲发放的时刻信息

    plt.subplot(2, 1, 1)
    plt.title('Monitor Example Appearance')
    plt.plot(time_line, value_line, label='V')
    plt.scatter(output_line_time, output_line_index, s=40, c='r', label='Spike')

    plt.ylabel("Membrane potential")
    plt.ylim((-0.1, 1.5))
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(time_line, input_line, label='input spike')
    plt.xlabel("time")
    plt.ylabel("Current")
    plt.legend()


最后的结果如图所示:

    .. image:: _static/monitor_VO_Appearance.png
