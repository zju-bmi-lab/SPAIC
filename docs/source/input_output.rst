输入输出
============
数据加载(Dataloader)
-------------------------------
:code:`Dataloader` 是数据集读取的接口，该接口的目的是将自定义的Dataset根据 :code:`batch_size` 大小、\
是否shuffle等封装成一个 :code:`batch_size` 大小的数组，用于网络的训练。

:code:`Dataloader` 由数据集和采样器组成，初始化参数如下：

- **dataset(Dataset)** -- 传入的数据集
- **batch_size(int, optional)** -- 每个batch的样本数, 默认为1
- **shuffle(bool, optional)** -- 在每个epoch开始的时候，对数据进行重新排序，默认为False
- **sampler(Sampler, optional)** -- 自定义从数据集中取样本的策略
- **batch_sampler(Sampler, optional)** -- 与sampler类似，但是一次只返回一个batch的索引

- **collate_fn(callable, optional)** -- 将一个list的sample组成一个mini-batch的函数
- **drop_last(bool, optional)** -- 如果设置为True，对于最后一个batch，如果样本数小于batch_size就会被扔掉，比如batch_size设置为64，而数据集只有100个样本，那么训练的时候后面的36个就会被扔掉。如果为False（默认），那么会继续正常执行，只是最后的batch_size会小一点。

以导入MNIST数据集为例：

.. code-block:: python

    root = './Datasets/MNIST' # 数据集的地址
    train_set = dataset(root, is_train=True)   # 训练集
    test_set = dataset(root, is_train=False)   # 测试集
    bat_size = 20
    # 创建DataLoader
    train_loader = spaic.Dataloader(train_set, batch_size=bat_size, shuffle=True)
    test_loader = spaic.Dataloader(test_set, batch_size=bat_size, shuffle=False)


.. note::

   需要注意的是：\
    1、创建 :code:`Dataloader` 时如果指定了 :code:`sampler` 这个参数，那么 :code:`shuffle` 必须为False

    2、如果指定 :code:`batch_sampler` 这个参数，那么 :code:`batch_size` ，:code:`shuffle` ，:code:`sampler`， :code:`drop_last` 就不能再指定了