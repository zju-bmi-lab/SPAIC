模拟后端 Backend
=================
``Backend`` 是 **SPAIC** 平台后端的核心部件，负责网络整体的运行模拟。 :code:`dt` 与 :code:`runtime` 是 ``backend`` 中最为重要的两个参数，分别\
代表了模拟的时间步与时间窗的长度。而 :code:`time` 代表了当前模拟到的时刻，:code:`n_time_step` 代表了当前模拟到的时间步。

``Backend`` 中可供用户使用的函数有：

- **set_runtime** -- 设置时间窗的长度，或者称之为模拟的时间长度
- **add_variable** -- 向后端添加变量，用于自定义算法时，需要添加新的变量至backend中
- **add_operation** -- 向后端添加新的计算式，用于自定义算法、神经元等操作中，需要依照一定的格式
- **register_standalone** -- 注册独立操作，主要用于向后端添加一些平台不支持的操作
- **register_initial** -- 注册初始化操作，初始化操作中的计算将会在每个时间窗的最开始运算一次，而不会随着时间步的运行每个dt都进行运算


add_variable
------------------
使用 :code:`add_variable` 时，必须添加的参数有 :code:`name` 与 :code:`shape` ，可以选择性输入的参数有 :code:`value`、 \
:code:`is_parameter` 、 :code:`is_sparse`、 :code:`init`、 :code:`min` 与 :code:`max` 。

:code:`name` 决定了该变量在后端存储时的 :code:`key`，而 :code:`shape` 决定了维度, :code:`value` 代表了这个变量的值，\
:code:`init` 决定了该变量在每一次初始化之后的值， :code:`is_parameter` 这个参数决定了该变量是否可训练。

add_operation
------------------
使用 :code:`add_operation` 时，必须添加的参数有 :code:`op` , :code:`op` 这个参数传入的是具体需要后端进行运算的计算式。\
使用该函数之后，将会把用户所提供的计算式添加至整体的网络计算图中，再经由后端的构建函数，具体安排计算流程。本函数主要用于自定义计算， \
用于将用户自定义的函数计算式添加至计算流程中。

