Backend
=================
:code:`Backend` is the core component of the backend of the SPAIC platform and is responsible for the overall running \
simulation of the network. :code:`dt` and :code:`runtime` are the two most important parameters in backend, \
representing the time step and the length of the time window of the simulation, respectively. And :code:`time` \
represents the moment that currently simulated, and :code:`n_time_step` represents the time step that is \
currently simulated.

The functions available to users in :code:`Backend` are:

- set_runtime: sets the length of the time window, or the length of the simulation
- add_variable: adds variables to the backend. When customizing the algorithm, you need to add new variables to the :code:`Backend`
- add_operation: adds a new calculation formula to the backend for custom algorithms, neurons and other operations, and needs to follow a certain format
- register_standalone: registers independent operations, mainly used to add some operations that are not supported by the platform to the backend
- register_initial: registers the initialization operation. The calculation in the initialization operation will be calculated once at the beginning of each time window, instead of every :code:`dt` as the time step runs.


add_variable
------------------
When using :code:`add_variable` , the parameters that must be added are  :code:`name` 与 :code:`shape` , \
and the optional parameters are :code:`value`, :code:`is_parameter`, :code:`is_sparse`, :code:`init`, \
 :code:`min` and :code:`max` 。

:code:`name` 决定了该变量在后端存储时的 :code:`key`，而 :code:`shape` 决定了维度, code:`value` 代表了这个变量的值，\
:code:`init` 决定了该变量在每一次初始化之后的值
:code:`is_parameter` 这个参数决定了该变量是否是可训练的参数，若为 :code:`true` ，则可以添加