# -*- coding: utf-8 -*-
"""
Created on 2020/8/6
@project: SPAIC
@filename: Backend
@author: Hong Chaofei
@contact: hongchf@gmail.com
@description:
定义网络仿真使用的backend，如 Pytorch, Tensorflow, CUDA, 达尔文芯片等，以及相应的微分方程求解方法比如 Euler, 2阶 Runge-Kutta等
"""
from abc import abstractmethod, ABC
from collections import OrderedDict
from ..Network.BaseModule import BaseModule, VariableAgent
from ..Network.DelayQueue import DelayQueue
import numpy as np
import torch

backends = dict()


class Backend(BaseModule, ABC):
    '''
    Basic backend class. All specified backend backend should subclass it.
    The backend is a parameter for the build function and becomes an attribute of all objects defined
    in the frontend backend network in building process. These objects build their initial data
    and specified operations into the attributes of backend, according to _variables
    and _operations respectively. The data will update in each step according the computation graph.
    Args:
        dt (float, optional): the length of a backend timestep, in millisecond.
    Attributes:
        device (str): the desired device of returned tensor. Its value can be 'cpu' or 'cuda'. If None, uses
            the current device for the default tensor type.
        builded (bool): whether the object defined in the frontend backend network has been builded.
        time (float): current backend time, in millisecond.
        n_time_step (int): the num of current time step.

        _variables (OrderedDict): records all variables from the build function of frontend objects.
        _parameters_dict (OrderedDict): records the variables to be trained.
        _InitVariables_dict (OrderedDict): reserves a copy of the initialization variables for initialization.
        _graph_var_dicts (dict): has following format: {'variables_dict': self._variables, 'temp_dict': dict(), 'update_dict': dict(), 'reduce_dict': dict()},
            recording the intermediate value of variables in computation progress.

        basic_operate (dict): dictionary of basic operators, mapping from operator names using in frontend to
            the funtion objects implemented in backend.
        _operations (list): records all basic operations from the build function of frontend objects, each of
            which has following format: [ret_var_name: str, operation_name, input_var_name1: str, input_var_name2 :str, ...].
        _graph_operations (list): redefine each basic operation, that is, add the corresponding keyword in the _graph_var_dicts to each variable,
            which has following format: [(dict_type, ret_var_name), operation_name, [(dict_type1, input_var_name1),(dict_type2, input_var_name2),...]].
        _standalone_operations (list): records all standalone operations from the build function of frontend objects,
            each of which has following format: (ret_var_name: str, function, input_var_names: list).
        _initial_operations (list): records all initial operations from the build function of frontend objects, each of
            which has following format: (ret_var_name: str, function, input_var_names: list).

        _monitors (list): records all monitors defined in fronted network through build function of Monitor object.

    Methods:
        build_graph: build a computation graph before performing the calculation.
        graph_update_step: update value of _graph_var_dicts.
        initial_step: initialize network variables.
        update_step: update the return variables of standalone operations and basic operations and current backend time.
        r_update_step: update the return variables of basic operations without using graph_update_step().
        add_variable: add variables from front objects to _variables of Backend.
        add_backend_variable: add variables according to the specified backend.
        add_operation: add basic operations from front objects to _operations of Backend.
        register_standalone: add standalone operations from front objects to _standalone_operations of Backend.
        register_initial: add initial operations from front objects to _initial_operations of Backend.
    '''
    basic_operate = dict()
    param_init_operate = dict()  # -> param_init_operate

    backend_name = 'None'
    def __init__(self, dt=0.1):
        super(Backend, self).__init__()
        self.device = None
        self.runtime = None
        self.builded = False
        self.dt = dt  # the length of a backend timestep
        self.time = 0.0  # current backend time
        self.n_time_step = 0  # the num of current time step
        self._batch_size = 1

        self._variables = dict()  # build from orderedDict to Tuple
        self._parameters_dict = dict()
        self._clamp_parameter_dict = dict()
        self._delay_dict = dict()  # store conduction delays
        self._SparseVariables_dict = dict()
        self._InitVariables_dict = dict()

        self._operations = list()
        self._standalone_operations = list()
        self._initial_operations = list()

        self._monitors = list()  # TODO: need to add to update
        self._stored_states = dict()  # TODO: store network self._variables in the dict

        self.basic_operate['threshold'] = self.threshold
        self.basic_operate['var_linear'] = self.var_linear
        self.basic_operate['mat_linear'] = self.mat_linear
        self.basic_operate['mat_mult_weight'] = self.mat_mult_weight
        self.basic_operate['mat_mult_pre'] = self.mat_mult_pre
        self.basic_operate['mat_mult'] = self.mat_mult
        self.basic_operate['bmm'] = self.bmm
        self.basic_operate['ger'] = self.ger
        self.basic_operate['sparse_mat_mult_weight'] = self.sparse_mat_mult_weight
        self.basic_operate['var_mult'] = self.var_mult
        self.basic_operate['add'] = self.add
        self.basic_operate['minus'] = self.minus
        self.basic_operate['div'] = self.div
        self.basic_operate['cat'] = self.cat
        self.basic_operate['stack'] = self.stack
        self.basic_operate['permute'] = self.permute
        self.basic_operate['view'] = self.view
        self.basic_operate['equal'] = self.equal
        self.basic_operate['unsqueeze'] = self.unsqueeze

        self.basic_operate['reduce_sum'] = self.reduce_sum
        self.basic_operate['conv_2d'] = self.conv_2d
        self.basic_operate['relu'] = self.relu

        self.basic_operate['sin'] = self.sin
        self.basic_operate['cos'] = self.cos
        self.basic_operate['tan'] = self.tan
        self.basic_operate['log'] = self.log
        self.basic_operate['log2'] = self.log2
        self.basic_operate['log10'] = self.log10

        self.basic_operate['conv_max_pool2d'] = self.conv_max_pool2d
        self.basic_operate['conv_avg_pool2d'] = self.conv_avg_pool2d
        self.basic_operate['conv_add_bias'] = self.conv_add_bias
        self.basic_operate['max_pool2d'] = self.max_pool2d
        self.basic_operate['avg_pool2d'] = self.avg_pool2d
        self.basic_operate['dropout'] = self.dropout
        self.basic_operate['reshape_mat_mult'] = self.reshape_mat_mult
        self.basic_operate['exp'] = self.exp
        self.basic_operate['mult_sum_weight'] = self.mult_sum_weight
        self.basic_operate['im2col_indices'] = self.im2col_indices
        self.basic_operate['conv2d_flatten'] = self.conv2d_flatten
        self.basic_operate['feature_map_flatten'] = self.feature_map_flatten

        self.param_init_operate['uniform'] = self.uniform
        self.param_init_operate['normal'] = self.normal
        self.param_init_operate['xavier_uniform'] = self.xavier_uniform
        self.param_init_operate['xavier_noraml'] = self.xavier_normal
        self.param_init_operate['kaiming_uniform'] = self.kaiming_uniform
        self.param_init_operate['kaiming_normal'] = self.kaiming_normal
        self.param_init_operate['zero_init'] = self.zero_init

        # self._graph_var_dicts = {'variables_dict': self._variables, 'temp_dict': dict(), 'update_dict': dict(),
        #                          'reduce_dict': dict()}

        self._graph_operations = list()
        self._push_operations = list()
        self._fetch_operations = list()

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def get_batch_size(self):
        return self._batch_size

    def set_runtime(self, runtime):
        self.runtime = runtime

    def build_graph(self):
        '''
        Build a computation graph before performing the calculation.
        Note that only the basic operations are redefiend into the _graph_operations list. The format of _graph_operations is as follows:
        [(dict_type, ret_var_name), operation_name, [(dict_type1, input_var_name1),(dict_type2, input_var_name2),...]].
        Traverse all basic operations and add the corresponding keyword in the _graph_var_dicts as dict_type to each variable in basic operation.
        '''

        variables_index = {k: i for i, k in enumerate(self._variables.keys())}

        self.initial_step()

        operation_type = 'update_dict or temp_dict or reduce_dict'
        # traverse basic operations

        fetch_operations = []
        push_operations = []
        graph_operations = []
        # self._graph_operations = list()
        # self._push_operations = list()
        # self._fetch_operations = list()

        for op in self._operations:
            if len(op[0]) == 0 and len(op[2]) == 0:
                # functions with no input and output will not push into the computation graph
                raise ValueError(" Operation lacks both input and output can't be build")
            elif len(op[0]) == 0:
                fetch_operations.append(op)
            elif len(op[2]) == 0:
                push_operations.append(op)
            else:
                graph_operations.append(op)

        ################################
        ##  for push_operation build  ##
        ################################
        update_dict = dict()
        reduce_dict = dict()

        for ind, op in enumerate(push_operations):
            outputs = []
            label_outputs = []
            # if the operation return one variable, then it is appended into a list, to accordant with multi-variable returns
            if len(op[0]) == 1:
                outputs.append(op[1]())
            else:
                outputs = op[1]()
            # return variable is a list
            for ind, var_name in enumerate(op[0]):
                if var_name in self._variables:
                    # when the same ret_var_name occurs more than once, op[0] is added to the reduce_dict of _graph_var_dicts
                    if var_name in update_dict:
                        reduce_dict[var_name] = [update_dict[var_name], outputs[ind]]
                        label_outputs.append(('reduce_dict', var_name))
                        # # add op[0] into graph: reduce_dict
                        # self._graph_var_dicts['reduce_dict'][op[0]] = []
                        # revise the first reduce operation
                        for gop in self._push_operations:
                            tmp_label_outputs = gop[0]
                            for tmp_ind, tmp_label in enumerate(tmp_label_outputs):
                                if tmp_label[1] == var_name:
                                    tmp_label_outputs[tmp_ind] = ('reduce_dict', var_name)
                                    break
                        del update_dict[var_name]
                    elif var_name in reduce_dict:
                        reduce_dict[var_name].append(outputs[ind])
                        label_outputs.append(('reduce_dict', var_name))
                    else:
                        # In the push_operation, new data is directly pushed to update_dict, as
                        # there is no need to remain the last step variable value
                        update_dict[var_name] = outputs[ind]
                        label_outputs.append(('update_dict', var_name))
                else:
                    raise ValueError("No state variable to get the input ")

            # add the operation to built graph
            self._push_operations.append([label_outputs, op[1], []])

        # for var_name in reduce_dict:
        #     # add the reduce_sum operation into the graph
        #     self._graph_operations.append(
        #         [[('update_dict', var_name)], self.reduce_sum_update, [('reduce_dict', var_name)]])

        #################################
        ##  for graph_operation build  ##
        #################################
        temp_dict = dict()
        # update_dict = dict()
        # reduce_dict = dict()
        temp_reduce_sum_ops = []

        for ind, op in enumerate(graph_operations):
            inputs = []
            label_inputs = []
            for var_name in op[2]:
                # try:
                #     var_name in self._variables
                # except:
                #     a = 1

                if '[updated]' in var_name:
                    var_name = var_name.replace("[updated]", "")

                    if var_name in update_dict:
                        inputs.append(update_dict[var_name])
                        label_inputs.append(('update_dict', var_name))
                    elif var_name in reduce_dict:
                        # if the reduce_dict[var_name] is frozen: do reduce_sum operation before this op, and put the value to update_dict
                        value = self.reduce_sum(self.stack(reduce_dict[var_name]))
                        inputs.append(value)
                        label_inputs.append(('update_dict', var_name))
                        temp_reduce_sum_ops.append((var_name, len(reduce_dict[var_name])))
                        # add the reduce_sum operation into the graph
                        self._graph_operations.append(
                            [[('update_dict', var_name)], self.reduce_sum_update, [('reduce_dict', var_name)]])
                    elif var_name in self._variables:
                        inputs.append(self._variables[var_name])
                        label_inputs.append(('variables_dict', var_name))

                    else:
                        raise ValueError(" No State Variable [%s] in the update_dict" % var_name)
                elif var_name in self._variables:
                    inputs.append(self._variables[var_name])
                    label_inputs.append(('variables_dict', var_name))

                elif var_name in temp_dict:
                    inputs.append(temp_dict[var_name])
                    label_inputs.append(('temp_dict', var_name))
                else:
                    raise ValueError(" No State Variable [%s] in the variable dict" % var_name)

            outputs = []
            label_outputs = []
            if len(op[0]) == 0:
                self.var_check(op[1], inputs)
                op[1](*inputs)
            else:
                self.var_check(op[1], inputs)
                if len(op[0]) == 1:
                    outputs.append(op[1](*inputs))
                else:
                    outputs = op[1](*inputs)
                for ind, var_name in enumerate(op[0]):
                    if var_name in self._variables:
                        # when the same ret_var_name occurs more than once, op[0] is added to the reduce_dict of _graph_var_dicts
                        if var_name in update_dict:
                            reduce_dict[var_name] = [update_dict[var_name], outputs[ind]]
                            label_outputs.append(('reduce_dict', var_name))
                            # # add op[0] into graph: reduce_dict
                            # self._graph_var_dicts['reduce_dict'][op[0]] = []
                            # revise the first reduce operation
                            InGop = True
                            for pop in self._push_operations:
                                tmp_label_outputs = pop[0]
                                for tmp_ind, tmp_label in enumerate(tmp_label_outputs):
                                    if tmp_label[1] == var_name:
                                        tmp_label_outputs[tmp_ind] = ('reduce_dict', var_name)
                                        InGop = False
                                        break
                            if InGop:
                                for gop in self._graph_operations:
                                    tmp_label_outputs = gop[0]
                                    for tmp_ind, tmp_label in enumerate(tmp_label_outputs):
                                        if tmp_label[1] == var_name:
                                            tmp_label_outputs[tmp_ind] = ('reduce_dict', var_name)
                                            break
                            del update_dict[var_name]
                        elif var_name in reduce_dict:
                            reduce_dict[var_name].append(outputs[ind])
                            label_outputs.append(('reduce_dict', var_name))
                        else:
                            update_dict[var_name] = outputs[ind]
                            label_outputs.append(('update_dict', var_name))
                    else:
                        temp_dict[var_name] = outputs[ind]
                        label_outputs.append(('temp_dict', var_name))

            # add the operation to built graph

            self._graph_operations.append([label_outputs, op[1], label_inputs])


        for reduce_op in temp_reduce_sum_ops:
            reduce_len = len(reduce_dict[reduce_op[0]])
            if reduce_len != reduce_op[1]:
                raise ValueError(
                    "Can't use [updated] tag for variable: %s, as it is a reduce_dict variable which is have updating conflict" %
                    reduce_op[0])
            else:
                del reduce_dict[reduce_op[0]]
        # for reduced variables that not used within [update]
        for var_name in reduce_dict:
            # add the reduce_sum operation into the graph
            self._graph_operations.append(
                [[('update_dict', var_name)], self.reduce_sum_update, [('reduce_dict', var_name)]])

        #################################
        ##  for fetch_operation build  ##
        #################################
        for ind, op in enumerate(fetch_operations):
            inputs = []
            label_inputs = []
            for var_name in op[2]:
                if '[updated]' in var_name:
                    # there is no need to have updated tag, as all variables computed in graph_operation have benn updated
                    var_name = var_name.replace("[updated]", "")
                if var_name in self._variables:
                    inputs.append(self._variables[var_name])
                    label_inputs.append(('variables_dict', var_name))
                # elif var_name in temp_dict:
                #     inputs.append(temp_dict[var_name])
                #     label_inputs.append(('temp_dict', var_name))
                else:
                    raise ValueError(" No State Variable [%s] in the update_dict" % var_name)

            self.var_check(op[1], inputs)
            op[1](*inputs)

            # add the operation to built graph
            self._fetch_operations.append([[], op[1], label_inputs])

        # self._variables.update(update_dict)
        for ii in range(len(self._graph_operations)):
            self._graph_operations[ii] = tuple(self._graph_operations[ii])
        self._graph_operations = tuple(self._graph_operations)

    def var_check(self, op, *args):
        '''
        For specified operation, check the type or the shape of input variables.
        '''
        if op == 'mat_mult':
            if args[0][0].shape[1] != args[0][1].shape[0]:
                raise ValueError("%s and %s do not match" % (args[0].shape, args[1].shape))
        pass

    def graph_update_step_r(self):

        for op in self._graph_operations:
            inputs = []
            for var in op[2]:
                inputs.append(self._graph_var_dicts[var[0]][var[1]])

            if op[0][0] is None:
                op[1](*inputs)
            elif op[0][0] == 'reduce_dict':
                self._graph_var_dicts['reduce_dict'][op[0][1]].append(op[1](*inputs))
            else:
                self._graph_var_dicts[op[0][0]][op[0][1]] = op[1](*inputs)

            # if '[updated]' in op[0][1]:
            #     op_name = op[0][1].strip('[updated]')
            #     if op_name in self._graph_var_dicts['update_dict'] and op_name in self._graph_var_dicts['variables_dict']:
            #         self._graph_var_dicts['update_dict'][op_name] = self._graph_var_dicts['temp_dict'][op[0][1]]  # 更新返回名中带[updated]的变量的值

        return  # tuple(self._graph_var_dicts['variables_dict'].values())

    def graph_update_step(self, variables, update_dict, reduce_dict):
        temp_dict = dict()
        # update_dict = dict()
        # reduce_dict = dict()

        for op in self._graph_operations:
            # for inputs
            inputs = []
            for var in op[2]:
                if var[0] == 'variables_dict':
                    inputs.append(variables[var[1]])
                elif var[0] == 'temp_dict':
                    inputs.append(temp_dict[var[1]])
                elif var[0] == 'update_dict':
                    inputs.append(update_dict[var[1]])
                elif var[0] == 'reduce_dict':
                    inputs.append(reduce_dict[var[1]])
            # compute the operation
            result = op[1](*inputs)
            if len(op[0]) == 1: result = [result]
            # assign the result variables
            for ind, var in enumerate(op[0]):
                if var[0] == 'temp_dict':
                    temp_dict[var[1]] = result[ind]
                elif var[0] == 'update_dict':
                    update_dict[var[1]] = result[ind]
                elif var[0] == 'reduce_dict':
                    if var[1] in reduce_dict:
                        reduce_dict[var[1]].append(result[ind])
                    else:
                        reduce_dict[var[1]] = [result[ind]]

        return update_dict

    def push_update_step(self):
        reduce_dict = dict()
        update_dict = dict()
        for op in self._push_operations:
            result = op[1]()
            if len(op[0]) == 1: result = [result]
            for ind, var in enumerate(op[0]):
                if var[0] == 'update_dict':
                    update_dict[var[1]] = result[ind]
                elif var[1] in reduce_dict:
                    reduce_dict[var[1]].append(result[ind])
                else:
                    reduce_dict[var[1]] = [result[ind]]
        return update_dict, reduce_dict

    def fetch_update_step(self):
        for op in self._fetch_operations:
            # for inputs
            inputs = []
            for var in op[2]:
                inputs.append(self._variables[var[1]])
            op[1](*inputs)

    def initial_step(self):
        '''
        Initialize network variables.
        '''

        # Initialize the current backend time and the num of time step
        self.last_time = 0.0
        self.time = 0.0  # current backend time
        self.n_time_step = 0
        for key, value in self._variables.items():
            if '[stay]' in key:
                self._InitVariables_dict[key] = self._variables[key]

        # Initialize untrainable variables
        self._variables.clear()

        for key, value in self._InitVariables_dict.items():
            self._variables[key] = value

        # Initialize the trainable parameters
        for key, clamp_code in self._clamp_parameter_dict.items():
            clamp_code[0](*clamp_code[1])

        for key, value in self._parameters_dict.items():
            self._variables[key] = value

        for key, value in self._SparseVariables_dict.items():
            index_name = key + '_sparse_index'
            value_name = key + '_sparse_value'
            shape_name = key + '_sparse_shape'
            if index_name in self._variables.keys() and value_name in self._variables.keys():
                self._variables[key] = self.sparse_to_dense(index_name, value_name, shape_name)

        # Initialize the record of Monitor
        for monitor in self._monitors:
            monitor.init_record()

        # Traverse initial operations
        for op in self._initial_operations:
            inputs = []
            for var_name in op[2]:
                if var_name in self._variables:
                    inputs.append(self._variables[var_name])
                else:
                    raise ValueError(" No State Variable [%s] in the variable dict" % var_name)
            if op[0] is None:
                op[1](*inputs)
            else:
                self._variables[op[0]] = op[1](*inputs)

        # Change intial variable's batch_size
        for key in self._variables.keys():
            if hasattr(self._variables[key], 'shape'):
                shape = self._variables[key].shape
                if self._variables[key].ndim > 1 and shape[0] == 1 and (key not in self._parameters_dict):
                    expand_shape = -np.ones_like(shape, dtype=int)
                    expand_shape[0] = self._batch_size
                    self._variables[key] = self._variables[key].expand(tuple(expand_shape))

            # if '{O}' in key:
            #     o_shape = self._variables[key].shape
            #
            #     shape = []
            #     for s in o_shape:
            #         if s != 1:
            #             shape.append(s)
            #         else:
            #             shape.append(self._batch_size)
            #     self._variables[key] = torch.zeros(shape, dtype=torch.float32, device=self.device)

    def clear_step(self):
        '''

        Returns:

        '''

        self._operations = list()
        self._graph_operations = list()
        self._push_operations = list()
        self._fetch_operations = list()

    def initial_continue_step(self):
        '''
        Initialize network for continuous run.
        '''

        self.last_time = self.time

    def update_step(self):
        '''
        Update the return variables of standalone operations and basic operations and current backend time.
        Returns:
            tuple(self._variables.values())
        '''

        # push input data
        update_dict, reduce_dict = self.push_update_step()

        # static graph compuation
        update_dict = self.graph_update_step(self._variables, update_dict, reduce_dict)

        # Update time and state variables
        self.n_time_step += 1
        self.time = round(self.n_time_step * self.dt, 2)
        self._variables.update(update_dict)

        # fetch output data
        self.fetch_update_step()

        # Record Variables
        for monitor in self._monitors:
            monitor.update_step(self._variables)

        return tuple(self._variables.values())

    def update_time_steps(self):
        while (self.runtime > self.time - self.last_time):
            self.update_step()

    def r_update_step(self):
        '''
        Update the return variables of basic operations without using graph_update_step().
        Returns:
            tuple(self._variables.values())
        '''

        reduce_dict = dict()
        self._graph_var_dicts['update_dict'].clear()
        self._graph_var_dicts['temp_dict'].clear()
        self._graph_var_dicts['reduce_dict'].clear()

        # Traverse standalone operations
        for op in self._standalone_operations:
            inputs = []
            for var_name in op[2]:
                if 'pytorch' in backends:
                    inputs.append(self._variables[var_name])
                else:
                    inputs.append(self.to_numpy(self._variables[var_name]))

            if op[0] is None:
                op[1](*inputs)
            else:
                if 'pytorch' in backends:
                    self._variables[op[0]] = op[1](*inputs)
                else:
                    self._variables[op[0]] = self.to_tensor(op[1](*inputs))

        # update one time_step
        for op in self._operations:
            if op[0] in self._graph_var_dicts['variables_dict']:
                inputs = []
                for var_name in op[2:]:
                    if '[updated]' in var_name:
                        var_name = var_name.replace("[updated]", "")
                        if var_name in self._graph_var_dicts['update_dict']:
                            inputs.append(self._graph_var_dicts['update_dict'][var_name])
                        else:
                            raise ValueError(" No State Variable [%s] in the update_dict" % var_name)
                    elif var_name in self._graph_var_dicts['variables_dict']:
                        inputs.append(self._graph_var_dicts['variables_dict'][var_name])
                    elif var_name in self._graph_var_dicts['temp_dict']:
                        inputs.append(self._graph_var_dicts['temp_dict'][var_name])
                    else:
                        raise ValueError(" No State Variable [%s] in the variable dict" % var_name)

                if op[0] in self._graph_var_dicts['update_dict']:
                    if op[0] in self._graph_var_dicts['reduce_dict']:
                        self._graph_var_dicts['reduce_dict'][op[0]].append(op[1](*inputs))
                    else:
                        self._graph_var_dicts['reduce_dict'][op[0]] = [self._graph_var_dicts['update_dict'][op[0]],
                                                                       op[1](*inputs)]
                else:
                    self._graph_var_dicts['update_dict'][op[0]] = op[1](*inputs)
                    pass

            else:
                inputs = []
                for var_name in op[2:]:
                    if '[updated]' in var_name:
                        var_name = var_name.replace("[updated]", "")
                        if var_name in self._graph_var_dicts['update_dict']:
                            inputs.append(self._graph_var_dicts['update_dict'][var_name])
                        else:
                            raise ValueError(" No State Variable [%s] in the update_dict" % var_name)
                    elif var_name in self._graph_var_dicts['variables_dict']:
                        inputs.append(self._graph_var_dicts['variables_dict'][var_name])
                    elif var_name in self._graph_var_dicts['temp_dict']:
                        inputs.append(self._graph_var_dicts['temp_dict'][var_name])
                    else:
                        raise ValueError(" No State Variable [%s] in the variable dict" % var_name)
                self._graph_var_dicts['temp_dict'][op[0]] = op[1](*inputs)

                if '[updated]' in op[0]:
                    op_name = op[0].replace("[updated]", "")
                    if op_name in self._graph_var_dicts['update_dict']:
                        self._graph_var_dicts['update_dict'][op_name] = self._graph_var_dicts['temp_dict'][
                            op[0]]  # update the variable in update_dict
                    else:
                        raise ValueError(" No State Variable [%s] in the update_dict" % var_name)

        # Update reduce_dict into update_dict
        for key, value in reduce_dict.items():
            value = self.stack(value)
            self._graph_var_dicts['update_dict'][key] = self.reduce_sum(value)
            self._graph_var_dicts['update_dict'][key] = []

        # update time
        self.n_time_step += 1
        self.time = round(self.n_time_step * self.dt, 2)

        self._graph_var_dicts['variables_dict'].update(self._graph_var_dicts['update_dict'])

        # Record Variables
        for monitor in self._monitors:
            monitor.update_step(self._graph_var_dicts)

        return tuple(self._variables.values())

    def reduce_sum_update(self, value):
        reduced = self.reduce_sum(self.stack(value))
        return reduced

    def get_varialble(self, name):
        if name in self._parameters_dict:
            return self._parameters_dict[name]
        elif name in self._variables:
            return self._variables[name]
        else:
            raise ValueError("not found variable:%s in the backend"%name)

    def set_variable_value(self, name, value, is_parameter):
        '''
        Set the backend value, in specific Backend
        Args:
            name:
            value:
            is_parameter:

        Returns:

        '''
        NotImplementedError()

    def add_variable(self, name, shape, value=None, is_parameter=False, is_sparse=False, init=None, init_param=None,
                     min=None, max=None, is_constant=False):
        '''
        Add variables from front objects to _variables of Backend and get copies to assign to _parameters_dict and _InitVariables_dict.
        Args:
            name (str): the name of the added variable
            shape (list, int): the shape of the variable
            value (optional): the value of the variable
            is_parameter (bool, optional): whether the variable is trainable
            init (optinal):
        '''
        if is_parameter:
            self._parameters_dict[name] = self.add_backend_variable(name, shape, value, grad=True, is_sparse=is_sparse,
                                                                    init=init, init_param=init_param)
            if min is not None and max is not None:
                self._clamp_parameter_dict[name] = (self.clamp_, [self._parameters_dict[name], min, max])
            elif min is not None:
                self._clamp_parameter_dict[name] = (self.clamp_min_, [self._parameters_dict[name], min])
            elif max is not None:
                self._clamp_parameter_dict[name] = (self.clamp_max_, [self._parameters_dict[name], max])


        # 稀疏矩阵weight非叶子节点，反传的时候更新的是weight中的value,但前向计算的时候用的是weight,所以对于稀疏矩阵要单独用个dict记录以便初始化
        elif is_sparse:
            self._SparseVariables_dict[name] = self.add_backend_variable(name, shape, value, grad=True,
                                                                         is_sparse=is_sparse, init=init,init_param=init_param)
        elif is_constant:
            self._InitVariables_dict[name] = value
            self._variables[name] = value
        else:
            self._InitVariables_dict[name] = self.add_backend_variable(name, shape, value, grad=False,
                                                                       is_sparse=is_sparse, init=init,
                                                                       init_param=init_param)

        var_agent = VariableAgent(self, name, is_parameter)
        return var_agent

    def has_variable(self, name):
        if name in self._variables:
            return True
        elif name in self._InitVariables_dict:
            return True
        elif name in self._parameters_dict:
            return True
        elif name in self._SparseVariables_dict:
            return True
        else:
            return False

    def add_delay(self, var_name, max_delay):
        max_len = int(max_delay / self.dt)
        if var_name in self._delay_dict:
            if self._delay_dict[var_name].max_len < max_len:
                self._delay_dict[var_name].max_len = max_len
        else:
            self._delay_dict[var_name] = DelayQueue(var_name, max_len, self)
            self.register_initial(None, self._delay_dict[var_name].initial, [var_name, ])
            self.register_standalone(var_name, self._delay_dict[var_name].push, [var_name, ])
        return self._delay_dict[var_name]


    @abstractmethod
    def add_backend_variable(self, name, shape, value=None, grad=False, is_sparse=False, init=None, init_param=None):
        '''
        This method will be overwritten by different subclasses to add variables to _variables of specified backend.
        Args:
            name (str): the name of the added variable
            shape (list, int): the shape of the variable
            value (optional): the value of the variable
            is_parameter (bool, optional): whether the variable is trainable
            init (optinal):
            grad (bool, optional): whether to use grad
        '''
        NotImplementedError()

    @abstractmethod
    def sparse_to_dense(self, index_name, value_name, shape_name):
        '''
        This method will be sparse matrix to dense matrix.
        Args:
            index_name (str)
            value_name (str)
            shape_name (str)
        '''
        NotImplementedError()

    def add_operation(self, op):
        '''
        Add basic operations from front objects to _operations of Backend.
        Args:
            op (list): the operation includes [ret_var_name: str, operation_name, input_var_name1: str, input_var_name2 :str, ...]
        transformed to : [[return_var_names], operation_name, [input_var_names]]
        '''
        if not isinstance(op[0], list):
            op[0] = [op[0]]
        if len(op)==2:
            op.append([])
        elif not isinstance(op[2], list):
            op[2] = op[2:]  # op[2]是list，说明本身就采用了list多输入的结构，如果op[3]还有数值，直接不考虑

        if op[1] in self.basic_operate:
            op[1] = self.basic_operate[op[1]]
            self._operations.append(op)
        elif callable(op[1]):
            self.register_standalone(op[0], op[1], op[2])
        else:
            raise ValueError("No operation %s in basic_operate" % op[1])

    def register_standalone(self, output_names, function, input_names: list):
        '''
        Add standalone operations from front objects to _standalone_operations of Backend.
        Args:
            output_name (str): the name of the return variable of the method
            funtion (): the standalone method
            input_names (list): the name of the arguments of the method
        '''
        # TODO:
        if isinstance(output_names, str):
            output_names = [output_names]
        elif output_names is None:
            output_names = []
        op = [output_names, function, input_names]
        self._operations.append(op)

        # self._standalone_operations.append((output_name, function, input_names))

    def register_initial(self, output_name: str, function, input_names: list):
        '''
        Add initial operations from front objects to _initial_operations of Backend..
        Args:
            output_name (str): the name of the return variable of the method
            funtion (): the standalone method
            input_names (list): the name of the arguments of the method
        '''
        self._initial_operations.append((output_name, function, input_names))

    def store(self, name='default'):
        '''
        Store backend_name and _variables into _stored_states dictionary.
        Args:
            name (str, optional): the name of network state.
        '''
        self._stored_states[name] = (self.backend_name, self._variables)

    def restore(self, name='default'):
        '''
        Restore network state from _stored_states dictionary.
        Args:
            name (str): the name of network state.
        '''
        if name not in self._stored_states:
            raise ValueError("No network state named: %s is stored" % name)
        else:
            stored_backend = self._stored_states[name][0]
            if stored_backend != self.backend_name:
                raise ValueError(
                    "The stored network is run by %s not %s" % (stored_backend, self.backend_name))
            else:
                self._variables = self._stored_states[name]

    def check_key(self, ckey, target_dict):
        cnetname = ckey[:ckey.find('<net>')]
        for key, value in target_dict.items():
            netname = key[:key.find('<net>')]
            break
        ckey = ckey.replace(cnetname, netname)
        if ckey in target_dict.keys():
            return ckey

        import warnings
        warnings.warn('Key error occurs, please check keys.')


        # result = [key for key in target_dict.keys() if key.endswith(variables[variables.find('<net>'):])]
        # if result:
        #     if len(result) > 1:
        #         import warnings
        #         warnings.warn('Given key matchs two variables in the backend dict, choose the first one as default')
        #     result = result[0]
        # return result

    # -------- basic backends operations -----
    @abstractmethod
    def threshold(self, v, v_th):
        '''
        Args:
            v: membrane voltage
            v_th: threshold
        Returns:
            v> v_th
        '''

    @abstractmethod
    def cat(self, x, dim=1):
        '''
        Joining data together along a dimension.
        Note that the total dimension of the data remains the same after cat.
        Args:
            x (list):
            dim (int): the dimension to cat.
        Returns:
            concat(x, dim)
        '''

    @abstractmethod
    def stack(self, x, dim=1):
        '''
        Add new dimension when stack data.
        Args:
            x (list):
            dim (int): the dimension to stack.
        Returns:
            stack(x, dim)
        '''

    @abstractmethod
    def permute(self, x, permute_dim):
        '''
        Parameters
        ----------
        x---> input
        permute_dim---> the dimension index of permute operation
        Returns
        -------
        '''

    @abstractmethod
    def view(self, x, view_dim):
        '''
        Parameters
        ----------
        x---> input
        view_dim---> the shape of view operation
        Returns
        -------
        '''

    def equal(self, x):
        '''
        Parameters
        ----------
        y---> target
        x---> input
        Returns
        -------
        '''
        y = x
        return y

    @abstractmethod
    def unsqueeze(self, x, dim):
        '''
        Parameters
        ----------
        x---> input
        dim---> the dim of unsqueeze operation
        Returns
        -------
        '''

    @abstractmethod
    def reduce_sum(self, x, *dim):
        '''
        Reduce the dimensions of the data
        Args:
            x (list):
            dim (tuple(int)): the dimension to reduce.
        Returns:
            sum(x, dim)
        '''

    @abstractmethod
    def index_select(self, x, indices, dim=1):
        '''
        Parameters
        ----------
        x
        indices
        Returns
        -------
        '''

    @abstractmethod
    def scatter(self, x, indices):
        '''
        Parameters
        ----------
        x
        indices
        Returns
        -------
        '''

    @abstractmethod
    def conv1d(self, x, kernel):
        '''
        Parameters
        ----------
        x
        kernel
        Returns
        -------
        '''

    @abstractmethod
    def conv_trans1d(self, x, kernel):
        '''
        Parameters
        ----------
        x
        kernel
        Returns
        -------
        '''

    @abstractmethod
    def im2col_indices(self, x, kh, kw, padding, stride):
        '''
        Parameters
        ----------
        x: 4D array  N, FH, FW, C_{in}
        kh: kernel_height
        kw: kernel_width
        stride:
        padding:
        Returns
        ----------
        '''

    @abstractmethod
    def conv2d_flatten(self, x):
        '''
        Parameters
        ----------
        x: 4D array (batch_size, out_channels, height, width)
        Returns
        3D array (batch_size, out_channels, height * width)
        ----------
        '''

    @abstractmethod
    def feature_map_flatten(self, x):
        '''
        For RSTDP and STDP learning rules which is  follwed with conv pre_layer
        Parameters
        ----------
        x: 4D array (batch_size, out_channels, height, width)
        Returns
        2D array (batch_size, out_channels * height * width)
        ----------
        '''

    @abstractmethod
    def add(self, x, y):
        '''
        Add the tensor y to the input x and returns a new result.
        Args:
            x (Tensor): input
            y (Tensor or Number): the second input
        Returns:
            x + y
        '''
        NotImplementedError()

    @abstractmethod
    def minus(self, x, y):
        '''
        The first input minus the second input
        Args:
            x (Tensor): input
            y (Tensor or Number): the second input
        Returns:
            x - y
        '''
        NotImplementedError()

    @abstractmethod
    def div(self, x, y):
        '''
        The first input div the second input
        Args:
            x (Tensor): input
            y (Tensor or Number): the second input

        Returns:
            x/y

        '''
        NotImplementedError()

    @abstractmethod
    def relu(self, x):
        '''
        Rectified Linear
        Args:
            x:

        Returns:
            x = x if x>0. else x = 0
        '''

    @abstractmethod
    def mat_mult_weight(self, A, X):
        '''
        Matrix product.
        Args:
            A (Tensor): the first input to be multiplied
            X (Tensor): the second input to be multiplied
        Returns:
            mat_mult_weight(A,X)
        '''
        NotImplementedError()

    @abstractmethod
    def mat_mult_pre(self, A, X):
        '''
        Matrix product.
        Args:
            A (Tensor): the first input to be multiplied
            X (Tensor): the second input to be multiplied
        Returns:
            mat_mult_pre(A,X)
        '''
        NotImplementedError()

    @abstractmethod
    def sigmoid(self, x):
        '''

        Args:
            x:

        Returns:

        '''

    @abstractmethod
    def mat_mult(self, A, X):
        '''
        Matrix product.
        Args:
            A (Tensor): the first input to be multiplied
            X (Tensor): the second input to be multiplied
        Returns:
            mat_mult(A,X)
        '''
        NotImplementedError()

    @abstractmethod
    def reshape_mat_mult(self, A, X):
        '''
        Matrix product.
        Args:
            A (Tensor): the first input to be multiplied
            X (Tensor): the second input to be multiplied
        Returns:
        '''
        NotImplementedError()

    @abstractmethod
    def bmm(self, A, X):
        '''
        Performs a batch matrix-matrix product.
        Args:
            A (Tensor): the first input to be multiplied  [batch_size, n, m]
            X (Tensor): the second input to be multiplied  [batch_size, m, p]
        Returns:
            bmm(A,X)   [batch_size, n, p]
        '''
        NotImplementedError()

    @abstractmethod
    def sparse_mat_mult_weight(self, A, X):
        '''
        Sparse matrix product.
        Args:
            A (Tensor): the first input to be multiplied
            X (Tensor): the second input to be multiplied
        Returns:
            sparse_mat_mult_weight(A,X)
        '''
        NotImplementedError()

    @abstractmethod
    def var_mult(self, A, X):
        '''
        Args:
            A, X
        Returns:
            A * X
        '''
        NotImplementedError()

    @abstractmethod
    def mult_sum_weight(self, A, X):
        '''
         sum(A*X, dim=-2)
        Args:
            A:
            X:

        Returns:

        '''
        NotImplementedError()

    @abstractmethod
    def mat_linear(self, A, X, b):
        '''
        Args:
            A
            X
            b
        Returns:
            mat_mul(A,X)+b
        '''
        NotImplementedError()

    @abstractmethod
    def ger(self, A, X):
        '''
        Args:
            A
            X
        Returns:
            ger(A,X)
        '''
        NotImplementedError()

    @abstractmethod
    def var_linear(self, A, X, b):
        '''
        If A is matrix, then A and X should have the same shape, A*X is elemen-wise multiplication
        else  A should be a scalar value.
        Returns:
            A*X +b
        '''
        NotImplementedError()

    @abstractmethod
    def to_numpy(self, data):
        '''
        Args：
            data
        Returns:
            data.numpy()
        '''
        NotImplementedError()

    @abstractmethod
    def to_tensor(self, data):
        '''
        Args:
            data
        Returns:
            torch.tensor(data)
        '''
        NotImplementedError()

    @abstractmethod
    def clamp_(self, data, min, max):
        '''
            in-place clamp the data
        '''
        NotImplementedError()

    @abstractmethod
    def clamp_max_(self, data, max):
        '''
            in-place clamp the max of the data
        '''
        NotImplementedError()

    @abstractmethod
    def clamp_min_(self, data, min):
        '''
            in-place clamp the min of the data
        '''
        NotImplementedError()

    @abstractmethod
    def uniform(self, data, a=0.0, b=1.0):
        '''
        Args:
            data(tensor): an n-dimensional torch.Tensor
            a(float): the lower bound of the uniform distribution
            b(float): the upper bound of the uniform distribution
        Returns:
            torch.nn.init.uniform_(data, a=0.0, b=1.0)
        '''
        NotImplementedError()

    @abstractmethod
    def normal(self, data, mean=0.0, std=1.0):
        '''
        Args:
            data(tensor): an n-dimensional torch.Tensor
            mean(float): the mean of the normal distribution
            std(float): the standard deviation of the normal distribution
        Returns:
            torch.nn.init.normal_(data, mean=0.0, std=1.0)
        '''
        NotImplementedError()

    @abstractmethod
    def xavier_normal(self, data, gain=1.0):
        '''
        Args:
            data(tensor): an n-dimensional torch.Tensor
            gain: an optional scaling factor
        Returns:
            torch.nn.init.xavier_normal_(data, gain=1.0)
        '''
        NotImplementedError()

    @abstractmethod
    def xavier_uniform(self, data, gain=1.0):
        '''
        Args:
            data(tensor): an n-dimensional torch.Tensor
            gain: an optional scaling factor
        Returns:
            torch.nn.init.xavier_uniform_(data, gain=1.0)
        '''
        NotImplementedError()

    @abstractmethod
    def kaiming_normal(self, data, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        '''
        Args:
            data(tensor): an n-dimensional torch.Tensor
            a: the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
            mode: either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
            nonlinearity: the non-linear function (nn.functional name), recommended to use only with 'relu' or 'leaky_relu' (default).
        Returns:
            torch.nn.init.kaiming_normal_(data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        '''
        NotImplementedError()

    @abstractmethod
    def kaiming_uniform(self, data, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        '''
        Args:
            data(tensor): an n-dimensional torch.Tensor
            a: the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
            mode: either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
            nonlinearity: the non-linear function (nn.functional name), recommended to use only with 'relu' or 'leaky_relu' (default).
        Returns:
            torch.nn.init.kaiming_uniform_(data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        '''
        NotImplementedError()

    @abstractmethod
    def zero_init(self, data, constant_value=0.0):
        '''
        Args:
            data(tensor): an n-dimensional torch.Tensor
            constant_value(float): the value to fill the tensor with
        Returns:
            torch.nn.init.constant_(data, constant_value)
        '''
        NotImplementedError()

    # @abstractmethod
    # def euler_update(self):
    #     pass
    #
    # @abstractmethod
    # def rk2_update(self):
    #     pass
    #
    # @abstractmethod
    # def reset(self, v, v_reset, u_reset, spike):
    #      '''
    #      voltage reset
    #
    #      Parameters
    #      ----------
    #      v
    #      v_reset
    #      u_reset
    #      spike
    #
    #      Returns
    #      -------
    #      v[spike] = v_reset
    #      v[spike] += u_reset
    #      '''
    #
    # @abstractmethod
    # def reset_u(self, u, u_reset, spike):
    #      '''
    #      recovery reset
    #
    #      Parameters
    #      ----------
    #      u
    #      _reset
    #      spike
    #
    #      Returns
    #      -------
    #      u[spike] = u+u_reset
    #      '''
    #      NotImplementedError()
    #
    # @abstractmethod
    # def next_stage(self, x):
    #     '''
    #
    #    Parameters
    #    ----------
    #    x: list
    #
    #    Returns
    #    -------
    #    x[index]
    #    '''
    #
    # @abstractmethod
    # def izh_v(self, v, u, psp):
    #     '''
    #
    #     Parameters
    #     ----------
    #     v: list
    #     u: list
    #     psp: list
    #
    #     Returns
    #     -------
    #     V=V+dt*(0.04*V^2+5*V+140-U+PSP)
    #     '''
    #     NotImplementedError()
    #
    # @abstractmethod
    # def izh_u(self, a, b, v, u):
    #     '''
    #
    #     Parameters
    #     ----------
    #     a: list
    #     b: list
    #     u: list
    #     v: list
    #
    #     Returns
    #     -------
    #     U=U+a*(b*V-U)
    #     '''
    #     NotImplementedError()

    def exp(self, x):
        '''
        Args:
            x(tensor): an n-dimensional torch.Tensor
        Returns:
           return exp(x)
        '''
        NotImplementedError()

    def sin(self, x):
        '''
        Args:
            x(tensor): an n-dimensional torch.Tensor
        Returns:
           return exp(x)
        '''
        NotImplementedError()

    def cos(self, x):
        '''
        Args:
            x(tensor): an n-dimensional torch.Tensor
        Returns:
           return exp(x)
        '''
        NotImplementedError()

    def tan(self, x):
        '''
        Args:
            x(tensor): an n-dimensional torch.Tensor
        Returns:
           return exp(x)
        '''
        NotImplementedError()

    def log(self, x):
        '''
        Args:
            x(tensor): an n-dimensional torch.Tensor
        Returns:
           return exp(x)
        '''
        NotImplementedError()

    def log2(self, x):
        '''
        Args:
            x(tensor): an n-dimensional torch.Tensor
        Returns:
           return exp(x)
        '''
        NotImplementedError()

    def log10(self, x):
        '''
        Args:
            x(tensor): an n-dimensional torch.Tensor
        Returns:
           return exp(x)
        '''
        NotImplementedError()




# class Darwin_Backend(Backend):
#
#     def __init__(self):
#         super(Darwin_Backend, self).__init__()
#         pass
