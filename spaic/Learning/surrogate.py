# encoding: utf-8
"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: surrogate.py
@time:2022/8/12 17:14
@description:
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def heaviside(input):
    return (input >= 0.).to(input.dtype)


class SurrogateFunctionBase(nn.Module):
    """
    Surrogate Function 的基类
    :param alpha: 为一些能够调控函数形状的代理函数提供参数.
    :param requires_grad: 参数 ``alpha`` 是否需要计算梯度, 默认为 ``False``
    """
    def __init__(self, alpha, requires_grad=True):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.tensor(alpha, dtype=torch.float),
            requires_grad=requires_grad)

    @staticmethod
    def firing_func(input, alpha):
        """
        :param input: 膜电位的输入
        :param alpha: 控制代理梯度形状的变量, 可以为 ``NoneType``
        :return: 激发之后的spike, 取值为 ``[0, 1]``
        """
        raise NotImplementedError

    def forward(self, input):
        """
        :param input: 膜电位输入
        :return: 激发之后的spike
        """
        return self.firing_func(input, self.alpha)


'''
    sigmoid surrogate function.
'''

class sigmoid(torch.autograd.Function):
    """
    使用 sigmoid 作为代理梯度函数
    对应的原函数为:
    .. math::
            g(input) = \\mathrm{sigmoid}(\\alpha input) = \\frac{1}{1+e^{-\\alpha input}}
    反向传播的函数为:
    .. math::
            g'(input) = \\alpha * (1 - \\mathrm{sigmoid} (\\alpha input)) \\mathrm{sigmoid} (\\alpha input)
    """
    @staticmethod
    def forward(ctx, input, alpha):
        if input.requires_grad:
            ctx.save_for_backward(input)
            ctx.alpha = alpha
        return heaviside(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            s_input = torch.sigmoid(ctx.alpha * ctx.saved_tensors[0])
            grad_input = grad_output * s_input * (1 - s_input) * ctx.alpha
        return grad_input, None


class SigmoidGrad(SurrogateFunctionBase):
    def __init__(self, alpha=1., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def firing_func(input, alpha):
        return sigmoid.apply(input, alpha)


'''
    atan surrogate function.
'''


class atan(torch.autograd.Function):
    """
    使用 Atan 作为代理梯度函数
    对应的原函数为:
    .. math::
            g(input) = \\frac{1}{\\pi} \\arctan(\\frac{\\pi}{2}\\alpha input) + \\frac{1}{2}
    反向传播的函数为:
    .. math::
            g'(input) = \\frac{\\alpha}{2(1 + (\\frac{\\pi}{2}\\alpha input)^2)}
    """
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input, alpha)
        return input.gt(0.).type_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        grad_alpha = None

        shared_c = grad_output / \
                   (1 + (ctx.saved_tensors[1] * math.pi /
                         2 * ctx.saved_tensors[0]).square())
        if ctx.needs_input_grad[0]:
            grad_input = ctx.saved_tensors[1] / 2 * shared_c
        if ctx.needs_input_grad[1]:
            grad_alpha = (ctx.saved_tensors[0] / 2 * shared_c).sum()

        return grad_input, grad_alpha


class AtanGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=True):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def firing_func(input, alpha):
        return atan.apply(input, alpha)


'''
    rectangle surrogate fucntion. 
'''


class rectangle(torch.autograd.Function):
    '''
    Here we use the Rectangle surrogate gradient as was done
    in Yu et al. (2018).
    '''

    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input, alpha)
        return input.gt(0).type_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sur_grad = (torch.abs(input) < alpha).float()
        return grad_input * sur_grad, None

class RectangleGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=True):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def firing_func(input, alpha):
        return rectangle.apply(input, alpha)


'''
    gate surrogate fucntion. 
'''


class gate(torch.autograd.Function):
    """
    使用 gate 作为代理梯度函数
    对应的原函数为:
    .. math::
            g(input) = \\mathrm{NonzeroSign}(input) \\log (|\\alpha input| + 1)
    反向传播的函数为:
    .. math::
            g'(input) = \\frac{\\alpha}{1 + |\\alpha input|} = \\frac{1}{\\frac{1}{\\alpha} + |input|}
    """
    @staticmethod
    def forward(ctx, input, alpha):
        if input.requires_grad:
            grad_input = torch.where(input.abs() < 1. / alpha, torch.ones_like(input), torch.zeros_like(input))
            ctx.save_for_backward(grad_input)
        return input.gt(0).type_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * ctx.saved_tensors[0]
        return grad_input, None


class GateGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def firing_func(input, alpha):
        return gate.apply(input, alpha)


'''
    quadratic_gate surrogate function.
'''


class quadratic_gate(torch.autograd.Function):
    """
    使用 quadratic_gate 作为代理梯度函数
    对应的原函数为:
    .. math::
        g(input) =
        \\begin{cases}
        0, & input < -\\frac{1}{\\alpha} \\\\
        -\\frac{1}{2}\\alpha^2|input|input + \\alpha x + \\frac{1}{2}, & |input| \\leq \\frac{1}{\\alpha}  \\\\
        1, & input > \\frac{1}{\\alpha} \\\\
        \\end{cases}
    反向传播的函数为:
    .. math::
        g'(input) =
        \\begin{cases}
        0, & |input| > \\frac{1}{\\alpha} \\\\
        -\\alpha^2|input|+\\alpha, & |input| \\leq \\frac{1}{\\alpha}
        \\end{cases}
    """
    @staticmethod
    def forward(ctx, input, alpha):
        if input.requires_grad:
            mask_zero = (input.abs() > 1 / alpha)
            grad_input = -alpha * alpha * input.abs() + alpha
            grad_input.masked_fill_(mask_zero, 0)
            ctx.save_for_backward(grad_input)
        return input.gt(0.).type_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * ctx.saved_tensors[0]
        return grad_input, None


class QGateGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def firing_func(input, alpha):
        return quadratic_gate.apply(input, alpha)


class relu_like(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        if input.requires_grad:
            ctx.save_for_backward(input, alpha)
        return heaviside(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_alpha = None, None
        input, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * input.gt(0.).type_as(input) * alpha
        if ctx.needs_input_grad[1]:
            grad_alpha = (grad_output * F.relu(input)).sum()
        return grad_input, grad_alpha


class ReLUGrad(SurrogateFunctionBase):
    """
    使用ReLU作为代替梯度函数, 主要用为相同结构的ANN的测试
    """
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def firing_func(input, alpha):
        return relu_like.apply(input, alpha)


'''
    Straight-Through (ST) Estimator
'''


class straight_through_estimator(torch.autograd.Function):
    """
    使用直通估计器作为代理梯度函数
    http://arxiv.org/abs/1308.3432
    """
    @staticmethod
    def forward(ctx, input):
        output = heaviside(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        return grad_input