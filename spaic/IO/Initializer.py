# encoding: utf-8
"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: Initializer.py
@time:2022/8/11 13:59
@description:
"""
__all__ = ['uniform', 'normal', 'xavier_normal', 'xavier_uniform', 'kaiming_normal', 'kaiming_uniform', 'constant', 'sparse']


class BaseInitializer(object):
    r"""
    """
    def __init__(self, **kwargs):
        super().__init__()


class uniform(BaseInitializer):
    def __init__(self, a=-0.0, b=1.0):
        super(uniform, self).__init__(a=a, b=b)
        '''
        Args:
            a(float): the lower bound of the uniform distribution
            b(float): the upper bound of the uniform distribution
        '''
        self.a = a
        self.b = b


class normal(BaseInitializer):
    def __init__(self, mean=0.0, std=1.0):
        super(normal, self).__init__(mean=mean, std=std)
        '''
        Args:
            mean(float): the mean of the normal distribution
            std(float): the standard deviation of the normal distribution
        Returns:
        '''
        self.mean = mean
        self.std = std


class xavier_normal(BaseInitializer):
    def __init__(self, gain=1.0):
        super(xavier_normal, self).__init__(gain=gain)
        '''
        Args:
            gain: an optional scaling factor
        '''
        self.gain = gain
        # return torch.nn.init.xavier_normal_(data, gain)


class xavier_uniform(BaseInitializer):
    def __init__(self, gain=1.0):
        super(xavier_uniform, self).__init__(gain=gain)
        '''
        Args:
            gain: an optional scaling factor
        '''
        self.gain = gain
        # return torch.nn.init.xavier_uniform_(data, gain)


class kaiming_normal(BaseInitializer):
    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(kaiming_normal, self).__init__(a=a, mode=mode, nonlinearity=nonlinearity)
        '''
        Args:
            a: the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
            mode: either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of 
            the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
            nonlinearity: the non-linear function (nn.functional name), recommended to use only with 'relu' or 
            'leaky_relu' (default).
       
        '''
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        # return torch.nn.init.kaiming_normal_(data, a, mode, nonlinearity)


class kaiming_uniform(BaseInitializer):
    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(kaiming_uniform, self).__init__(a=a, mode=mode, nonlinearity=nonlinearity)
        '''
        Args:
            a: the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
            mode: either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance 
            of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
            nonlinearity: the non-linear function (nn.functional name), recommended to use only with 'relu' or 
            'leaky_relu' (default).
        '''
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        # return torch.nn.init.kaiming_uniform_(data, a, mode, nonlinearity)


class constant(BaseInitializer):
    def __init__(self, constant_value=0.0):
        super(constant, self).__init__(constant_value=constant_value)
        '''
        Args:
            constant_value(float): the value to fill the tensor with
        '''
        self.constant_value = constant_value
        # return torch.nn.init.constant_(data, constant_value)


class sparse(BaseInitializer):
    def __init__(self, sparsity=0.1, std=0.01):
        super(sparse, self).__init__(sparsity=sparsity, std=std)
        '''
        Args:
            sparsity(float): The fraction of elements in each column to be set to zero
            std(float): the standard deviation of the normal distribution used to generate
            the non-zero values
        Returns:
            torch.nn.init.sparse_(data, sparsity, std)
        '''
        self.sparsity = sparsity
        self.std = std
        # return torch.nn.init.sparse_(data, sparsity, std)
