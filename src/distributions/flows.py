import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution
from pyro.distributions.torch_transform import TransformModule

class SimpleTransformedModule(TransformedDistribution):
    """ Subclass of torch TransformedDistribution with mean, variance, entropy 
        implementation for simple transformations """
    def __init__(self, base_distribution, transforms, validate_args=None):
        super().__init__(base_distribution, transforms, validate_args)
    
    @property
    def mean(self):
        mean = self.base_dist.mean
        for transform in self.transforms:
            if transform.__class__.__name__ == "BatchNormTransform":
                mean = transform._call(mean)
            else:
                raise NotImplementedError
        return mean
    
    @property
    def variance(self):
        variance = self.base_dist.variance
        for transform in self.transforms:
            if transform.__class__.__name__ == "BatchNormTransform":
                variance *= transform.moving_variance / transform.constrained_gamma**2
            else:
                raise NotImplementedError
        return variance
    
    def entropy(self):
        entropy = self.base_dist.entropy()
        for transform in self.transforms:
            if transform.__class__.__name__ == "BatchNormTransform":
                scale = torch.sqrt(transform.moving_variance) / transform.constrained_gamma
                entropy += torch.log(scale).sum()
            else:
                raise NotImplementedError
        return entropy


class TanhTransform(TransformModule):
    """ Adapted from Pytorch implementation """
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 0
    def __init__(self, limits):
        super().__init__()
        self.limits = limits

    def __call__(self, x):
        return self.limits * torch.tanh(x)
    
    def _inverse(self, y):
        y = y / self.limits
        return torch.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        ldj = (2. * (math.log(2.) - x - F.softplus(-2. * x)))#.sum(-1, keepdim=True)
        return ldj


""" TODO
test batch norm on multidimension input, verify log_prob accuracy
add torch transformation class with affine mean, variance, and entropy transforms
"""
class BatchNormTransform(TransformModule):
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 0
    def __init__(self, input_dim, momentum=0.1, epsilon=1e-5, affine=False):
        super().__init__()
        self.input_dim = input_dim
        self.momentum = momentum
        self.epsilon = epsilon
        
        self.register_buffer('moving_mean', torch.zeros(input_dim))
        self.register_buffer('moving_variance', torch.ones(input_dim))
        
        self.gamma = nn.Parameter(torch.ones(input_dim), requires_grad=affine)
        self.beta = nn.Parameter(torch.zeros(input_dim), requires_grad=affine)
            
    @property   
    def constrained_gamma(self):
        """ Enforce positivity """
        return torch.relu(self.gamma) + 1e-6
    
    def _call(self, x):
        return (x - self.beta) / self.constrained_gamma * torch.sqrt(
            self.moving_variance + self.epsilon
        ) + self.moving_mean
            
    def _inverse(self, y):
        op_dims = [i for i in range(len(y.shape) - 1)]
        
        if self.training:
            mean, var = y.mean(op_dims), y.var(op_dims)
            with torch.no_grad():
                self.moving_mean.mul_(1 - self.momentum).add_(mean * self.momentum)
                self.moving_variance.mul_(1 - self.momentum).add_(var * self.momentum)
        else:
            mean, var = self.moving_mean, self.moving_variance
        return (y - mean) * self.constrained_gamma / torch.sqrt(
            var + self.epsilon
        ) + self.beta
    
    def log_abs_det_jacobian(self, x, y):
        op_dims = [i for i in range(len(y.shape) - 1)]
        if self.training:
            var = torch.var(y, dim=op_dims, keepdim=True)
        else:
            var = self.moving_variance
        return -self.constrained_gamma.log() + 0.5 * torch.log(var + self.epsilon)