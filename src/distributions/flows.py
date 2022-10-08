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
            elif transform.__class__.__name__ == "TanhTransform":
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
    
    @property
    def covariance_matrix(self):
        if self.base_dist.__class__.__name__ != "MultivariateNormal":
            raise NotImplementedError
        else:
            covariance_matrix = self.base_dist.covariance_matrix
            for transform in self.transforms:
                if transform.__class__.__name__ == "BatchNormTransform":
                    w = transform.moving_variance / transform.constrained_gamma**2
                    w_mask = w.unsqueeze(-1) * w.unsqueeze(-2)
                    covariance_matrix *= w_mask
                else:
                    raise NotImplementedError
        return covariance_matrix

    def entropy(self):
        entropy = self.base_dist.entropy()
        for transform in self.transforms:
            if transform.__class__.__name__ == "BatchNormTransform":
                scale = torch.sqrt(transform.moving_variance) / transform.constrained_gamma
                entropy += torch.log(scale).sum()
            elif transform.__class__.__name__ == "TanhTransform": # skip tanh transform
                pass
            else:
                raise NotImplementedError
        return entropy

""" NOTE: too unstable for certain action space """
class TanhTransform(TransformModule):
    """ Adapted from Pytorch implementation """
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 0
    def __init__(self, limits):
        super().__init__() 
        self.limits = nn.Parameter(limits, requires_grad=False)
        self.eps = 1e-5

    def __call__(self, x):
        return self.limits * torch.tanh(x)
    
    def _inverse(self, y):
        y = torch.clip(y / self.limits, -1. + self.eps, 1. - self.eps) # prevent overflow
        return torch.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        ldj = (2. * (math.log(2.) - x - F.softplus(-2. * x)))
        ldj += torch.abs(self.limits).log()
        return ldj


class BatchNormTransform(TransformModule):
    """ Adapted from pyro's BatchNormTransform. However, we do not use sample mean in _inverse """
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 0
    def __init__(self, input_dim, momentum=0.1, epsilon=1e-5, affine=False):
        super().__init__()
        self.input_dim = input_dim
        self.momentum = momentum
        self.epsilon = epsilon
         
        self.moving_mean = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        self.moving_variance = nn.Parameter(torch.ones(input_dim), requires_grad=False)
        
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
        
        return (y - self.moving_mean) * self.constrained_gamma / torch.sqrt(
                self.moving_variance + self.epsilon
            ) + self.beta
    
    def log_abs_det_jacobian(self, x, y):
        var = self.moving_variance
        return -self.constrained_gamma.log() + 0.5 * torch.log(var + self.epsilon)