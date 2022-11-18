import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torch_dist

import pyro.nn as pyro_nn
import pyro.distributions.transforms as pyro_transform

from src.distributions.nn_models import Model
from src.distributions.flows import SimpleTransformedModule, BatchNormTransform
from src.distributions.utils import make_covariance_matrix

class ConditionalGaussian(Model):
    """ Conditional gaussian distribution used to create mixture distributions """
    def __init__(self, x_dim, z_dim, cov="full", batch_norm=True):
        """
        Args:
            x_dim (int): observed output dimension
            z_dim (int): latent conditonal dimension
            cov (str): covariance type. choices=["diag", "full", "tied", "tied_full"]
            batch_norm (bool, optional): whether to use input batch normalization. Default=True
        """
        super().__init__()
        assert cov in ["diag", "full", "tied", "tied_full"]
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.cov = cov
        self.batch_norm = batch_norm
        self.eps = 1e-6
        
        self.mu = nn.Parameter(torch.randn(1, z_dim, x_dim))
        nn.init.uniform_(self.mu, a=-1, b=1)
    
        if cov == "full":
            self.lv = nn.Parameter(torch.randn(1, z_dim, x_dim))
            self.tl = nn.Parameter(torch.zeros(1, z_dim, x_dim, x_dim))
            nn.init.normal_(self.lv, mean=0, std=0.01)
        elif cov == "diag":
            self.lv = nn.Parameter(torch.randn(1, z_dim, x_dim))
            self.tl = nn.Parameter(torch.zeros(1, z_dim, x_dim, x_dim), requires_grad=False)
            nn.init.normal_(self.lv, mean=0, std=0.01)
        elif cov == "tied":
            self.lv = nn.Parameter(torch.randn(1, 1, x_dim))
            self.tl = nn.Parameter(torch.zeros(1, 1, x_dim, x_dim), requires_grad=False)
            nn.init.normal_(self.lv, mean=0, std=0.01)
        elif cov == "tied_full":
            self.lv = nn.Parameter(torch.randn(1, 1, x_dim))
            self.tl = nn.Parameter(torch.zeros(1, 1, x_dim, x_dim))
            nn.init.normal_(self.lv, mean=0, std=0.01)

        if batch_norm:
            self.bn = BatchNormTransform(x_dim, momentum=0.1, affine=False, update_stats=False)
        
    def __repr__(self):
        s = "{}(x_dim={}, z_dim={}, cov={}, batch_norm={})".format(
            self.__class__.__name__, self.x_dim, self.z_dim, self.cov, self.batch_norm
        )
        return s
    
    def init_params(self, means, covariances, requires_grad=False):
        """ initialize mean and covariance, set bn momentum to 0 """
        assert self.cov != "full"
        assert means.shape[-2] == self.z_dim
        means = torch.from_numpy(means).unsqueeze(0).to(torch.float32).to(self.device)
        covs = torch.from_numpy(covariances).unsqueeze(0).to(torch.float32).to(self.device)
        variances = torch.diagonal(covs, dim1=-2, dim2=-1) 
        
        self.mu.data = means
        self.lv.data = 0.5 * torch.log(variances)
        
        self.mu.requires_grad = requires_grad
        self.lv.requires_grad = requires_grad

        # disable batch norm
        self.batch_norm = False
        self.bn = None

    def init_batch_norm(self, mean, variance):
        if self.batch_norm:
            self.bn.moving_mean.data = mean
            self.bn.moving_variance.data = variance

    def get_distribution_class(self, requires_grad=True, **kwargs):
        [mu, lv, tl] = self.mu, self.lv, self.tl
        L = make_covariance_matrix(lv, tl, cholesky=True, lv_rectify="exp")
        
        # handle tied covariance case
        if L.shape[1] != mu.shape[1]:
            L = torch.repeat_interleave(L, mu.shape[1], dim=1)

        if requires_grad is False:
            mu, L = mu.data, L.data

        distribution = torch_dist.MultivariateNormal(mu, scale_tril=L)
        
        transforms = []
        if self.batch_norm:
            transforms.append(self.bn)
        
        distribution = SimpleTransformedModule(distribution, transforms)
        return distribution
    
    def mean(self, **kwargs):
        distribution = self.get_distribution_class()
        return distribution.mean
    
    def variance(self, **kwargs):
        distribution = self.get_distribution_class()
        return distribution.variance
    
    def entropy(self, **kwargs):
        distribution = self.get_distribution_class()
        return distribution.entropy()
    
    def log_prob(self, x, **kwargs):
        """ Component log probabilities 

        Args:
            x (torch.tensor): size=[batch_size, x_dim]
        """
        distribution = self.get_distribution_class()
        return distribution.log_prob(x.unsqueeze(-2))
    
    def mixture_log_prob(self, pi, x, **kwargs):
        """ Compute mixture log probabilities 
        
        Args:
            pi (torch.tensor): mixing weights. size=[..., z_dim]
            x (torch.tensor): observervations. size[..., x_dim]
        """
        logp_pi = torch.log(pi + self.eps)
        logp_x = self.log_prob(x)
        logp = torch.logsumexp(logp_pi + logp_x, dim=-1)
        return logp

    def sample(self, sample_shape=torch.Size(), **kwargs):
        """ Sample components """
        distribution = self.get_distribution_class()
        return distribution.rsample(sample_shape)

    def bayesian_average(self, pi, **kwargs):
        """ Compute weighted average of component means 
        
        Args:
            pi (torch.tensor): mixing weights. size=[..., z_dim]
        """
        mu = self.mean()
        x = torch.sum(pi.unsqueeze(-1) * mu.unsqueeze(0), dim=-2)
        return x
    
    def ancestral_sample(self, pi, num_samples=1, sample_mean=False, tau=0.1, hard=True, **kwargs):
        """ Ancestral sampling
        
        Args:
            pi (torch.tensor): mixing weights. size=[T, batch_size, z_dim]
            num_samples (int, optional): number of samples to draw. Default=1
            sample_mean (bool, optional): whether to sample component mean. Default=False
            tau (float, optional): gumbel softmax temperature. Default=0.1
            hard (float, optional): if hard use straight-through gradient. Default=True

        Returns:
            x (torch.tensor): sampled observations. size[num_samples, T, batch_size, x_dim]
        """
        log_pi_ = torch.repeat_interleave(torch.log(pi + self.eps).unsqueeze(0), num_samples, 0)
        z_ = F.gumbel_softmax(log_pi_, tau=tau, hard=hard).unsqueeze(-1)
        
        # sample component
        if sample_mean:
            x_ = self.mean()
        else:
            x_ = self.sample((num_samples, pi.shape[0])).squeeze(1)
        x = torch.sum(z_ * x_, dim=-2)
        return x


class ConditionalFlow(Model):
    """ Conditional autoregressive flow distribution with used to create mixture distributions """
    def __init__(self, x_dim, z_dim, hidden_dim=None, cov="diag", batch_norm=True):
        """
        Args:
            x_dim (int): observed output dimension
            z_dim (int): latent conditonal dimension
            hidden_dim (int): hidden layer dimension in autoregressive network. 
                If None hidden_dim=10*x_dim Default=None
            cov (str): covariance type. choices=["diag", "full", "tied", "tied_full"]
            batch_norm (bool, optional): whether to use input batch normalization. Default=True
        """
        super().__init__()
        assert cov in ["diag", "full", "tied", "tied_full"]
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else 10 * x_dim
        self.cov = cov
        self.batch_norm = batch_norm
        self.eps = 1e-6
        
        self.mu = nn.Parameter(torch.randn(1, z_dim, x_dim))
        nn.init.uniform_(self.mu, a=-1, b=1)
    
        if cov == "full":
            self.lv = nn.Parameter(torch.randn(1, z_dim, x_dim))
            self.tl = nn.Parameter(torch.zeros(1, z_dim, x_dim, x_dim))
            nn.init.normal_(self.lv, mean=0, std=0.01)
        elif cov == "diag":
            self.lv = nn.Parameter(torch.randn(1, z_dim, x_dim))
            self.tl = nn.Parameter(torch.zeros(1, z_dim, x_dim, x_dim), requires_grad=False)
            nn.init.normal_(self.lv, mean=0, std=0.01)
        elif cov == "tied":
            self.lv = nn.Parameter(torch.randn(1, 1, x_dim))
            self.tl = nn.Parameter(torch.zeros(1, 1, x_dim, x_dim), requires_grad=False)
            nn.init.normal_(self.lv, mean=0, std=0.01)
        elif cov == "tied_full":
            self.lv = nn.Parameter(torch.randn(1, 1, x_dim))
            self.tl = nn.Parameter(torch.zeros(1, 1, x_dim, x_dim))
            nn.init.normal_(self.lv, mean=0, std=0.01)
        
        self.flow = pyro_transform.AffineAutoregressive(
            pyro_nn.AutoRegressiveNN(x_dim, [self.hidden_dim, self.hidden_dim])
        )
        
        if batch_norm:
            self.bn = BatchNormTransform(x_dim, momentum=0.1, affine=False, update_stats=False)
        
    def __repr__(self):
        s = "{}(x_dim={}, z_dim={}, hidden_dim={}, cov={}, batch_norm={})".format(
            self.__class__.__name__, self.x_dim, self.z_dim, self.hidden_dim, self.cov, self.batch_norm
        )
        return s
    
    def init_params(self, means, covariances, requires_grad=False):
        """ initialize mean and covariance, set bn momentum to 0 """
        assert self.cov != "full"
        assert means.shape[-2] == self.z_dim
        means = torch.from_numpy(means).unsqueeze(0).to(torch.float32).to(self.device)
        covs = torch.from_numpy(covariances).unsqueeze(0).to(torch.float32).to(self.device)
        variances = torch.diagonal(covs, dim1=-2, dim2=-1) 
        
        self.mu.data = means
        self.lv.data = 0.5 * torch.log(variances)
        
        self.mu.requires_grad = requires_grad
        self.lv.requires_grad = requires_grad

        # disable batch norm
        self.batch_norm = False
        self.bn = None

    def init_batch_norm(self, mean, variance):
        if self.batch_norm:
            self.bn.moving_mean.data = mean
            self.bn.moving_variance.data = variance

    def get_distribution_class(self, requires_grad=True, **kwargs):
        [mu, lv, tl] = self.mu, self.lv, self.tl
        L = make_covariance_matrix(lv, tl, cholesky=True, lv_rectify="exp")
        
        # handle tied covariance case
        if L.shape[1] != mu.shape[1]:
            L = torch.repeat_interleave(L, mu.shape[1], dim=1)

        if requires_grad is False:
            mu, L = mu.data, L.data

        distribution = torch_dist.MultivariateNormal(mu, scale_tril=L)
        
        transforms = [self.flow]
        if self.batch_norm:
            transforms.append(self.bn)
        
        distribution = SimpleTransformedModule(distribution, transforms)
        return distribution
    
    def mean(self, **kwargs):
        distribution = self.get_distribution_class()
        return distribution.mean
    
    def variance(self, **kwargs):
        distribution = self.get_distribution_class()
        return distribution.variance
    
    def entropy(self, **kwargs):
        distribution = self.get_distribution_class()
        return distribution.entropy()
    
    def log_prob(self, x, **kwargs):
        """ Component log probabilities 

        Args:
            x (torch.tensor): size=[batch_size, x_dim]
        """
        distribution = self.get_distribution_class()
        return distribution.log_prob(x.unsqueeze(-2))
    
    def mixture_log_prob(self, pi, x, **kwargs):
        """ Compute mixture log probabilities 
        
        Args:
            pi (torch.tensor): mixing weights. size=[..., z_dim]
            x (torch.tensor): observervations. size[..., x_dim]
        """
        logp_pi = torch.log(pi + self.eps)
        logp_x = self.log_prob(x)
        logp = torch.logsumexp(logp_pi + logp_x, dim=-1)
        return logp

    def sample(self, sample_shape=torch.Size(), **kwargs):
        """ Sample components """
        distribution = self.get_distribution_class()
        return distribution.rsample(sample_shape)

    def bayesian_average(self, pi, **kwargs):
        """ Compute weighted average of component means 
        
        Args:
            pi (torch.tensor): mixing weights. size=[..., z_dim]
        """
        mu = self.mean()
        x = torch.sum(pi.unsqueeze(-1) * mu.unsqueeze(0), dim=-2)
        return x
    
    def ancestral_sample(self, pi, num_samples=1, sample_mean=False, tau=0.1, hard=True, **kwargs):
        """ Ancestral sampling
        
        Args:
            pi (torch.tensor): mixing weights. size=[T, batch_size, z_dim]
            num_samples (int, optional): number of samples to draw. Default=1
            sample_mean (bool, optional): whether to sample component mean. Default=False
            tau (float, optional): gumbel softmax temperature. Default=0.1
            hard (float, optional): if hard use straight-through gradient. Default=True

        Returns:
            x (torch.tensor): sampled observations. size[num_samples, T, batch_size, x_dim]
        """
        log_pi_ = torch.repeat_interleave(torch.log(pi + self.eps).unsqueeze(0), num_samples, 0)
        z_ = F.gumbel_softmax(log_pi_, tau=tau, hard=hard).unsqueeze(-1)
        
        # sample component
        if sample_mean:
            x_ = self.mean()
        else:
            x_ = self.sample((num_samples, pi.shape[0])).squeeze(1)
        x = torch.sum(z_ * x_, dim=-2)
        return x