import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torch_dist

import pyro.nn as pyro_nn
import pyro.distributions.transforms as pyro_transform

from src.distributions.nn_models import Model
from src.distributions.flows import SimpleTransformedModule, BatchNormTransform
from src.distributions.utils import make_covariance_matrix

class HyperConditionalGaussian(Model):
    """ Hypernet version of conditional gaussian distribution """
    def __init__(
        self, x_dim, z_dim, hyper_dim, cov="full", hyper_cov=False, batch_norm=True
        ):
        """
        Args:
            x_dim (int): observed output dimension
            z_dim (int): latent conditonal dimension
            hyper_dim (int): hyper dimension
            cov (str): covariance type ["diag", "full", "tied"]
            hyper_cov (bool, optional): whether to use hyper variable for cov. Default=False
            batch_norm (bool, optional): whether to use input batch normalization. Default=True
        """
        super().__init__()
        assert cov in ["diag", "full", "tied"]
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hyper_dim = hyper_dim
        self.cov = cov
        self.hyper_cov = hyper_cov
        self.batch_norm = batch_norm
        self.eps = 1e-6
        
        self._mu = nn.Parameter(torch.randn(1, z_dim, x_dim))
        self._mu_offset = nn.Linear(hyper_dim, z_dim * x_dim, bias=False)
        nn.init.uniform_(self._mu, a=-1, b=1)
        self._mu_offset.weight.data *= 0.1
        
        if not hyper_cov:
            if cov == "full":
                self._lv = nn.Parameter(torch.randn(1, z_dim, x_dim))
                self._tl = nn.Parameter(torch.zeros(1, z_dim, x_dim, x_dim))
                nn.init.normal_(self._lv, mean=0, std=0.01)
            elif cov == "diag":
                self._lv = nn.Parameter(torch.randn(1, z_dim, x_dim))
                self._tl = nn.Parameter(torch.zeros(1, z_dim, x_dim, x_dim), requires_grad=False)
                nn.init.normal_(self._lv, mean=0, std=0.01)
            elif cov == "tied":
                self._lv = nn.Parameter(torch.randn(1, 1, x_dim))
                self._tl = nn.Parameter(torch.zeros(1, z_dim, x_dim, x_dim), requires_grad=False)
                nn.init.normal_(self._lv, mean=0, std=0.01)
        else:
            if cov == "full":
                self._lv = nn.Parameter(torch.randn(1, z_dim, x_dim))
                self._lv_offset = nn.Linear(hyper_dim, z_dim * x_dim, bias=False)
                self._tl = nn.Parameter(torch.zeros(1, z_dim, x_dim, x_dim))
                nn.init.normal_(self._lv, mean=0, std=0.01)
                self._lv_offset.weight.data *= 0.1
            elif cov == "diag":
                self._lv = nn.Parameter(torch.randn(1, z_dim, x_dim))
                self._lv_offset = nn.Linear(hyper_dim, z_dim * x_dim, bias=False)
                self._tl = nn.Parameter(torch.zeros(1, z_dim, x_dim, x_dim), requires_grad=False)
                nn.init.normal_(self._lv, mean=0, std=0.01)
                self._lv_offset.weight.data *= 0.1
            elif cov == "tied":
                self._lv = nn.Parameter(torch.randn(1, 1, x_dim))
                self._lv_offset = nn.Linear(hyper_dim, x_dim, bias=False)
                self._tl = nn.Parameter(torch.zeros(1, z_dim, x_dim, x_dim), requires_grad=False)
                nn.init.normal_(self._lv, mean=0, std=0.01)
                self._lv_offset.weight.data *= 0.1

        if batch_norm:
            self.bn = BatchNormTransform(x_dim, momentum=0.1, affine=False, update_stats=False)
        
    def __repr__(self):
        s = "{}(x_dim={}, z_dim={}, hyper_dim={}, cov={}, batch_norm={}, hyper_cov={})".format(
            self.__class__.__name__, self.x_dim, self.z_dim, self.hyper_dim, self.cov, 
            self.batch_norm, self.hyper_cov
        )
        return s
    
    def init_batch_norm(self, mean, variance):
        if self.batch_norm:
            self.bn.moving_mean.data = mean
            self.bn.moving_variance.data = variance
        
    def mu(self, z):
        mu = self._mu + self._mu_offset(z).view(-1, self.z_dim, self.x_dim)
        return mu
    
    def lv(self, z):
        if not self.hyper_cov:
            lv = torch.repeat_interleave(self._lv, len(z), dim=0)
        else:
            lv = self._lv + self._lv_offset(z).view(len(z), -1, self.x_dim)
        return lv

    def get_distribution_class(self, z, transform=True, requires_grad=True):
        [mu, lv, tl] = self.mu(z), self.lv(z), self._tl
        
        L = make_covariance_matrix(lv, tl, cholesky=True, lv_rectify="exp")
        
        if requires_grad is False:
            mu, L = mu.data, L.data
        
        distribution = torch_dist.MultivariateNormal(mu, scale_tril=L)
        if not transform:
            return distribution
        
        transforms = []
        if self.batch_norm:
            transforms.append(self.bn)
        
        distribution = SimpleTransformedModule(distribution, transforms)
        return distribution
    
    def mean(self, z):
        distribution = self.get_distribution_class(z)
        return distribution.mean
    
    def variance(self, z):
        distribution = self.get_distribution_class(z)
        return distribution.variance
    
    def entropy(self, z):
        distribution = self.get_distribution_class(z)
        return distribution.entropy()
    
    def log_prob(self, x, z):
        """ Component log probabilities 

        Args:
            x (torch.tensor): observation vector. size=[batch_size, x_dim]
            z (torch.tensor): hyper vector. size=[batch_size, hyper_dim]
        """
        distribution = self.get_distribution_class(z)
        return distribution.log_prob(x.unsqueeze(-2))
    
    def mixture_log_prob(self, pi, x, z):
        """ Compute mixture log probabilities 
        
        Args:
            pi (torch.tensor): mixing weights. size=[..., z_dim]
            x (torch.tensor): observervations. size[..., x_dim]
            z (torch.tensor): hyper vector. size=[batch_size, hyper_dim]
        """
        logp_pi = torch.log(pi + self.eps)
        logp_x = self.log_prob(x, z)
        logp = torch.logsumexp(logp_pi + logp_x, dim=-1)
        return logp

    def sample(self, sample_shape, z):
        """ Sample components """
        distribution = self.get_distribution_class(z)
        return distribution.rsample(sample_shape)

    def bayesian_average(self, pi, z):
        """ Compute weighted average of component means 
        
        Args:
            pi (torch.tensor): mixing weights. size=[..., z_dim]
        """
        mu = self.mean(z)
        x = torch.sum(pi.unsqueeze(-1) * mu.unsqueeze(0), dim=-2)
        return x
    
    def ancestral_sample(self, pi, z, num_samples=1, sample_mean=False, tau=0.1, hard=True):
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
            x_ = self.mean(z)
        else:
            x_ = self.sample((num_samples, pi.shape[0]), z).squeeze(1)
        x = torch.sum(z_ * x_, dim=-2)
        return x


class HyperConditionalFlow(Model):
    """ Hypernet version of conditional autoregressive flow distribution """
    def __init__(
        self, x_dim, z_dim, hyper_dim, hidden_dim=None, cov="full", hyper_cov=False, batch_norm=True
        ):
        """
        Args:
            x_dim (int): observed output dimension
            z_dim (int): latent conditonal dimension
            hyper_dim (int): hyper dimension
            hidden_dim (int, optional): hidden layer dimension in autoregressive network. 
                If None hidden_dim=10*x_dim Default=None
            cov (str): covariance type ["diag", "full", "tied"]
            hyper_cov (bool, optional): whether to use hyper variable for cov. Default=False
            batch_norm (bool, optional): whether to use input batch normalization. Default=True
        """
        super().__init__()
        assert cov in ["diag", "full", "tied"]
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hyper_dim = hyper_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else 10 * x_dim
        self.cov = cov
        self.hyper_cov = hyper_cov
        self.batch_norm = batch_norm
        self.eps = 1e-6
        
        self._mu = nn.Parameter(torch.randn(1, z_dim, x_dim))
        self._mu_offset = nn.Linear(hyper_dim, z_dim * x_dim, bias=False)
        nn.init.uniform_(self._mu, a=-1, b=1)
        self._mu_offset.weight.data *= 0.1
        
        if not hyper_cov:
            if cov == "full":
                self._lv = nn.Parameter(torch.randn(1, z_dim, x_dim))
                self._tl = nn.Parameter(torch.zeros(1, z_dim, x_dim, x_dim))
                nn.init.normal_(self._lv, mean=0, std=0.01)
            elif cov == "diag":
                self._lv = nn.Parameter(torch.randn(1, z_dim, x_dim))
                self._tl = nn.Parameter(torch.zeros(1, z_dim, x_dim, x_dim), requires_grad=False)
                nn.init.normal_(self._lv, mean=0, std=0.01)
            elif cov == "tied":
                self._lv = nn.Parameter(torch.randn(1, 1, x_dim))
                self._tl = nn.Parameter(torch.zeros(1, z_dim, x_dim, x_dim), requires_grad=False)
                nn.init.normal_(self._lv, mean=0, std=0.01)
        else:
            if cov == "full":
                self._lv = nn.Parameter(torch.randn(1, z_dim, x_dim))
                self._lv_offset = nn.Linear(hyper_dim, z_dim * x_dim, bias=False)
                self._tl = nn.Parameter(torch.zeros(1, z_dim, x_dim, x_dim))
                nn.init.normal_(self._lv, mean=0, std=0.01)
                self._lv_offset.weight.data *= 0.1
            elif cov == "diag":
                self._lv = nn.Parameter(torch.randn(1, z_dim, x_dim))
                self._lv_offset = nn.Linear(hyper_dim, z_dim * x_dim, bias=False)
                self._tl = nn.Parameter(torch.zeros(1, z_dim, x_dim, x_dim), requires_grad=False)
                nn.init.normal_(self._lv, mean=0, std=0.01)
                self._lv_offset.weight.data *= 0.1
            elif cov == "tied":
                self._lv = nn.Parameter(torch.randn(1, 1, x_dim))
                self._lv_offset = nn.Linear(hyper_dim, x_dim, bias=False)
                self._tl = nn.Parameter(torch.zeros(1, z_dim, x_dim, x_dim), requires_grad=False)
                nn.init.normal_(self._lv, mean=0, std=0.01)
                self._lv_offset.weight.data *= 0.1
        
        self.flow = pyro_transform.AffineAutoregressive(
            pyro_nn.AutoRegressiveNN(x_dim, [self.hidden_dim, self.hidden_dim])
        )
        
        if batch_norm:
            self.bn = BatchNormTransform(x_dim, momentum=0.1, affine=False, update_stats=False)
        
    def __repr__(self):
        s = "{}(x_dim={}, z_dim={}, hyper_dim={}, hidden_dim={}, cov={}, batch_norm={}, hyper_cov={})".format(
            self.__class__.__name__, self.x_dim, self.z_dim, self.hyper_dim, self.hidden_dim, self.cov, 
            self.batch_norm, self.hyper_cov
        )
        return s
    
    def init_batch_norm(self, mean, variance):
        if self.batch_norm:
            self.bn.moving_mean.data = mean
            self.bn.moving_variance.data = variance
        
    def mu(self, z):
        mu = self._mu + self._mu_offset(z).view(-1, self.z_dim, self.x_dim)
        return mu
    
    def lv(self, z):
        if not self.hyper_cov:
            lv = torch.repeat_interleave(self._lv, len(z), dim=0)
        else:
            lv = self._lv + self._lv_offset(z).view(len(z), -1, self.x_dim)
        return lv

    def get_distribution_class(self, z, transform=True, requires_grad=True):
        [mu, lv, tl] = self.mu(z), self.lv(z), self._tl
        
        L = make_covariance_matrix(lv, tl, cholesky=True, lv_rectify="exp")
        
        if requires_grad is False:
            mu, L = mu.data, L.data
        
        distribution = torch_dist.MultivariateNormal(mu, scale_tril=L)
        if not transform:
            return distribution
        
        transforms = [self.flow]
        if self.batch_norm:
            transforms.append(self.bn)
        
        distribution = SimpleTransformedModule(distribution, transforms)
        return distribution
    
    def mean(self, z):
        distribution = self.get_distribution_class(z)
        return distribution.mean
    
    def variance(self, z):
        distribution = self.get_distribution_class(z)
        return distribution.variance
    
    def entropy(self, z):
        distribution = self.get_distribution_class(z)
        return distribution.entropy()
    
    def log_prob(self, x, z):
        """ Component log probabilities 

        Args:
            x (torch.tensor): observation vector. size=[batch_size, x_dim]
            z (torch.tensor): hyper vector. size=[batch_size, hyper_dim]
        """
        distribution = self.get_distribution_class(z)
        return distribution.log_prob(x.unsqueeze(-2))
    
    def mixture_log_prob(self, pi, x, z):
        """ Compute mixture log probabilities 
        
        Args:
            pi (torch.tensor): mixing weights. size=[..., z_dim]
            x (torch.tensor): observervations. size[..., x_dim]
            z (torch.tensor): hyper vector. size=[batch_size, hyper_dim]
        """
        logp_pi = torch.log(pi + self.eps)
        logp_x = self.log_prob(x, z)
        logp = torch.logsumexp(logp_pi + logp_x, dim=-1)
        return logp

    def sample(self, sample_shape, z):
        """ Sample components """
        distribution = self.get_distribution_class(z)
        return distribution.rsample(sample_shape)

    def bayesian_average(self, pi, z):
        """ Compute weighted average of component means 
        
        Args:
            pi (torch.tensor): mixing weights. size=[..., z_dim]
        """
        mu = self.mean(z)
        x = torch.sum(pi.unsqueeze(-1) * mu.unsqueeze(0), dim=-2)
        return x
    
    def ancestral_sample(self, pi, z, num_samples=1, sample_mean=False, tau=0.1, hard=True):
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
            x_ = self.mean(z)
        else:
            x_ = self.sample((num_samples, pi.shape[0]), z).squeeze(1)
        x = torch.sum(z_ * x_, dim=-2)
        return x
