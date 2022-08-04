import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torch_dist
from src.distributions.flows import SimpleTransformedModule, BatchNormTransform
from src.distributions.flows import TanhTransform
from src.distributions.utils import make_covariance_matrix
from src.distributions.nn_models import Model

class ConditionalGaussian(Model):
    """ Conditional gaussian distribution used to create mixture distributions """
    def __init__(
        self, x_dim, z_dim, cov="full", batch_norm=True, 
        use_tanh=False, limits=None
        ):
        """
        Args:
            x_dim (int): observed output dimension
            z_dim (int): latent conditonal dimension
            cov (str): covariance type. choices=["diag", "full", "tied"]
            batch_norm (bool, optional): whether to use input batch normalization. default=True
        """
        super().__init__()
        assert cov in ["diag", "full", "tied"]
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.cov = cov
        self.batch_norm = batch_norm
        self.use_tanh = use_tanh
        self.eps = 1e-6
        
        self.mu = nn.Parameter(torch.randn(1, z_dim, x_dim), requires_grad=True)
        nn.init.normal_(self.mu, mean=0, std=1)
    
        if cov == "full":
            self.lv = nn.Parameter(torch.randn(1, z_dim, x_dim), requires_grad=True)
            self.tl = nn.Parameter(torch.randn(1, z_dim, x_dim, x_dim), requires_grad=True)
            nn.init.normal_(self.lv, mean=0, std=0.01)
            nn.init.normal_(self.tl, mean=0, std=0.01)
        elif cov == "diag":
            self.lv = nn.Parameter(torch.randn(1, z_dim, x_dim), requires_grad=True)
            self.tl = nn.Parameter(torch.zeros(1, z_dim, x_dim, x_dim), requires_grad=False)
            nn.init.normal_(self.lv, mean=0, std=0.01)
        elif cov == "tied":
            self.lv = nn.Parameter(torch.randn(1, 1, x_dim), requires_grad=True)
            self.tl = nn.Parameter(torch.zeros(1, z_dim, x_dim, x_dim), requires_grad=False)
            nn.init.normal_(self.lv, mean=0, std=0.01)

        if batch_norm:
            self.bn = BatchNormTransform(x_dim, momentum=0.1, affine=False)

        if use_tanh:
            self.tanh_transform = TanhTransform(limits)
        
    def __repr__(self):
        s = "{}(x_dim={}, z_dim={}, cov={}, batch_norm={}, use_tanh={})".format(
            self.__class__.__name__, self.x_dim, self.z_dim, self.cov, 
            self.batch_norm, self.use_tanh
        )
        return s
    
    def init_batch_norm(self, mean, variance):
        self.bn.moving_mean.data = mean
        self.bn.moving_variance.data = variance

    def get_distribution_class(self, requires_grad=True):
        [mu, lv, tl] = self.mu, self.lv, self.tl
        L = make_covariance_matrix(lv, tl, cholesky=True, lv_rectify="exp")
        
        if requires_grad is False:
            mu, L = mu.data, L.data

        distribution = torch_dist.MultivariateNormal(mu, scale_tril=L)
        
        transforms = []
        if self.batch_norm:
            transforms.append(self.bn)
        if self.use_tanh:
            transforms.append(self.tanh_transform)
        distribution = SimpleTransformedModule(distribution, transforms)
        return distribution
    
    def mean(self):
        distribution = self.get_distribution_class()
        return distribution.mean
    
    def variance(self):
        distribution = self.get_distribution_class()
        return distribution.variance
    
    def entropy(self):
        distribution = self.get_distribution_class()
        return distribution.entropy()
    
    def log_prob(self, x):
        """ Component log probabilities 

        Args:
            x (torch.tensor): size=[batch_size, x_dim]
        """
        distribution = self.get_distribution_class()
        return distribution.log_prob(x.unsqueeze(-2))
    
    def mixture_log_prob(self, pi, x):
        """ Compute mixture log probabilities 
        
        Args:
            pi (torch.tensor): mixing weights. size=[..., z_dim]
            x (torch.tensor): observervations. size[..., x_dim]
        """
        logp_pi = torch.log(pi + self.eps)
        logp_x = self.log_prob(x)
        logp = torch.logsumexp(logp_pi + logp_x, dim=-1)
        return logp

    def sample(self, sample_shape=torch.Size()):
        """ Sample components """
        distribution = self.get_distribution_class()
        return distribution.rsample(sample_shape)
    
    def infer(self, prior, x, logp_x=None):
        """ Compute posterior distributions

        Args:
            prior (torch.tensor): prior probabilities. size=[..., z_dim]
            x (torch.tensor): observations. size=[..., x_dim]
            logp_x (torch.tensor, None, optional): log likelihood to avoid
                computing again. default=None

        Returns:
            post (torch.tensor): posterior probabilities. size=[..., z_dim]
        """
        if logp_x is None:
            logp_x = self.log_prob(x)
        post = torch.softmax(torch.log(prior + self.eps) + logp_x, dim=-1)
        return post

    def bayesian_average(self, pi):
        """ Compute weighted average of component means 
        
        Args:
            pi (torch.tensor): mixing weights. size=[..., z_dim]
        """
        mu = self.mean()
        x = torch.sum(pi.unsqueeze(-1) * mu.unsqueeze(0), dim=-2)
        return x
    
    def ancestral_sample(self, pi, num_samples=1, sample_mean=False, tau=0.1, hard=True):
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


class HyperConditionalGaussian(Model):
    """ Hypernet version of conditional gaussian distribution
    
    Hyper vector parameterizes mu, lv and tl are shared
    """
    def __init__(
        self, x_dim, z_dim, hyper_dim, cov="full", batch_norm=True, 
        use_tanh=False, limits=None
        ):
        """
        Args:
            x_dim (int): observed output dimension
            z_dim (int): latent conditonal dimension
            hyper_dim (int): hyper dimension
            cov (str): covariance type ["diag", "full"]
            batch_norm (bool, optional): whether to use input batch normalization. default=True
        """
        super().__init__()
        assert cov in ["diag", "full"]
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hyper_dim = hyper_dim
        self.cov = cov
        self.batch_norm = batch_norm
        self.use_tanh = use_tanh
        self.eps = 1e-6
        
        self._mu = nn.Linear(hyper_dim, z_dim * x_dim)
        self.lv = nn.Parameter(torch.randn(1, z_dim, x_dim))
        self.tl = nn.Parameter(torch.randn(1, z_dim, x_dim, x_dim), requires_grad=True)
        nn.init.normal_(self.lv, mean=0, std=0.01)
        nn.init.normal_(self.tl, mean=0, std=0.01)
    
        if cov == "diag":
            del self.tl
            self.parameter_size = self.parameter_size[:-1]
            self.tl = torch.zeros(1, z_dim, x_dim, x_dim).to(self.device)
        
        if batch_norm:
            self.bn = BatchNormTransform(x_dim, momentum=0.1, affine=False)

        if use_tanh:
            self.tanh_transform = TanhTransform(limits)
        
    def __repr__(self):
        s = "{}(x_dim={}, z_dim={}, hyper_dim={}, cov={}, batch_norm={}, use_tanh={})".format(
            self.__class__.__name__, self.x_dim, self.z_dim, self.hyper_dim, self.cov, 
            self.batch_norm, self.use_tanh
        )
        return s
    
    def init_batch_norm(self, mean, variance):
        self.bn.moving_mean.data = mean
        self.bn.moving_variance.data = variance
    
    def mu(self, z):
        return self._mu(z).view(-1, self.z_dim, self.x_dim)

    def get_distribution_class(self, z, requires_grad=True):
        [mu, lv, tl] = self.mu(z), self.lv, self.tl
        lv = torch.repeat_interleave(lv, len(mu), dim=0)
        L = make_covariance_matrix(lv, tl, cholesky=True, lv_rectify="exp")
        
        if requires_grad is False:
            mu, L = mu.data, L.data
        
        distribution = torch_dist.MultivariateNormal(mu, scale_tril=L)
        
        transforms = []
        if self.batch_norm:
            transforms.append(self.bn)
        if self.use_tanh:
            transforms.append(self.tanh_transform)
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

    def sample(self, z, sample_shape=torch.Size()):
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
            x_ = self.sample(z, (num_samples, pi.shape[0])).squeeze(1)
        x = torch.sum(z_ * x_, dim=-2)
        return x
