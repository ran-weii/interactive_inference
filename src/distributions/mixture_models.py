import torch
import torch.nn as nn
import torch.distributions as torch_dist
from src.distributions.flows import SimpleTransformedModule, BatchNormTransform
from src.distributions.utils import make_covariance_matrix, straight_through_sample

class ConditionalGaussian(nn.Module):
    """ Conditional gaussian distribution used to create mixture distributions """
    def __init__(self, x_dim, z_dim, cov="full", batch_norm=True, device=torch.device("cpu")):
        """
        Args:
            x_dim (int): observed output dimension
            z_dim (int): latent conditonal dimension
            cov (str): covariance type ["diag", "full"]
            batch_norm (bool, optional): whether to use input batch normalization. default=True
        """
        super().__init__()
        assert cov in ["diag", "full"]
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.cov = cov
        self.parameter_size = [
            z_dim * x_dim,
            z_dim * x_dim,
            z_dim * x_dim * x_dim
        ]
        self.batch_norm = batch_norm
        self.device = device
        
        self.mu = nn.Parameter(torch.randn(1, z_dim, x_dim), requires_grad=True)
        self.lv = nn.Parameter(torch.randn(1, z_dim, x_dim), requires_grad=True)
        self.tl = nn.Parameter(torch.randn(1, z_dim, x_dim, x_dim), requires_grad=True)
        
        nn.init.normal_(self.mu, mean=0, std=1)
        nn.init.normal_(self.lv, mean=0, std=0.01)
        nn.init.normal_(self.tl, mean=0, std=0.01)
        
        if cov == "diag":
            del self.tl
            self.parameter_size = self.parameter_size[:-1]
            self.tl = torch.zeros(1, z_dim, x_dim, x_dim).to(device)
        
        if batch_norm:
            self.bn = BatchNormTransform(x_dim, momentum=0.1, affine=False, device=device)
        
    def __repr__(self):
        s = "{}(x_dim={}, z_dim={}, cov={})".format(
            self.__class__.__name__, self.x_dim, self.z_dim, self.cov
        )
        return s
    
    def get_distribution_class(self, transform=True, requires_grad=True):
        [mu, lv, tl] = self.mu, self.lv, self.tl
        L = make_covariance_matrix(lv, tl, cholesky=True, lv_rectify="exp")
        
        if requires_grad is False:
            mu, L = mu.data, L.data

        distribution = torch_dist.MultivariateNormal(mu, scale_tril=L)
        
        if self.batch_norm and transform:
            distribution = SimpleTransformedModule(distribution, [self.bn])
        return distribution
    
    def mean(self, params=None):
        distribution = self.get_distribution_class(params)
        return distribution.mean
    
    def variance(self, params=None):
        distribution = self.get_distribution_class(params)
        return distribution.variance
    
    def entropy(self, params=None):
        distribution = self.get_distribution_class(params)
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
        logp_pi = torch.log(pi + 1e-6)
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
        post = torch.softmax(torch.log(prior + 1e-6) + logp_x, dim=-1)
        return post

    def bayesian_average(self, pi):
        """ Compute weighted average of component means 
        
        Args:
            pi (torch.tensor): mixing weights. size=[..., z_dim]
        """
        mu = self.mean()
        x = torch.sum(pi.unsqueeze(-1) * mu.unsqueeze(0), dim=-2)
        return x
    
    def ancestral_sample(self, pi, num_samples=1, sample_mean=False):
        """ Ancestral sampling
        
        Args:
            pi (torch.tensor): mixing weights
            num_samples (int, optional): number of samples to draw. Default=1
            sample_mean (bool, optional): whether to sample component mean. Default=False
        """
        z_ = torch_dist.RelaxedOneHotCategorical(1, pi).rsample((num_samples,))
        z_ = straight_through_sample(z_, dim=-1).unsqueeze(-1)
        
        # sample component
        if sample_mean:
            x_ = self.mean()
        else:
            x_ = self.sample((num_samples, pi.shape[0])).squeeze(1)
        x = torch.sum(z_ * x_, dim=-2)
        return x
