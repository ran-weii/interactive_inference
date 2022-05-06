import torch
import torch.nn as nn
from src.distributions.flows import SimpleTransformedModule, BatchNormTransform
from src.distributions.utils import straight_through_sample, rectify

class FactoredHiddenMarkovModel(nn.Module):
    def __init__(self, state_dim, act_dim, num_agents):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.num_agents = num_agents
        self.parameter_size = [act_dim**num_agents * state_dim * state_dim, state_dim]
        
        B_shape = [1] + [act_dim]*num_agents + [state_dim]*2
        self.B = nn.Parameter(
            torch.randn(B_shape), requires_grad=True
        )
        self.D = nn.Parameter(torch.randn(1, state_dim), requires_grad=True)
        nn.init.xavier_normal_(self.B, gain=1.)
        nn.init.xavier_normal_(self.D, gain=1.)
        
    def __repr__(self):
        s = "{}(s={}, a={}, n={})".format(
            self.__class__.__name__, self.state_dim, self.act_dim, self.num_agents
        )
        return s
    
    def transform_parameters(self, B):
        B_shape = [-1] + [self.act_dim]*self.num_agents + [self.state_dim]*2
        return B.view(B_shape)
    
    def forward(self, logp_o, a, b, B=None):
        """ 
        Args:
            logp_o (torch.tensor): observation log likelihood [batch_size, state_dim]
            a (torch.tensor): soft action vector [batch_size, num_agents, act_dim]
            b (torch.tensor): belief [batch_size, state_dim]
            B (torch.tensor, optional): transition parameters 
                [batch_size, act_dim, state_dim, state_dim]

        Returns:
            b_t(torch.tensor): next belief [batch_size, state_dim]
        """
        if B is not None:
            B = self.transform_parameters(B)
            B = torch.softmax(B, dim=-1)
        else:
            B = torch.softmax(self.B, dim=-1)
        
        B_a = [torch.empty(0)] * (self.num_agents + 1)
        B_a[0] = B
        for i in range(self.num_agents):
            a_i = a[:, i]
            a_i = a_i.view(list(a_i.shape) + torch.ones(len(B_a[i].shape) - 2).long().tolist())
            B_a[i+1] = torch.sum(B_a[i] * a_i, dim=1)

        logp_s = torch.log(torch.sum(b.unsqueeze(-1) * (B_a[-1]), dim=-2) + 1e-6)
        b_t = torch.softmax(logp_o + logp_s, dim=-1)
        return b_t

class FactoredConditionalDistribution(nn.Module):
    def __init__(self, x_dim, z_dim, dist="mvn", cov="full", batch_norm=True):
        """
        Args:
            x_dim (int): observed output dimension
            z_dim (int): latent conditonal dimension
            dist (str): distribution type ["mvn", "mvsn"]
            cov (str): covariance type ["diag", "full"]
            batch_norm (bool, optional): use input batch normalization. default=True
        """
        super().__init__()
        assert dist in ["mvn", "mvsn", "laplace"]
        assert cov in ["diag", "full"]
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.dist = dist
        self.cov = cov
        self.parameter_size = [
            z_dim * x_dim,
            z_dim * x_dim,
            z_dim * x_dim
        ]
        self.batch_norm = batch_norm
        
        self.mu = nn.Parameter(torch.randn(1, z_dim, x_dim), requires_grad=True)
        self.lv = nn.Parameter(torch.randn(1, z_dim, x_dim), requires_grad=True)
        self.sk = nn.Parameter(torch.randn(1, z_dim, x_dim), requires_grad=True)
        
        nn.init.normal_(self.mu, mean=0, std=1)
        nn.init.normal_(self.lv, mean=0, std=0.01)
        nn.init.normal_(self.sk, mean=0, std=0.01)
        
        if dist in ["mvn", "laplace"]:
            nn.init.constant_(self.sk, 0)
            self.sk.requires_grad = False
            self.parameter_size = self.parameter_size[:-1]
            self.sk.data = torch.zeros_like(self.sk.data)
        
        if batch_norm:
            self.bn = BatchNormTransform(x_dim, momentum=0.1, affine=False)
        
    def __repr__(self):
        s = "{}(x_dim={}, z_dim={}, class={}, cov={})".format(
            self.__class__.__name__, self.x_dim, self.z_dim, self.dist, self.cov
        )
        return s
    
    def transform_parameters(self, params):
        if self.dist in ["mvn", "laplace"]:
            mu, lv = torch.split(params, self.parameter_size, dim=-1)
            mu = mu.view(-1, self.z_dim, self.x_dim)
            lv = lv.view(-1, self.z_dim, self.x_dim)
            sk = torch.zeros(len(params), self.z_dim, self.x_dim)
        else:
            raise NotImplementedError
        
        return mu, lv, sk
    
    def get_distribution_class(self, params=None, transform=True):
        if params is not None:
            [mu, lv, sk] = self.transform_parameters(params)
        else:
            [mu, lv, sk] = self.mu, self.lv, self.sk

        if self.dist == "mvn":
            distribution = torch.distributions.Normal(mu, rectify(lv))
        elif self.dist == "laplace":
            distribution = torch.distributions.Laplace(mu, rectify(lv))
        
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
    
    def log_prob(self, x, params=None):
        """
        Args:
            x (torch.tensor): [batch_size, x_dim]
            params (torch.tensor, optional): parameter vector. Defaults to None.
        """
        distribution = self.get_distribution_class(params)
        return distribution.log_prob(x.unsqueeze(-2))
    
    def mixture_log_prob(self, pi, x, params=None):
        logp_pi = torch.log(pi + 1e-6).transpose(-1, -2)
        logp_x = self.log_prob(x, params=params)
        logp = torch.logsumexp(logp_pi + logp_x, dim=-2)
        return logp

    def sample(self, sample_shape=torch.Size(), params=None):
        distribution = self.get_distribution_class(params)
        return distribution.rsample(sample_shape)
    
    def infer(self, prior, x, logp_x=None, params=None):
        if logp_x is None:
            logp_x = self.log_prob(x, params)
        logp_x = logp_x.transpose(-1, -2)
        post = torch.softmax(torch.log(prior + 1e-6) + logp_x, dim=-1)
        return post

    def bayesian_average(self, pi, params=None):
        mu = self.mean(params).transpose(-1, -2)
        x = torch.sum(pi * mu.unsqueeze(0), dim=-1)
        return x
    
    def ancestral_sample(self, pi, num_samples, params=None):
        z_ = torch.distributions.RelaxedOneHotCategorical(1, pi).rsample((num_samples,))
        z_ = straight_through_sample(z_, dim=-1)
        
        # sample component
        x_ = self.sample((num_samples, pi.shape[-3]), params).squeeze(-3)
        x_ = x_.transpose(-1, -2).unsqueeze(1)
        x = torch.sum(z_ * x_, dim=-1).squeeze(0)
        return x
