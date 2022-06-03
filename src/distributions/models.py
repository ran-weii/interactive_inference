import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from src.distributions.distributions import MultivariateSkewNormal
from src.distributions.flows import SimpleTransformedModule, BatchNormTransform
from src.distributions.utils import make_covariance_matrix, straight_through_sample

class EmbeddedTransitionModel(nn.Module):
    """ CP tensor decomposition of transition matrix """
    def __init__(self, state_dim, act_dim, rank=10):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.rank = rank
        
        self.e = nn.Parameter(torch.randn(act_dim, 1, 1, rank))
        self.w1 = nn.Parameter(torch.randn(1, state_dim, 1, rank))
        self.w2 = nn.Parameter(torch.randn(1, 1, state_dim, rank))
        
        nn.init.xavier_normal_(self.w1, gain=1.)
        nn.init.xavier_normal_(self.w2, gain=1.)
        nn.init.xavier_normal_(self.e, gain=1.)
    
    @property
    def transition(self):
        a = torch.eye(self.act_dim)
        return self.forward(a)
    
    """ TODO: find better embedding method using tensor decomposition """
    def forward(self, a):
        w_a = torch.sum(self.w1 * self.w2 * self.e, dim=-1)
        return w_a


class HiddenMarkovModel(nn.Module):
    def __init__(self, state_dim, act_dim, rank=10):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.parameter_size = [act_dim * state_dim * state_dim, state_dim]
        self.rank = rank

        if rank != 0:
            self.B = EmbeddedTransitionModel(state_dim, act_dim, rank=rank)
            self.parameter_size[0] = state_dim * state_dim + act_dim * state_dim
        else:
            self.B = nn.Parameter(
                torch.randn(1, act_dim, state_dim, state_dim), requires_grad=True
            )
            nn.init.xavier_normal_(self.B, gain=1.)
        self.D = nn.Parameter(torch.randn(1, state_dim), requires_grad=True)
        nn.init.xavier_normal_(self.D, gain=1.)
        
    def __repr__(self):
        s = "{}(s={}, a={}, rank={})".format(
            self.__class__.__name__, self.state_dim, self.act_dim, self.rank
        )
        return s
    
    """ TODO: temp solution to work with embedding """
    def get_default_parameters(self):
        if self.rank != 0:
            return self.B.transition.unsqueeze(0)
        else:
            return self.B

    def transform_parameters(self, B):
        return B.view(-1, self.act_dim, self.state_dim, self.state_dim)
    
    def forward(self, logp_o, a, b, B=None):
        """ 
        Args:
            logp_o (torch.tensor): observation log likelihood [batch_size, state_dim]
            a (torch.tensor): soft action vector [batch_size, act_dim]
            b (torch.tensor): belief [batch_size, state_dim]
            B (torch.tensor, optional): transition parameters 
                [batch_size, act_dim, state_dim, state_dim]

        Returns:
            b_t(torch.tensor): next belief [batch_size, state_dim]
        """
        if B is not None and self.rank != 0:
            B = self.transform_parameters(B)
            B = torch.softmax(B, dim=-1)
        else:
            B = self.get_default_parameters()
            B = torch.softmax(B, dim=-1)
        
        B_a = torch.sum(B * a.unsqueeze(-1).unsqueeze(-1), dim=-3)
        logp_s = torch.log(torch.sum(b.unsqueeze(-1) * (B_a), dim=-2) + 1e-6)
        b_t = torch.softmax(logp_o + logp_s, dim=-1)
        return b_t
    
    
class ConditionalDistribution(nn.Module):
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
        assert dist in ["mvn", "mvsn"]
        assert cov in ["diag", "full"]
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.dist = dist
        self.cov = cov
        self.parameter_size = [
            z_dim * x_dim,
            z_dim * x_dim,
            z_dim * x_dim * x_dim,
            z_dim * x_dim
        ]
        self.batch_norm = batch_norm
        
        self.mu = nn.Parameter(torch.randn(1, z_dim, x_dim), requires_grad=True)
        self.lv = nn.Parameter(torch.randn(1, z_dim, x_dim), requires_grad=True)
        self.tl = nn.Parameter(torch.randn(1, z_dim, x_dim, x_dim), requires_grad=True)
        self.sk = nn.Parameter(torch.randn(1, z_dim, x_dim), requires_grad=True)
        
        nn.init.normal_(self.mu, mean=0, std=1)
        nn.init.normal_(self.lv, mean=0, std=0.01)
        nn.init.normal_(self.tl, mean=0, std=0.01)
        nn.init.normal_(self.sk, mean=0, std=0.01)
        
        if dist == "mvn":
            nn.init.constant_(self.sk, 0)
            self.sk.requires_grad = False
            self.parameter_size = self.parameter_size[:-1]
            self.sk.data = torch.zeros_like(self.sk.data)
        
        if cov == "diag":
            nn.init.constant_(self.tl, 0)
            self.tl.requires_grad = False
            self.parameter_size = self.parameter_size[:-1]
            self.tl.data = torch.zeros_like(self.tl.data)
        
        if batch_norm:
            self.bn = BatchNormTransform(x_dim, momentum=0.1, affine=False)
        
    def __repr__(self):
        s = "{}(x_dim={}, z_dim={}, class={}, cov={})".format(
            self.__class__.__name__, self.x_dim, self.z_dim, self.dist, self.cov
        )
        return s
    
    def transform_parameters(self, params):
        if self.dist == "mvn":
            if self.cov == "full":
                mu, lv, tl = torch.split(params, self.parameter_size, dim=-1)
                mu = mu.view(-1, self.z_dim, self.x_dim)
                lv = lv.view(-1, self.z_dim, self.x_dim)
                tl = tl.view(-1, self.z_dim, self.x_dim, self.x_dim)
                sk = torch.zeros(len(params), self.z_dim, self.x_dim)
            elif self.cov == "diag":
                mu, lv = torch.split(params, self.parameter_size, dim=-1)
                mu = mu.view(-1, self.z_dim, self.x_dim)
                lv = lv.view(-1, self.z_dim, self.x_dim)
                tl = torch.zeros(len(params), self.z_dim, self.x_dim, self.x_dim)
                sk = torch.zeros(len(params), self.z_dim, self.x_dim)
            else:
                raise NotImplementedError
        elif self.dist == "mvsn":
            if self.cov == "full":
                mu, lv, tl, sk = torch.split(params, self.parameter_size, dim=-1)
                mu = mu.view(-1, self.z_dim, self.x_dim)
                lv = lv.view(-1, self.z_dim, self.x_dim)
                tl = tl.view(-1, self.z_dim, self.x_dim, self.x_dim)
                sk = sk.view(-1, self.z_dim, self.x_dim)
            elif self.cov == "diag":
                mu, lv, sk = torch.split(params, self.parameter_size, dim=-1)
                mu = mu.view(-1, self.z_dim, self.x_dim)
                lv = lv.view(-1, self.z_dim, self.x_dim)
                tl = torch.zeros(len(params), self.z_dim, self.x_dim, self.x_dim)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        return mu, lv, tl, sk
    
    def get_distribution_class(self, params=None, transform=True, requires_grad=True):
        if params is not None:
            [mu, lv, tl, sk] = self.transform_parameters(params)
        else:
            [mu, lv, tl, sk] = self.mu, self.lv, self.tl, self.sk
        L = make_covariance_matrix(lv, tl, cholesky=True, lv_rectify="exp")
        
        if requires_grad is False:
            mu = mu.data
            L = L.data
            sk = sk.data

        if self.dist == "mvn":
            distribution = MultivariateNormal(mu, scale_tril=L)
        elif self.dist == "mvsn":
            distribution = MultivariateSkewNormal(mu, sk, scale_tril=L)
        
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
        logp_pi = torch.log(pi + 1e-6)
        logp_x = self.log_prob(x, params=params)
        logp = torch.logsumexp(logp_pi + logp_x, dim=-1)
        return logp

    def sample(self, sample_shape, params=None):
        distribution = self.get_distribution_class(params)
        return distribution.rsample(sample_shape)
    
    def infer(self, prior, x, logp_x=None, params=None):
        if logp_x is None:
            logp_x = self.log_prob(x, params)
        post = torch.softmax(torch.log(prior + 1e-6) + logp_x, dim=-1)
        return post

    def bayesian_average(self, pi, params=None):
        mu = self.mean(params)
        x = torch.sum(pi.unsqueeze(-1) * mu.unsqueeze(0), dim=-2)
        return x
    
    def ancestral_sample(self, pi, num_samples, params=None):
        num_samples_ = 1 if num_samples == 0 else num_samples
        z_ = torch.distributions.RelaxedOneHotCategorical(1, pi).rsample((num_samples_,))
        z_ = straight_through_sample(z_, dim=-1).unsqueeze(-1)
        
        # sample component
        if num_samples == 0:
            x_ = self.mean(params)
        else:
            x_ = self.sample((num_samples, pi.shape[0]), params).squeeze(1)
        x = torch.sum(z_ * x_, dim=-2)
        # z_ = torch.distributions.RelaxedOneHotCategorical(1, pi).rsample((num_samples,))
        # z_ = straight_through_sample(z_, dim=-1).unsqueeze(-1)
        
        # # sample component
        # x_ = self.sample((num_samples, pi.shape[0]), params).squeeze(1)
        # x = torch.sum(z_ * x_, dim=-2)
        return x


class GeneralizedLinearModel(ConditionalDistribution):
    def __init__(self, x_dim, z_dim, dist="mvn", cov="full", batch_norm=False):
        """
        Args:
            x_dim (int): observed output dimension
            z_dim (int): latent conditonal dimension
            dist (str): distribution type ["mvn", "mvsn"]
            cov (str): covariance type ["diag", "full"]
            batch_norm (bool, optional): use input batch normalization. default=True
        """
        super().__init__(x_dim, z_dim, dist=dist, cov=cov, batch_norm=batch_norm)
        self.b_lv = nn.Parameter(torch.randn(1, z_dim, x_dim), requires_grad=True)
        self.b_tl = nn.Parameter(torch.randn(1, z_dim, x_dim, x_dim), requires_grad=True)
        self.b_sk = nn.Parameter(torch.randn(1, z_dim, x_dim), requires_grad=True)
        
        nn.init.normal_(self.b_lv, mean=0, std=0.01)
        nn.init.normal_(self.b_tl, mean=0, std=0.01)
        nn.init.normal_(self.b_sk, mean=0, std=0.01)
        
        if self.cov == "diag":
            self.b_tl.requires_grad = False
            self.b_tl.data = torch.zeros_like(self.b_tl.data)

        if self.dist == "mvn":
            self.b_sk.requires_grad = False
            self.b_sk.data = torch.zeros_like(self.b_sk.data)
    
    def get_mixture_distribution_class(self, pi, params=None):
        if params is not None:
            [mu, lv, tl, sk] = self.transform_parameters(params)
        else:
            [mu, lv, tl, sk] = self.mu, self.lv, self.tl, self.sk
        
        mu_ = torch.sum(pi[..., None] * mu, dim=-2)
        lv_ = torch.sum(pi[..., None] * (lv * mu.abs() + self.b_lv), dim=-2)
        tl_ = torch.sum(pi[..., None, None] * (tl * mu.abs()[..., None] + self.b_tl), dim=-3)
        sk_ = torch.sum(pi[..., None] * (sk + self.b_sk), dim=-2)
        L = make_covariance_matrix(lv_, tl_, cholesky=True)
        
        if self.dist == "mvn":
            distribution = MultivariateNormal(mu_, scale_tril=L)
        elif self.dist == "mvsn":
            distribution = MultivariateSkewNormal(mu_, sk_, scale_tril=L)
        
        if self.batch_norm:
            distribution = SimpleTransformedModule(distribution, [self.bn])
        return distribution

    def mixture_log_prob(self, pi, x, params=None):
        distribution = self.get_mixture_distribution_class(pi, params)
        return distribution.log_prob(x)
    
    def infer(self, prior, x, logp_x=None, params=None):
        return prior

    def ancestral_sample(self, pi, num_samples, params=None):
        distribution = self.get_mixture_distribution_class(pi, params)
        return distribution.sample((num_samples,))
        

class MixtureDensityNetwork(ConditionalDistribution):
    def __init__(self, x_dim, z_dim, dist="mvn", cov="full", batch_norm=False):
        super().__init__(x_dim, z_dim, dist, cov, batch_norm)
        self.fc1 = nn.Linear(z_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.head = nn.Linear(64, x_dim * 2)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.head(x)
        [mu, lv] = torch.split(x, self.x_dim, dim=-1)
        return mu, lv
    
    def get_mixture_distribution_class(self, pi, params=None):
        mu, lv = self.forward(pi)
        tl = torch.zeros(lv.shape[:-1] + (self.x_dim,) * 2)
        L = make_covariance_matrix(lv, tl, cholesky=True)
        
        distribution = MultivariateNormal(mu, scale_tril=L)

        if self.batch_norm:
            distribution = SimpleTransformedModule(distribution, [self.bn])
        return distribution

    def mixture_log_prob(self, pi, x, params=None):
        distribution = self.get_mixture_distribution_class(pi, params)
        return distribution.log_prob(x)

    def bayesian_average(self, pi, params=None):
        distribution = self.get_mixture_distribution_class(pi, params)
        return distribution.mean

    def ancestral_sample(self, pi, num_samples, params=None):
        distribution = self.get_mixture_distribution_class(pi, params)
        return distribution.sample((num_samples, ))