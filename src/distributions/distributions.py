import math
import torch
import torch.distributions as dist

class SkewNormal(dist.Normal):
    def __init__(self, skewness, loc, scale, validate_args=None):
        super().__init__(loc, scale, validate_args)
        self.skewness = skewness
    
    @property
    def delta(self):
        a = self.skewness 
        s = self.scale
        return a / torch.sqrt(1 + a**2) 
    
    @property
    def mean(self): 
        mu = (2 / math.pi)**0.5 * self.delta
        return self.loc + self.scale * mu
    
    @property
    def variance(self):
        coef = (1 - 2 * self.delta.pow(2) / math.pi)
        return self.scale.pow(2) * coef

    def log_prob(self, value):
        a = self.skewness
        
        # z transform for cdf
        z = (value - self.loc) / self.scale
        
        pdf = dist.Normal(self.loc, self.scale).log_prob(value)
        cdf = dist.Normal(0, 1).cdf(z * a).log()
        out = torch.tensor([2]).log() + pdf + cdf
        return out

    def rsample(self, sample_shape=torch.Size()):
        delta = self.delta[..., None, None]
        v = self.scale.pow(2)[..., None, None]
        ones = torch.ones_like(delta)
        cov_star_1 = torch.cat([ones, delta], dim=-1)
        cov_star_2 = torch.cat([delta, v], dim=-1)
        cov_star = torch.cat([cov_star_1, cov_star_2], dim=-2)
        
        # sample with reparameterization trick
        shape = list(self.loc.shape) + [2]
        eps = dist.Normal(
            torch.zeros(shape), torch.ones(shape)
        ).rsample(sample_shape)
        x = cov_star.matmul(eps.unsqueeze(-1)).squeeze(-1)
        
        x0, x1 = x[..., 0], x[..., 1]
        idx = x0 <= 0
        x1[idx] = -1 * x1[idx]
        return x1 + self.loc


""" TODO: 
test batch creation in colab 
test property functions
"""
class MultivariateSkewNormal(dist.MultivariateNormal):
    """
    The Skew-normal Distribution and Related Multivariate Families, Azzalini, 2005
    Kullback Leibler Divergence Measure for Multivariate 
        Skew-Normal Distributions, Contreras-Reyes & Arellano-Valle, 2012
    """
    def __init__(
        self, loc, skewness, covariance_matrix=None, 
        precision_matrix=None, scale_tril=None, validate_args=None
    ):
        assert skewness.shape[-1] == loc.shape[-1]
        super().__init__(
            loc, 
            covariance_matrix=covariance_matrix, 
            precision_matrix=precision_matrix, 
            scale_tril=scale_tril, 
            validate_args=validate_args
        )
        self.dim = loc.shape[-1]
        self.skewness = skewness
    
    @property
    def delta(self):
        a = self.skewness 
        cov = self.covariance_matrix
        w_inv = torch.diag(1 / torch.diagonal(cov).sqrt())
        C = w_inv.matmul(cov).matmul(w_inv)
        aCa = a.T.matmul(C).matmul(a)

        delta = C.matmul(a) / torch.sqrt(1 + aCa)
        return delta
    
    @property
    def mean(self): 
        mu = (2 / math.pi)**0.5 * self.delta
        w = torch.diagonal(self.covariance_matrix).sqrt()
        return self.loc + w * mu
    
    @property
    def variance(self):
        cov = self.covariance_matrix
        mu = (2 / math.pi)**0.5 * self.delta
        w = torch.diag(torch.diagonal(cov).sqrt())
        mu_square = mu.unsqueeze(-1).matmul(mu.unsqueeze(-2))
        wMw = w.matmul(mu_square).matmul(w)
        return cov - wMw
    
    def entropy(self):
        """ upper bound on entropy """
        dim = self.loc.shape[-1]
        ent_norm = 0.5 * dim + 0.5 * dim * math.log(2 * math.pi) \
            + 0.5 * torch.linalg.det(self.covariance_matrix).log()

        a = self.skewness 
        cov = self.covariance_matrix
        w_inv = torch.diag(1 / torch.diagonal(cov).sqrt())
        C = w_inv.matmul(cov).matmul(w_inv)
        aCa = a.T.matmul(C).matmul(a)
        
        b = 0.5 * torch.log(1 - 2/math.pi * aCa / (1 + aCa))
        return ent_norm + b

    def pdf(self, value):
        return self.log_prob(value).exp()
        
    def log_prob(self, value):
        a = self.skewness
        cov = self.covariance_matrix
        
        # z transform for cdf
        w = torch.diagonal(cov).sqrt()
        z = (value - self.loc) / w
        
        pdf = dist.MultivariateNormal(
            self.loc, scale_tril=self.scale_tril
        ).log_prob(value)
        cdf = dist.Normal(0, 1).cdf(z.matmul(a)).log()
        out = torch.tensor([2]).log() + pdf + cdf
        return out

    def rsample(self, sample_shape=torch.Size()):
        delta = self.delta
        cov = self.covariance_matrix
        cov_star_1 = torch.block_diag(torch.ones(1), cov)
        cov_star_2 = torch.block_diag(
            delta.unsqueeze(-2).flip(-1), delta.unsqueeze(-1)
        ).flip(-1)
        cov_star = cov_star_1 + cov_star_2

        shape = list(self.loc.shape)
        shape[-1] += 1
        # sample with reparameterization trick
        eps = dist.Normal(
            torch.zeros(shape), torch.ones(shape)
        ).rsample(sample_shape)
        x = eps.matmul(cov_star)

        x0, x1 = x[..., 0], x[..., 1:]
        idx = x0 <= 0
        x1[idx] = -1 * x1[idx]
        return x1 + self.loc