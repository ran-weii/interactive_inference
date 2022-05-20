import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, kl
from src.distributions.utils import kl_divergence, make_covariance_matrix

class ExpectedFreeEnergy(nn.Module):
    def __init__(self, hmm, obs_model):
        super().__init__()
        self.state_dim = hmm.state_dim
        self.hmm = hmm
        self.obs_model = obs_model
        
        self.C = nn.Parameter(torch.randn(1, self.state_dim), requires_grad=True)
        nn.init.xavier_normal_(self.C, gain=1.)
    
    def __repr__(self):
        s = "{}(s={})".format(
            self.__class__.__name__, self.state_dim
        )
        return s
    
    def forward(self, s_next, b_next, A=None, C=None):
        """
        Args:
            s_next (torch.tensor): predictive state distribution [..., state_dim]
            b_next (torch.tensor): predictive belief distribution [..., state_dim]
            A (torch.tensor): observaiton model parameters
            C (torch.tensor): reward model parameters

        Returns:
            r (torch.tensor): negative EFE belief action reward [..., act_dim, state_dim]
        """
        assert b_next.shape[-1] == self.C.shape[-1]

        # reshape C to align with belief shape
        C = self.C if C is None else C
        diff_dims = len(b_next.shape) - len(self.C.shape)
        target_shape = list(C.shape) + [1 for i in range(diff_dims)]
        
        H = self.obs_model.entropy(A).view(target_shape).transpose(1, -1)
        C = torch.softmax(C, dim=-1).view(target_shape).transpose(1, -1)

        kl = kl_divergence(b_next, C)
        eh = torch.sum(s_next * H, dim=-1)
        r = -kl - eh
        return r


class GeneralizedFreeEnergy(nn.Module):
    def __init__(self, hmm, obs_model):
        super().__init__()
        self.state_dim = hmm.state_dim
        self.obs_dim = obs_model.x_dim
        self.hmm = hmm
        self.obs_model = obs_model
        self.parameter_size = [
            self.obs_dim,
            self.obs_dim,
            self.obs_dim * self.obs_dim
        ]
        
        self.C_mu = nn.Parameter(torch.randn(1, self.obs_dim), requires_grad=True)
        self.C_lv = nn.Parameter(torch.randn(1, self.obs_dim), requires_grad=True)
        self.C_tl = nn.Parameter(torch.randn(1, self.obs_dim, self.obs_dim), requires_grad=True)
        
        nn.init.normal_(self.C_mu, mean=0, std=1)
        nn.init.normal_(self.C_lv, mean=0, std=0.01)
        nn.init.normal_(self.C_tl, mean=0, std=0.01)
    
    def __repr__(self):
        s = "{}(o={})".format(
            self.__class__.__name__, self.obs_dim
        )
        return s
    
    def get_distribution_class(self):
        L = make_covariance_matrix(self.C_lv, self.C_tl, cholesky=True, lv_rectify="exp")
        distribution = MultivariateNormal(self.C_mu, scale_tril=L)
        return distribution
    
    @property
    def C(self):
        """ Log of cross entropy to compare with EFE reward """
        target_dist = self.get_distribution_class()
        pred_dist = self.obs_model.get_distribution_class(transform=False)
        
        kld = kl.kl_divergence(pred_dist, target_dist)
        ent = pred_dist.entropy()
        cross_ent = kld + ent
        return -torch.log(cross_ent + 1e-6)

    def forward(self, s_next, b_next, A=None, C=None):
        """
        Args:
            s_next (torch.tensor): predictive state distribution [..., state_dim]
            b_next (torch.tensor): predictive belief distribution [..., state_dim]
            A (torch.tensor): observaiton model parameters
            C (torch.tensor): reward model parameters

        Returns:
            r (torch.tensor): negative GFE belief action reward [..., act_dim, state_dim]
        """
        target_dist = self.get_distribution_class()
        pred_dist = self.obs_model.get_distribution_class(transform=False)
        
        kld = kl.kl_divergence(pred_dist, target_dist)
        ent = pred_dist.entropy()
        cross_ent = kld + ent

        r = -torch.sum(s_next * cross_ent.unsqueeze(-2).unsqueeze(-2), dim=-1)
        return r