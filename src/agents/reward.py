import torch
import torch.nn as nn
from src.distributions.utils import kl_divergence

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