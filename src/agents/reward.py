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

    def get_default_params(self):
        theta = {
            "A": None, "B": self.hmm.B, "C": self.C, 
            "D": None, "F": None, "tau": None
        }
        return theta

    def state_reward(self, theta=None):
        if theta == None:
            theta = self.get_default_params()

        H = self.obs_model.entropy(theta["A"]).unsqueeze(-2)
        C = torch.softmax(theta["C"], dim=-1).unsqueeze(-2).unsqueeze(-2)
        B = torch.softmax(self.hmm.transform_parameters(theta["B"]), dim=-1)

        kl = kl_divergence(B, C)
        eh = torch.sum(B * H.unsqueeze(-2), dim=-1)
        R = -kl - eh
        return R