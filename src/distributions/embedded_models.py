import torch
import torch.nn as nn

class LMDPLayer(nn.Module):
    """ Linearly solvable MDP transition model 
    
    Parameters:
        w: passive dynamics embedding
        e: control embedding
    """
    def __init__(self, state_dim, act_dim, ctl_dim):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.ctl_dim = ctl_dim

        self.w = nn.Parameter(torch.randn(state_dim, state_dim))
        self.e = nn.Parameter(torch.randn(ctl_dim, act_dim, state_dim))

        nn.init.xavier_normal_(self.w, gain=1.)
        nn.init.xavier_normal_(self.e, gain=1.)
    
    def __repr__(self):
        s = "{}(s={}, a={}, u={})".format(
            self.__class__.__name__, self.state_dim, self.act_dim, self.ctl_dim
        )
        return s
    
    @property
    def passive_dynamics(self):
        return torch.softmax(self.w, dim=-1)

    def forward(self, a):
        """
        Args:
            a (torch.tensor): stacked control vectors [batch_size, ctl_dim, act_dim]

        returns:
            transition (torch.tensor): controlled transition matrix [batch_size, state_dim, state_dim]
        """
        e_a = torch.sum(a.unsqueeze(-1) * self.e, dim=-2).sum(-2)
        w_a = self.w + e_a.unsqueeze(-2)
        B_a = torch.softmax(w_a, dim=-1)
        return B_a


class EmbeddedHiddenMarkovModel(nn.Module):
    def __init__(self, state_dim, act_dim, ctl_dim):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.ctl_dim = ctl_dim
        
        self.lmdp = LMDPLayer(state_dim, act_dim, ctl_dim)
        self.D = nn.Parameter(torch.randn(1, state_dim), requires_grad=True)
        
        nn.init.xavier_normal_(self.D, gain=1.)
        
    def __repr__(self):
        s = "{}(s={}, a={}, u={})".format(
            self.__class__.__name__, self.state_dim, self.act_dim, self.ctl_dim
        )
        return s

    def forward(self, logp_o, a, b, B=None):
        """ 
        Args:
            logp_o (torch.tensor): observation log likelihood [batch_size, state_dim]
            a (torch.tensor): soft action vector [batch_size, ctl_dim, act_dim]
            b (torch.tensor): belief [batch_size, state_dim]
            B (torch.tensor, optional): transition parameters 
                [batch_size, act_dim, state_dim, state_dim]

        Returns:
            b_t(torch.tensor): next belief [batch_size, state_dim]
        """
        B_a = self.lmdp(a)
        logp_s = torch.log(torch.sum(b.unsqueeze(-1) * (B_a), dim=-2) + 1e-6)
        b_t = torch.softmax(logp_o + logp_s, dim=-1)
        return b_t