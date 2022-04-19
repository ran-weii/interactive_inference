import math
import torch
import torch.nn as nn
from src.agents.models import MLP
from src.distributions.utils import poisson_pdf, kl_divergence

def value_iteration(R, B, H):
    """
    Args:
        R (torch.tensor): reward matrix [batch_size, act_dim, state_dim]
        B (torch.tensor): transition matrix [batch_size, act_dim, state_dim, state_dim]
        H (int): planning horizon
        
    Returns:
        Q (torch.tensor): Q value [batch_size, H, act_dim, state_dim]
    """
    Q = [torch.empty(0)] * H
    Q[0] = R
    for h in range(H-1):
        V_next = torch.logsumexp(Q[h], dim=-2, keepdim=True).unsqueeze(-2)
        Q_next = torch.sum(B * V_next, dim=-1)
        Q[h+1] = R + Q_next
    Q = torch.stack(Q).transpose(0, 1)
    return Q


class QMDP(nn.Module):
    """ Finite horizon QMDP planner """
    def __init__(self, hmm, obs_model, rwd_model, H):
        super().__init__()
        self.H = H
        self.hmm = hmm
        self.obs_model = obs_model
        self.rwd_model = rwd_model

        self.tau = nn.Parameter(torch.randn(1, 1), requires_grad=True)
        nn.init.uniform_(self.tau, a=-1, b=1)
        
        self.reset()
    
    def __repr__(self):
        s = "{}(h={})".format(self.__class__.__name__, self.H)
        return s

    def reset(self):
        self._Q = None

    def get_default_params(self):
        theta = {
            "A": None, "B": self.hmm.B, "C": self.C, 
            "D": None, "F": None, "tau": self.tau
        }
        return theta
    
    """ TODO: implement this for model explaination """
    def horizon_dist(self, theta=None):
        return 

    def forward(self, b):
        a = torch.softmax(torch.sum(b.unsqueeze(-2) * self._Q, dim=-1), dim=-1)
        return a

    def plan(self, theta=None):
        R = self.rwd_model.state_reward(theta)
        if theta == None:
            theta = self.get_default_params()
        
        B = torch.softmax(self.hmm.transform_parameters(theta["B"]), dim=-1)
        h = theta["tau"].clip(math.log(1e-6), math.log(1e6)).exp()
        h = poisson_pdf(h, self.H)[..., None, None]
        
        Q = value_iteration(R, B, self.H)
        Q = torch.sum(h * Q, dim=-3)
        self._Q = Q
        return Q


class NNPlanner(nn.Module):
    def __init__(
        self, state_dim, act_dim, hidden_dim, num_hidden, activation,
        hmm, obs_model, C
        ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim

        self.mlp = MLP(
            state_dim, act_dim, hidden_dim, num_hidden, activation
        )
        self.hmm = hmm
        self.obs_model = obs_model
        self.C = C

    def forward(self, b):
        a = torch.softmax(self.mlp(b), dim=-1)
        return a

    def td_loss(self, b):
        b = b.view(-1, self.state_dim)
        s_next, b_next = self.sample_next_belief(b)
        
        # compute efe reward
        ent = self.obs_model.entropy().unsqueeze(0)
        C = torch.softmax(self.C, dim=-1).unsqueeze(0)
        ekl = kl_divergence(b_next, C)
        eh = torch.sum(s_next * ent, dim=-1)
        r = -ekl - eh
        # print(ent.shape)
        print("b", b_next.shape)
        # print(C.shape)
        # print(ekl.shape)
        # print(eh.shape)
        print("r", r.shape)

        # compute td error
        Q = self.mlp(b)
        Q_next = self.mlp(b_next)
        V_next = torch.logsumexp(Q_next, dim=-1)
        td_error = 0.5 * (Q - r - V_next).pow(2)
        print(Q.shape, Q_next.shape, V_next.shape)
        print(td_error.shape)
        print(td_error.data.numpy().round(2))
        print(td_error.sum())
        return td_error

    def sample_next_belief(self, b):
        B = torch.softmax(self.hmm.B, dim=-1)
        s_next = torch.sum(B * b.unsqueeze(1).unsqueeze(-1), dim=-1)
        
        # sample next observation
        o_next = self.obs_model.ancestral_sample(s_next, 1).squeeze(0)
        logp_o = self.obs_model.log_prob(o_next)
        a_next = torch.eye(self.act_dim).unsqueeze(0)
        b_next = self.hmm(logp_o, a_next, b.unsqueeze(-2))
        
        # print(o_next.shape, logp_o.shape, a_next.shape, b.shape)
        # print(b_next.shape)
        # print(b[0].data.numpy().round(3))
        # print(b_next[0, 0].data.numpy().round(3))
        return s_next, b_next