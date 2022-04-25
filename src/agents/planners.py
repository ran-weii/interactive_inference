import math
import torch
import torch.nn as nn
from src.agents.models import MLP, PopArt
from src.distributions.utils import poisson_pdf

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


class AbstractPlanner(nn.Module):
    def __init__(self, hmm, obs_model, rwd_model):
        super().__init__()
        self.hmm = hmm
        self.obs_model = obs_model
        self.rwd_model = rwd_model
    
    def reset(self):
        """ Reset hidden state for online inference """
        pass

    def forward(self, b):
        """ Generate action distribution """
        raise NotImplementedError

    def plan(self, theta=None):
        return None

    def loss(self, b):
        return torch.zeros(1)


class QMDP(AbstractPlanner):
    """ Finite horizon QMDP planner """
    def __init__(self, hmm, obs_model, rwd_model, H):
        super().__init__(hmm, obs_model, rwd_model)
        assert isinstance(H, int)
        self.H = H
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
            "A": None, "B": self.hmm.B, "C": None, 
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
        if theta is None:
            theta = self.get_default_params()
        
        B = torch.softmax(self.hmm.transform_parameters(theta["B"]), dim=-1)
        h = theta["tau"].clip(math.log(1e-6), math.log(1e6)).exp()
        h = poisson_pdf(h, self.H)[..., None, None]
        
        R = self.rwd_model(B, B, A=theta["A"], C=theta["C"])
        Q = value_iteration(R, B, self.H)
        Q = torch.sum(h * Q, dim=-3)
        self._Q = Q
        return Q


class MCVI(AbstractPlanner):
    """ Monte Carlo POMDP solvers with neural network value function """
    def __init__(
        self, hmm, obs_model, rwd_model, tau, 
        hidden_dim, num_hidden, activation, popart=False
        ):
        super().__init__(hmm, obs_model, rwd_model)
        assert tau <= 1
        self.tau  = tau
        self.state_dim = hmm.state_dim
        self.act_dim = hmm.act_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.activation = activation
        self.popart = popart

        self.mlp = MLP(
            self.state_dim, self.act_dim, hidden_dim, num_hidden, activation
        )
        if self.popart:
            self.head = PopArt(self.act_dim, self.act_dim)
    
    def __repr__(self):
        s = "{}(tau={}, hidden_dim={}, num_hidden={}, activation={})".format(
            self.__class__.__name__, self.tau, self.hidden_dim, self.num_hidden, self.activation
        )
        return s

    def forward(self, b):
        q, _ = self.q_function(b)
        a = torch.softmax(q, dim=-1)
        return a
    
    def q_function(self, b):
        q = self.mlp(b)
        if self.popart:
            q, q_norm = self.head(q)
        else:
            q_norm = q
        return q, q_norm

    def loss(self, b):
        """ 
        Args:
            b (torch.tensor): tensor of beliefs [batch_size, state_dim]

        Returns:
            td_error (torch.tensor): bellman error [batch_size, act_dim]
        """
        s_next, b_next = self.sample_next_belief(b)
        r = self.rwd_model(s_next, b_next)

        # compute td error
        q_next, _ = self.q_function(b_next)
        v_next = torch.logsumexp(q_next, dim=-1)
        q_target = r + self.tau * v_next
        if self.popart:
            q_target_norm = self.head.normalize(q_target)
        else:
            q_target_norm = q_target

        q, q_norm = self.q_function(b)
        td_error = 0.5 * (q_norm - q_target_norm).pow(2)
        return td_error

    def sample_next_belief(self, b):
        """ 
        Args:
            b (torch.tensor): current belief distribution [batch_size, state_dim]
        
        Returns:
            s_next (torch.tensor): predictive state distribution [batch_size, act_dim, state_dim]
            b_next (torch.tensor): next belief distribution [batch_size, act_dim, state_dim]
        """
        B = torch.softmax(self.hmm.B, dim=-1)
        s_next = torch.sum(B * b.unsqueeze(1).unsqueeze(-1), dim=-1)

        # sample next observation and compute belief update
        o_next = self.obs_model.ancestral_sample(s_next, 1).squeeze(0)
        logp_o = self.obs_model.log_prob(o_next)
        a = torch.eye(self.act_dim).unsqueeze(0)
        b_next = self.hmm(logp_o, a, b.unsqueeze(-2))
        return s_next, b_next