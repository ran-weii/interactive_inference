import torch
import torch.nn as nn
from src.distributions.models import (
    HiddenMarkovModel, ConditionalDistribution)
from src.distributions.utils import poisson_pdf
from src.agents.planners import value_iteration

class ActiveInference(nn.Module):
    def __init__(
        self, state_dim, act_dim, obs_dim, ctl_dim, H, 
        obs_dist="mvn", obs_cov="full", ctl_dist="mvn", ctl_cov="full"
        ):
        super(). __init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ctl_dim = ctl_dim
        self.H = H
        
        self.C = nn.Parameter(torch.randn(1, state_dim), requires_grad=True)
        self.tau = nn.Parameter(torch.randn(1, 1), requires_grad=True)
        self.hmm = HiddenMarkovModel(state_dim, act_dim)
        self.obs_model = ConditionalDistribution(obs_dim, state_dim, obs_dist, obs_cov)
        self.ctl_model = ConditionalDistribution(ctl_dim, act_dim, ctl_dist, ctl_cov)
        
        nn.init.xavier_normal_(self.C, gain=1.)
        nn.init.xavier_normal_(self.tau, gain=1.)
    
    def get_default_parameters(self):
        theta = {
            "A": None, "B": self.hmm.B, "C": self.C, 
            "D": self.hmm.D, "F": None, "tau": self.tau
        }
        return theta
    
    def forward(self, o, u, theta=None, inference=False):
        """
        Args:
            o (torch.tensor): observation sequence [T, batch_size, obs_dim]
            u (torch.tensor): control sequence [T, batch_size, ctl_dim]
            theta (dict, optional): agent parameters dict. Defaults to None.
            inference (bool, optional): whether in inference model. Defaults to False
            
        Returns:
            logp_pi (torch.tensor): predicted control likelihood [T, batch_size]
            logp_obs (torch.tensor): predicted observation likelihood [T, batch_size]
        """
        if theta is None:
            theta = self.get_default_parameters()
            
        T = len(o)
        # encode observation and control
        logp_o = self.obs_model.log_prob(o.unsqueeze(-2), theta["A"])
        logp_u = self.ctl_model.log_prob(u.unsqueeze(-2), theta["F"])
        p_a = torch.softmax(logp_u, dim=-1)
        
        # belief update
        b = [torch.empty(0)] * (T + 1)
        b[0] = torch.softmax(theta["D"], dim=-1) * torch.ones_like(logp_o[0])
        for t in range(T):
            b[t+1] = self.hmm(logp_o[t], p_a[t], b[t], B=theta["B"])
        b = torch.stack(b)
        
        # decode actions
        Q = self.plan(theta)
        G = torch.sum(b[:-1].unsqueeze(-2) * Q.unsqueeze(0), dim=-1)
        
        if not inference:
            logp_a = torch.softmax(G, dim=-1).log()
            logp_pi = torch.logsumexp(logp_a + logp_u, dim=-1)
            
            logp_b = torch.log(b[1:] + 1e-6)
            logp_obs = torch.logsumexp(logp_b * logp_o, dim=-1)
            return logp_pi, logp_obs
        else:
            return G, b
    
    def get_reward(self, theta):
        obs_entropy = self.obs_model.entropy(theta["A"]).unsqueeze(-2)
        
        C = torch.softmax(theta["C"], dim=-1).unsqueeze(-2).unsqueeze(-2)
        B = self.hmm.transform_parameters(theta["B"])
        B = torch.softmax(B, dim=-1)
        kl = torch.sum(B * B.log() - C.log(), dim=-1)
        
        R = -kl - obs_entropy
        return R
    
    def plan(self, theta):
        R = self.get_reward(theta)
        B = self.hmm.transform_parameters(theta["B"])
        B = torch.softmax(B, dim=-1)
        h = poisson_pdf(theta["tau"].exp(), self.H).unsqueeze(-1).unsqueeze(-1)
        
        Q = value_iteration(R, B, self.H)
        Q = torch.sum(h * Q, dim=-3)
        return Q
    
    def choose_action(self, o, u, theta=None):
        """ Choose action via Bayesian model averaging

        Args:
            o (torch.tensor): observation sequence [T, batch_size, obs_dim]
            u (torch.tensor): control sequence [T, batch_size, ctl_dim]
            theta (dict, optional): agent parameters dict. Defaults to None.

        Returns:
            u: predicted control [T, batch_size, ctl_dim]
        """
        G, b = self.forward(o, u, theta=theta, inference=True)
        p_a = torch.softmax(G, dim=-1)
        
        # bayesian model averaging
        if theta is None:
            mu_u = self.ctl_model.mean()
        else:
            mu_u = self.ctl_model.mean(theta["F"])
        
        u = torch.sum(p_a.unsqueeze(-1) * mu_u.unsqueeze(0), dim=-2)
        return u