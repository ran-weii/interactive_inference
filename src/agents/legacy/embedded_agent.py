
import torch
import torch.nn as nn
from src.distributions.legacy.models import ConditionalDistribution
from src.distributions.legacy.embedded_models import EmbeddedHiddenMarkovModel
from src.distributions.legacy.factored_models import FactoredConditionalDistribution
from src.agents.legacy.reward import ExpectedFreeEnergy
from src.agents.planners import QMDP
from src.distributions.utils import poisson_pdf, rectify

def message_passing(R, B, H):
    """ Bayesian network style message passing
    Args:
        R (torch.tensor): reward matrix [batch_size, state_dim]
        B (torch.tensor): transition matrix [batch_size, state_dim, state_dim]
        H (int): planning horizon
        
    Returns:
        V (torch.tensor): value [batch_size, H, state_dim]
    """
    V = [torch.empty(0)] * H
    V[0] = R
    for h in range(H-1):
        V_next = torch.sum(B * V[h].view(1, -1), dim=-1)
        V[h+1] = R + V_next
    V = torch.stack(V).transpose(0, 1)
    return V


class MessagePassingPlanner(QMDP):
    def __init__(self, hmm, obs_model, rwd_model, H):
        super().__init__(hmm, obs_model, rwd_model, H)
    
    def forward(self, b):
        Q_b = torch.sum(self._Q * b.unsqueeze(-2).unsqueeze(-2), dim=-1)
        pi = torch.softmax(Q_b, dim=-1)
        return pi

    def plan(self, theta=None):
        """
        Returns:
            Q (torch.tensor): action value function [ctl_dim, act_dim, state_dim]
        """
        B = self.hmm.lmdp.passive_dynamics
        h = poisson_pdf(rectify(self.tau), self.H)
        
        R = self.rwd_model(B, B)
        V = message_passing(R, B, self.H)
        V = torch.sum(h * V, dim=1)

        Q = [torch.empty(0)] * self.hmm.ctl_dim
        for i in range(self.hmm.ctl_dim):
            a = torch.zeros(self.hmm.act_dim, self.hmm.ctl_dim, self.hmm.act_dim)
            a[:, i] = torch.eye(self.hmm.act_dim)
            B_a = self.hmm.lmdp(a)
            Q[i] = torch.sum(B_a * V.view(1, 1, -1), dim=-1)
            """ new theoretically justified implementation """
            # log_B_a = torch.log(self.hmm.lmdp(a) + 1e-6)
            # Q[i] = torch.logsumexp(V.view(1, 1, -1) + log_B_a, dim=-1)
        Q = torch.stack(Q)
        self._Q = Q
        return Q


class EmbeddedActiveInference(nn.Module):
    def __init__(
        self, state_dim, act_dim, obs_dim, ctl_dim, H, 
        obs_model="gmm", obs_dist="mvn", obs_cov="full", 
        ctl_model="gmm", ctl_dist="mvn", ctl_cov="full", 
        planner="qmdp", tau=1, hidden_dim=64, num_hidden=2, activation="relu"
        ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ctl_dim = ctl_dim
        self.H = H

        self.hmm = EmbeddedHiddenMarkovModel(state_dim, act_dim, ctl_dim)
        self.obs_model = ConditionalDistribution(obs_dim, state_dim, obs_dist, obs_cov, batch_norm=True)
        self.ctl_model = FactoredConditionalDistribution(ctl_dim, act_dim, ctl_dist, ctl_cov, batch_norm=True)
        self.rwd_model = ExpectedFreeEnergy(self.hmm, self.obs_model)
        self.planner = MessagePassingPlanner(self.hmm, self.obs_model, self.rwd_model, H)
    
        self.reset()
    
    def reset(self):
        """ reset internal states for online inference """
        self._b = None
        self._a = None

    def get_default_parameters(self):
        theta = {
            "A": None, "B": None, "C": None, 
            "D": self.hmm.D, "F": None, "tau": None
        }
        return theta

    def forward(self, o, u, h=None, theta=None, inference=False):
        T = len(u)
        if theta is None:
            theta = self.get_default_parameters()

        b = [torch.empty(0)] * (T + 1)
        a = [torch.empty(0)] * (T + 1)
        if h is None:
            b[0], a[0] = self.init_hidden(o, theta)
        else:
            b[0], a[0] = h
        
        logp_u = self.ctl_model.log_prob(u, theta["F"])
        logp_o = self.obs_model.log_prob(o, theta["A"])
        for t in range(T):
            p_a = self.ctl_model.infer(a[t], u[t], logp_x=logp_u[t], params=theta["F"])
            b[t+1] = self.hmm(logp_o[t], p_a, b[t], B=theta["B"])
            a[t+1] = self.planner(b[t+1])
        a = torch.stack(a)
        b = torch.stack(b)
        
        if not inference:
            logp_pi = self.ctl_model.mixture_log_prob(a[:-1], u, theta["F"]).sum(-1)
            logp_obs = self.obs_model.mixture_log_prob(b[1:], o, theta["A"])
            return logp_pi, logp_obs, b
        else:
            return b, a
    
    def init_hidden(self, o, theta):
        if theta is None:
            theta = self.get_default_parameters() 
        self.planner.plan(theta)

        b = torch.softmax(theta["D"], dim=-1)
        b = b * torch.ones(o.shape[-2], self.state_dim)
        a = self.planner(b)
        return b, a

    def choose_action(self, o, u, batch=False, theta=None, num_samples=None):
        """ 
        Args:
            o (torch.tensor): observation sequence [T, batch_size, obs_dim]
            u (torch.tensor): control sequence [T, batch_size, ctl_dim]
            batch (bool, optional): whether to perform batch inference, Defaults to False
            theta (dict, optional): agent parameters dict. Defaults to None.
            num_samples (int, optional): number of samples for ancestral sampling. 
                Use bayesian averaging if None. Defaulst to None.
            
        Returns:
            u: predicted control [T, batch_size, ctl_dim]
        """
        if batch:
            b, a = self.forward(o, u, theta=theta, inference=True)
            b, a = b[:-1], a[:-1]
        else:
            o, u = o.unsqueeze(0), u.unsqueeze(0)
            if self._b is None: # initial step
                b, a = self.init_hidden(o, theta=theta)
            else:
                h = [self._b, self._a]
                b, a = self.forward(o, u, h=h, theta=theta, inference=True)
                b, a = b[1], a[1]
            self._b, self._a = b, a

        F = None if theta is None else theta["F"]
        if num_samples is None:
            u_pred = self.ctl_model.bayesian_average(a, F)
        else:
            u_pred = self.ctl_model.ancestral_sample(a, num_samples, F)
        return u_pred