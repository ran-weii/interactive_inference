
import torch
import torch.nn as nn
from src.distributions.legacy.models import ConditionalDistribution
from src.distributions.legacy.factored_models import (
    FactoredHiddenMarkovModel, FactoredConditionalDistribution)
from src.agents.legacy.reward import ExpectedFreeEnergy
from src.agents.planners import QMDP
from src.agents.legacy.active_inference import ActiveInference
from src.distributions.utils import poisson_pdf, rectify, softmax
    
def factored_value_iteration(R, B, H):
    """ Value function factorization for multi-agent planning
    Args:
        R (torch.tensor): reward matrix [batch_size, act_dim * num_agents, state_dim]
        B (torch.tensor): transition matrix [batch_size, act_dim * num_agents, state_dim, state_dim]
        H (int): planning horizon
        
    Returns:
        Q (torch.tensor): Q value [batch_size, H, num_agents, act_dim, state_dim]
    """
    num_agents = len(B.shape) - 3
    agent_dims = [-i - 2 for i in range(num_agents)]
    Q = [torch.empty(0)] * H
    Q[0] = R
    for h in range(H-1):
        V_next = torch.logsumexp(Q[h], dim=agent_dims, keepdim=True).unsqueeze(-2)
        Q_next = torch.sum(B * V_next, dim=-1)
        Q[h+1] = R + Q_next
    Q = torch.stack(Q).transpose(0, 1)
    return Q


class FactoredQMDP(QMDP):
    def __init__(self, hmm, obs_model, rwd_model, H):
        super().__init__(hmm, obs_model, rwd_model, H)
        self.num_agents = len(hmm.B.shape) - 3

    def forward(self, b):
        num_agents = self.num_agents
        agent_index = -torch.arange(1, num_agents+1).flip(-1)
        b = b.view(list(b.shape) + [1]*num_agents).transpose(-1, -num_agents-1)
        Q = torch.sum(b * self._Q, dim=-1)
        pi = softmax(Q, dims=list(agent_index))
        
        # agent marginal actions
        pi_a = [torch.empty(0)] * num_agents
        for i in range(num_agents):
            sum_dims = agent_index.roll(-i)[1:]
            pi_a[i] = pi.sum(list(sum_dims))
        pi_a = torch.stack(pi_a).transpose(0, 1)
        return pi_a
    
    def plan(self, theta=None):
        if theta is None:
            theta = self.get_default_params()
        
        num_agents = self.num_agents
        B = torch.softmax(self.hmm.transform_parameters(theta["B"]), dim=-1)
        h = poisson_pdf(rectify(theta["tau"]), self.H).view([1, -1] + [1] * num_agents + [1])
        
        R = self.rwd_model(B, B, A=theta["A"], C=theta["C"])
        Q = factored_value_iteration(R, B, self.H)
        Q = torch.sum(h * Q, dim=1)
        self._Q = Q
        return Q

    
class FactoredActiveInference(ActiveInference):
    def __init__(
        self, state_dim, act_dim, obs_dim, ctl_dim, H, 
        obs_model="gmm", obs_dist="mvn", obs_cov="full", 
        ctl_model="gmm", ctl_dist="mvn", ctl_cov="full", 
        planner="qmdp", tau=1, hidden_dim=64, num_hidden=2, activation="relu"
        ):
        super().__init__(
            state_dim, act_dim, obs_dim, ctl_dim, H, 
            obs_model, obs_dist, obs_cov, 
            ctl_model, ctl_dist, ctl_cov, 
            planner, tau, hidden_dim, num_hidden, activation
        )
        self.hmm = FactoredHiddenMarkovModel(state_dim, act_dim, ctl_dim)
        self.ctl_model = FactoredConditionalDistribution(ctl_dim, act_dim)
        self.planner = FactoredQMDP(self.hmm, self.obs_model, self.rwd_model, H)

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
        return u_pred.squeeze(-3)