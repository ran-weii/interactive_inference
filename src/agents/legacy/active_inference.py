import torch
import torch.nn as nn
from src.distributions.legacy.models import (
    HiddenMarkovModel, ConditionalDistribution, GeneralizedLinearModel)
from src.agents.reward import ExpectedFreeEnergy, GeneralizedFreeEnergy
from src.agents.planners import QMDP, MCVI

class ActiveInference(nn.Module):
    def __init__(
        self, state_dim, act_dim, obs_dim, ctl_dim, H, 
        obs_model="gmm", obs_dist="mvn", obs_cov="full", 
        ctl_model="gmm", ctl_dist="mvn", ctl_cov="full",
        rwd_model="efe", hmm_rank=0, planner="qmdp", tau=1, hidden_dim=64, 
        num_hidden=2, activation="relu"
        ):
        super(). __init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ctl_dim = ctl_dim
        self.H = H
        
        self.hmm = HiddenMarkovModel(state_dim, act_dim, rank=hmm_rank)
        if obs_model == "gmm":
            self.obs_model = ConditionalDistribution(obs_dim, state_dim, obs_dist, obs_cov, batch_norm=True)
        else:
            raise NotImplementedError
        if ctl_model == "gmm":
            self.ctl_model = ConditionalDistribution(ctl_dim, act_dim, ctl_dist, ctl_cov, batch_norm=True)
        elif ctl_model == "glm":
            self.ctl_model = GeneralizedLinearModel(ctl_dim, act_dim, ctl_dist, ctl_cov, batch_norm=True)
        else:
            raise NotImplementedError
        
        if rwd_model == "efe":
            self.rwd_model = ExpectedFreeEnergy(self.hmm, self.obs_model)
        elif rwd_model == "gfe":
            self.rwd_model = GeneralizedFreeEnergy(self.hmm, self.obs_model)

        if planner == "qmdp":
            self.planner = QMDP(self.hmm, self.obs_model, self.rwd_model, self.H)
        elif planner == "mcvi":
            self.planner = MCVI(
                self.hmm, self.obs_model, self.rwd_model, tau,
                hidden_dim=hidden_dim, num_hidden=num_hidden, activation=activation
            )

        self.reset()
    
    def reset(self):
        """ reset internal states for online inference """
        self._b = None
        self._a = None
    
    """ TODO: find better ways to feed parameters """
    def get_default_parameters(self):
        # theta = {
        #     "A": None, "B": self.hmm.B, "C": None, 
        #     "D": self.hmm.D, "F": None, "tau": self.planner.tau
        # }
        theta = {
            "A": None, "B": self.hmm.get_default_parameters(), "C": None, 
            "D": self.hmm.D, "F": None, "tau": self.planner.tau
        }
        return theta
    
    def forward(self, o, u, h=None, theta=None, inference=False):
        """
        Args:
            o (torch.tensor): observation sequence [T, batch_size, obs_dim]
            u (torch.tensor): control sequence [T, batch_size, ctl_dim]
            h (list of torch.tensor, optional): hidden states [b, a] 
                used for online inference. Defaults to None.
            theta (dict, optional): agent parameters dict. Defaults to None.
            inference (bool, optional): whether in inference model. Defaults to False
            
        Returns:
            logp_pi (torch.tensor): predicted control likelihood [T, batch_size]
            logp_obs (torch.tensor): predicted observation likelihood [T, batch_size]
        """
        T = len(u)
        if theta is None:
            theta = self.get_default_parameters()

        b = [torch.empty(0)] * (T + 1)
        a = [torch.empty(0)] * (T + 1)
        if h is None:
            b[0], a[0] = self.init_hidden(o, theta)
        else:
            b[0], a[0] = h
        
        logp_o = self.obs_model.log_prob(o, theta["A"])
        logp_u = self.ctl_model.log_prob(u, theta["F"])
        for t in range(T):
            p_a = self.ctl_model.infer(a[t], u[t], logp_x=logp_u[t], params=theta["F"])
            b[t+1] = self.hmm(logp_o[t], p_a, b[t], B=theta["B"])
            a[t+1] = self.planner(b[t+1])
        a = torch.stack(a)
        b = torch.stack(b)
        
        if not inference:
            logp_pi = self.ctl_model.mixture_log_prob(a[:-1], u, theta["F"])
            
            logp_b = torch.log(b[1:] + 1e-6)
            logp_obs = torch.logsumexp(logp_b + logp_o, dim=-1)
            """ TODO adding a output for reverse kl training """
            return logp_pi, logp_obs, b, a 
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

    def choose_action(self, o, u, batch=False, theta=None, sample_method="ace", num_samples=None):
        """ 
        Args:
            o (torch.tensor): observation sequence [T, batch_size, obs_dim]
            u (torch.tensor): control sequence [T, batch_size, ctl_dim]
            batch (bool, optional): whether to perform batch inference, Defaults to False
            theta (dict, optional): agent parameters dict. Defaults to None.
            sample_method (str, optional): sample method. Choices=["ace", "acm", "bma"]
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
        if sample_method == "bma":
            u_pred = self.ctl_model.bayesian_average(a, F)
        else:
            sample_mean = True if sample_method == "acm" else False
            u_pred = self.ctl_model.ancestral_sample(
                a, num_samples=num_samples, sample_mean=sample_mean, params=F
            )
        return u_pred.squeeze(-3)