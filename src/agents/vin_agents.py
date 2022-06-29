import torch
import torch.nn as nn
from src.agents.baseline import AbstractAgent
from src.distributions.hmm import ContinuousGaussianHMM
from src.agents.planners import value_iteration
from src.distributions.utils import kl_divergence, poisson_pdf, rectify

class EFEPlanner(nn.Module):
    """ Expected free energy planner """
    def __init__(self, state_dim, horizon):
        super().__init__()
        self.horizon = horizon
        self.eps = 1e-6
        
        self.c = nn.Parameter(torch.randn(1, state_dim))
        self.tau = nn.Parameter(torch.randn(1, 1))
        nn.init.xavier_normal(self.c, gain=1.)
        nn.init.uniform_(self.tau, a=-1, b=1)
        
        self.reset()
    
    def reset(self):
        self._q = None
    
    def compute_reward(self, s_next, entropy):
        """ Compute single stage reward
        
        Args:
            s_next (torch.tensor): predictive state distribution. size=[act_dim, state_dim, state_dim]
            entropy (torch.tensor): observation entropy. size=[1, state_dim]
        
        Returns:
            r (torch.tensor): reward. size=[..., state_dim]
        """
        c = torch.softmax(self.c + self.eps, dim=-1)
        kl = kl_divergence(s_next, c)
        eh = torch.sum(s_next * entropy, dim=-1)
        r = -kl - eh
        return r
    
    def plan(self, transition_matrix, entropy):
        """ Compute q value function 
        
        Args:
            transition_matrix (torch.tensor): transition matrix. 
                size=[batch_size, act_dim, state_dim, state_dim]
            entropy (torch.tensor): observation entropy. size=[batch_size, state_dim]
        """
        if self._q is None:
            s_next = transition_matrix.clone()
            reward = self.compute_reward(s_next, entropy)
            self._q = value_iteration(reward, transition_matrix, self.horizon) # [batch_size, horizon, act_dim, state_dim]
    
    def policy(self, b):
        """ Compute belief action policy
        
        Args:
            b (torch.tensor): belief distribution. size=[batch_size, state_dim]
        
        Returns:
            policy (torch.tensor): action distribution. size=[batch_size, act_dim]
        """
        b_ = b.unsqueeze(-2).unsqueeze(-2)
        q = torch.sum(self._q * b_, dim=-1)
        policy = torch.softmax(q, dim=-1) # [batch_size, horizon, act_dim]
        
        h = poisson_pdf(rectify(self.tau), self.horizon).unsqueeze(-1)
        policy = torch.sum(h * policy, dim=-2)
        return policy


class VINAgent(AbstractAgent):
    """ Value iteraction network agent with ContinuousGaussianHMM dynamics model"""
    def __init__(
        self, state_dim, act_dim, obs_dim, ctl_dim, horizon,
        hmm_rank=0, obs_cov="full", ctl_cov="full", rwd_model="efe"
        ):
        super().__init__(state_dim, act_dim, obs_dim, ctl_dim, horizon)
        
        self.hmm = ContinuousGaussianHMM(
            state_dim, act_dim, obs_dim, ctl_dim, rank=hmm_rank, 
            obs_cov=obs_cov, ctl_cov=ctl_cov
        )
        self.planner = EFEPlanner(state_dim, horizon)
    
    def load_dynamics_model(self, state_dict, requires_grad):
        """ Load pretrained dynamics model 
        
        Args:
            state_dict (dict): dynamics model state dict
            requires_grad (bool): whether to require grad on dynamics model
        """
        self.hmm.load_state_dict(state_dict)
        for n, p in self.hmm.named_parameters():
            p.requires_grad = requires_grad
    
    def reset(self):
        """ Reset internal states for online inference """
        self._b = None
        self._a = None # previous action distribution
        self.planner.reset() # reset q value function
    
    def plan(self, b):
        """ Compute a single time step action distribution for belief b 
        
        Args:
            b (torch.tensor): belief distribution. size=[batch_size, state_dim]

        Returns:
            a (torch.tensor): action distribution. size=[batch_size, act_dim]
        """
        if self.planner._q is None:
            a = torch.eye(self.act_dim).unsqueeze(0)
            transition_matrix = self.hmm.get_transition_matrix(a)
            entropy = self.hmm.obs_model.entropy()            
            self.planner.plan(transition_matrix, entropy)
        
        a = self.planner.policy(b)
        return a
    
    def alpha(self, b, o, u=None, a=None, logp_o=None, logp_u=None):
        """ Update belief and compute the action distribution for a single time step
        
        Args:
            b (torch.tensor): prior belief. size=[batch_size, state_dim]
            x (torch.tensor): observation vector. size=[batch_size, obs_dim]
            u (torch.tensor, None, optional): control vector. size=[batch_size, ctl_dim]. 
                None for initial state. Default=None
            a (torch.tensor, None, optional): action prior, to be supplied by a planner.
                size=[batch_size, act_dim]. Default=None
            logp_x (torch.tensor, None, optional): observation likelihood. Supplied during training. 
                size=[batch_size, state_dim]
            logp_u (torch.tensor, None, optional): control likelihood. Supplied during training. 
                size=[batch_size, act_dim]

        Returns:
            b_t (torch.tensor): state posterior distribution. size=[batch_size, state_dim]
            a_t (torch.tensor): action predictive distribution. size=[batch_size, act_dim]
        """
        b_t, _ = self.hmm.alpha(
            b, o, u=u, a=a, logp_x=logp_o, logp_u=logp_u
        )
        a_t = self.plan(b_t)
        return b_t, a_t
    
    def forward(self, o, u):
        """ Forward algorithm
        
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
        
        Returns:
            alpha_b (torch.tensor): state belief distributions. size=[T, batch_size, state_dim]
            alpha_a (torch.tensor): action predictive distributions. size=[T, batch_size, act_dim]
        """
        batch_size = o.shape[1]
        T = o.shape[0]
        
        logp_o = self.hmm.obs_model.log_prob(o) # supplying this increase test likelihood
        logp_u = self.hmm.ctl_model.log_prob(u) # supplying this increase test likelihood
        alpha_b = [torch.empty(0)] * (T + 1)
        alpha_b[0] = torch.ones(batch_size, self.state_dim) # filler initial belief
        alpha_a = [torch.empty(0)] * (T)
        for t in range(T):
            o_t = o[t]
            u_t = None if t == 0 else u[t-1]
            a_t = None if t == 0 else alpha_a[t-1]
            logp_o_t = logp_o[t]
            logp_u_t = None if t == 0 else logp_u[t-1]
            alpha_b[t+1], alpha_a[t] = self.alpha(
                alpha_b[t], o_t, u_t, a=a_t, logp_o=logp_o_t, logp_u=logp_u_t
            )
        
        alpha_b = torch.stack(alpha_b)[1:]
        alpha_a = torch.stack(alpha_a)
        return alpha_b, alpha_a
    
    def act_loss(self, o, u, mask, forward_out):
        """ Compute action loss 
        
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            mask (torch.tensor): binary mask sequence. size=[T, batch_size]
            forward_out (list): outputs of forward method

        Returns:
            loss (torch.tensor): action loss. size=[batch_size]
        """
        _, alpha_a = forward_out
        logp_u = self.hmm.ctl_model.mixture_log_prob(alpha_a, u)
        loss = -torch.sum(logp_u * mask, dim=0) / mask.sum(0)
        return loss
    
    def obs_loss(self, o, u, mask, forward_out):
        """ Compute observation loss 
        
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            mask (torch.tensor): binary mask tensor. size=[T, batch_size]
            forward_out (list): outputs of forward method
        """
        alpha_b, _ = forward_out
        logp_o = self.hmm.obs_model.mixture_log_prob(alpha_b, o)
        loss = -torch.sum(logp_o * mask, dim=0) / mask.sum(0)
        return loss

    def choose_action(self, o, u, sample_method="ace", num_samples=1):
        """ Choose action online for a single time step
        
        Args:
            o (torch.tensor): observation sequence. size[batch_size, obs_dim]
            u (torch.tensor): control sequence. size[batch_size, ctl_dim]
            sample_method (str, optional): sampling method. 
                choices=["bma", "ace", "ace"]. Default="ace"
            num_samples (int, optional): number of samples to draw. Default=1
        
        Returns:
            u_sample (torch.tensor): sampled controls. size=[num_samples, batch_size, ctl_dim]
        """
        b_t, a_t = self.alpha(self._b, o, u, self._a)
        
        if sample_method == "bma":
            u_sample = self.hmm.ctl_model.bayesian_average(a_t)
        else:
            sample_mean = True if sample_method == "acm" else False
            u_sample = self.hmm.ctl_model.ancestral_sample(
                a_t.unsqueeze(0), num_samples, sample_mean
            ).squeeze(-3)
        
        self._b, self._a = b_t, a_t
        return u_sample
    
    def choose_action_batch(self, o, u, sample_method="ace", num_samples=1):
        """ Choose action offline for a batch of sequences 
        
        Args:
            o (torch.tensor): observation sequence. size[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size[T, batch_size, ctl_dim]
            sample_method (str, optional): sampling method. 
                choices=["bma", "ace", "ace"]. Default="ace"
            num_samples (int, optional): number of samples to draw. Default=1

        Returns:
            u_sample (torch.tensor): sampled controls. size=[num_samples, T, batch_size, ctl_dim]
        """
        alpha_b, alpha_a = self.forward(o, u)

        if sample_method == "bma":
            u_sample = self.hmm.ctl_model.bayesian_average(alpha_a)
        else:
            sample_mean = True if sample_method == "acm" else False
            u_sample = self.hmm.ctl_model.ancestral_sample(
                alpha_a, num_samples, sample_mean
            )
        return u_sample