import torch
import torch.nn as nn
from src.agents.core import AbstractAgent
from src.agents.planners import value_iteration
from src.distributions.utils import kl_divergence, poisson_pdf, rectify

""" add a beta similar to beta vae """
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

        # self.gamma = nn.Parameter(torch.randn(1, 1))
        # nn.init.uniform_(self.gamma, a=-1, b=1)
        
        self.reset()
    
    def __repr__(self):
        s = "{}(h={})".format(self.__class__.__name__, self.horizon)
        return s

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
            s_next = transition_matrix
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
        
        # apply horizon first
        # gamma = rectify(self.gamma)
        # h = poisson_pdf(rectify(self.tau), self.horizon).unsqueeze(-1)
        # b_ = b.unsqueeze(-2).unsqueeze(-2)
        # q = torch.sum(self._q * b_, dim=-1) # [batch_size, horizon, act_dim]
        # q = torch.sum(h * q, dim=-2)
        # policy = torch.softmax(q, dim=-1) 
        return policy


class VINAgent(AbstractAgent):
    """ Value iteraction network agent with ContinuousGaussianHMM dynamics model"""
    def __init__(self, dynamics_model, horizon, rwd_model="efe"):
        super().__init__()
        self.state_dim = dynamics_model.state_dim
        self.act_dim = dynamics_model.act_dim
        self.obs_dim = dynamics_model.obs_dim
        self.ctl_dim = dynamics_model.ctl_dim
        
        self.hmm = dynamics_model
        self.planner = EFEPlanner(self.state_dim, horizon)
    
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
        self._b = torch.ones(1, self.state_dim)
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
            a = torch.eye(self.act_dim).unsqueeze(0).to(self.device)
            transition_matrix = self.hmm.get_transition_matrix(a)
            entropy = self.hmm.obs_entropy()         
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
        
        logp_o = self.hmm.obs_log_prob(o) # supplying this increase test likelihood
        logp_u = self.hmm.ctl_log_prob(u) # supplying this increase test likelihood
        alpha_b = [torch.empty(0)] * (T + 1)
        alpha_b[0] = torch.ones(batch_size, self.state_dim).to(self.device) # filler initial belief
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
    
    def predict(self, x, u, prior=True, inference=True, sample_method="ace", num_samples=1):
        """ Predict observations and controls

        Args:
            x (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            a (torch.tensor): action sequence. size=[T, batch_size, ctl_dim]
            prior (bool, optional): whether to use prior predictive. 
                If false use posterior predictive. Default=True
            inference (bool, optional): inference mode return likelihood. 
                If false return samples. Default=True
            sample_method (str, optional): sampling method. choices=["ace", "bma"], Default="ace"
            num_samples (int, optional): number of samples to return. Default=1

        Returns:
            logp_o (torch.tensor): observation likelihood. size=[T, batch_size]
            logp_u (torch.tensor): control likelihood. size=[T-1, batch_size]
            x_sample (torch.tensor): sampled observation sequence. size=[num_samples, T, batch_size, obs_dim]
            u_sample (torch.tensor): sampled control sequence. size=[num_samples, T-1, batch_size, ctl_dim]
            alpha_b (torch.tensor): state forward messages. size=[T, batch_size, state_dim]
            alpha_a (torch.tensor): action forward messages. size=[T-1, batch_size, act_dim]
        """
        alpha_b, alpha_a = self.forward(x, u)
        
        batch_size = x.shape[1]
        if prior:
            z0 = self.hmm.transition_model.get_initial_state()
            z0 = z0 * torch.ones(batch_size, 1).to(self.device)
            z = torch.cat([z0.unsqueeze(0), alpha_b[:-1]], dim=0)
        else:
            z = alpha_b
        
        if not inference:
            logp_o = self.hmm.obs_mixture_log_prob(z, x)
            logp_u = self.hmm.ctl_mixture_log_prob(alpha_a, u[:-1])
            return logp_o, logp_u
        else:
            # model average prediction
            if sample_method == "ace":
                x_sample = self.hmm.obs_ancestral_sample(z, num_samples)
                u_sample = self.hmm.ctl_ancestral_sample(alpha_a, num_samples)
            else:
                x_sample = self.hmm.obs_bayesian_average(z)
                u_sample = self.hmm.ctl_bayesian_average(alpha_a)
            
            return x_sample, u_sample, alpha_b, alpha_a

    def act_loss(self, o, u, mask, forward_out):
        """ Compute action loss 
        
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            mask (torch.tensor): binary mask sequence. size=[T, batch_size]
            forward_out (list): outputs of forward method

        Returns:
            loss (torch.tensor): action loss. size=[batch_size]
            stats (dict): action loss stats
        """
        _, alpha_a = forward_out
        logp_u = self.hmm.ctl_mixture_log_prob(alpha_a, u)
        loss = -torch.sum(logp_u * mask, dim=0) / mask.sum(0)

        # compute stats
        nan_mask = mask.clone()
        nan_mask[nan_mask == 0] = torch.nan
        logp_u_mean = -torch.nanmean((nan_mask * logp_u)).cpu().data
        stats = {"loss_u": logp_u_mean}
        return loss, stats
    
    def obs_loss(self, o, u, mask, forward_out):
        """ Compute observation loss 
        
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            mask (torch.tensor): binary mask tensor. size=[T, batch_size]
            forward_out (list): outputs of forward method

        Returns:
            loss (torch.tensor): observation loss. size=[batch_size]
            stats (dict): observation loss stats
        """
        alpha_b, _ = forward_out
        logp_o = self.hmm.obs_mixture_log_prob(alpha_b, o)
        loss = -torch.sum(logp_o * mask, dim=0) / mask.sum(0)

        # compute stats
        nan_mask = mask.clone()
        nan_mask[nan_mask == 0] = torch.nan
        logp_o_mean = -torch.nanmean((nan_mask * logp_o)).cpu().data
        stats = {"loss_o": logp_o_mean}
        return loss, stats

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