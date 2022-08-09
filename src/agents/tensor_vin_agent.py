import torch
import torch.nn as nn
import torch.nn.functional as F
from src.agents.core import AbstractAgent
from src.agents.tensor_qmdp_layers import TensorQMDPLayer
from src.distributions.mixture_models import ConditionalGaussian
from src.distributions.utils import kl_divergence

from typing import Union, Tuple, Optional
from torch import Tensor

class TensorVINAgent(AbstractAgent):
    """ Value iteraction network agent with 
    conditinal gaussian observation and discrete control models and 
    tensor QMDP hidden layer
    """
    def __init__(
        self, state_dim, act_dim, obs_dim, ctl_dim, rank, horizon,
        obs_cov="full", # use_tanh=False, ctl_lim=None
        ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.ctl_dim = ctl_dim
        self.horizon = horizon
        
        self.rnn = TensorQMDPLayer(state_dim, act_dim, ctl_dim, rank, horizon)
        self.obs_model = ConditionalGaussian(
            obs_dim, state_dim, cov=obs_cov, batch_norm=True, 
            use_tanh=False, limits=None
        )
        # self.ctl_model = ConditionalGaussian(
        #     ctl_dim, act_dim, cov=ctl_cov, batch_norm=True, 
        #     use_tanh=use_tanh, limits=ctl_lim
        # )
        self.c = nn.Parameter(torch.randn(1, state_dim))
        self._pi0 = nn.Parameter(torch.randn(1, ctl_dim, act_dim, state_dim))
        
        nn.init.xavier_normal_(self.c, gain=1.)
        nn.init.xavier_normal_(self._pi0, gain=1.)

        self.parameter_size = [
            self.c.shape, self._pi0.shape,
            self.rnn.b0.shape, self.rnn.u.shape, self.rnn.v.shape, self.rnn.w.shape, self.rnn.tau.shape,
            self.obs_model.mu.shape, self.obs_model.lv.shape, self.obs_model.tl.shape,
            # self.ctl_model.mu.shape, self.ctl_model.lv.shape, self.ctl_model.tl.shape
        ]
    
    def reset(self):
        """ Reset internal states for online inference """
        self._b = None # torch.ones(1, self.state_dim)
        self._a = None # previous action distribution
        self._prev_ctl = None

    @property
    def target_dist(self):
        return torch.softmax(self.c, dim=-1)
    
    @property
    def pi0(self):
        """ Prior policy """
        return torch.softmax(self._pi0, dim=-2)
    
    @property
    def transition(self):
        return self.rnn.transition
    
    @property
    def value(self):
        value = self.rnn.compute_value(self.transition, self.reward)
        return value
    
    @property
    def policy(self):
        """ Optimal planned policy """
        b = torch.eye(self.state_dim)
        pi = self.rnn.plan(b, self.value)
        return pi
    
    @property
    def passive_dynamics(self):
        """ Optimal controlled dynamics """
        policy = self.policy.T.unsqueeze(-1)
        transition = self.transition.squeeze(0)
        return torch.sum(transition * policy, dim=-3)
    
    @property
    def efe(self):
        """ Negative expected free energy """
        entropy = self.obs_model.entropy()
        c = self.target_dist
        kl = kl_divergence(torch.eye(self.state_dim), c)
        return -kl - entropy
    
    @property
    def reward(self):
        """ State action reward """
        transition = self.rnn.transition
        entropy = self.obs_model.entropy()
        
        c = self.target_dist
        kl = kl_divergence(transition, c[:, None, None, None, :])
        eh = torch.sum(transition * entropy[:, None, None, None, :], dim=-1)
        log_pi0 = torch.log(self.pi0 + 1e-6)
        r = -kl - eh + log_pi0
        return r

    def forward(
        self, o: Tensor, u: Union[Tensor, None], 
        hidden: Optional[Union[Tuple[Tensor, Tensor], None]]=None, **kwargs
        ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """ 
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            hidden ([tuple[torch.tensor, torch.tensor], None], optional). initial hidden state.
        
        Returns:
            alpha_b (torch.tensor): state belief distributions. size=[T, batch_size, state_dim]
            alpha_a (torch.tensor): action predictive distributions. size=[T, batch_size, act_dim]
        """
        b, a = None, None
        if hidden is not None:
            b, a = hidden

        logp_o = self.obs_model.log_prob(o)
        # logp_u = None if u is None else self.ctl_model.log_prob(u)
        reward = self.reward
        alpha_b, alpha_a = self.rnn(logp_o, u, reward, b, a)
        return [alpha_b, alpha_a], [alpha_b, alpha_a] # second tuple used in bptt
    
    def act_loss(self, o, u, mask, hidden):
        """ Compute action loss 
        
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            mask (torch.tensor): binary mask sequence. size=[T, batch_size]
            hidden (list): hidden outputs of forward method

        Returns:
            loss (torch.tensor): action loss. size=[batch_size]
            stats (dict): action loss stats
        """
        _, alpha_a = hidden
        u_oh = F.one_hot(u.long(), num_classes=self.act_dim).float()
        logp_u = torch.log(alpha_a + 1e-6)
        logp_u = torch.sum(u_oh * logp_u, dim=-1).sum(-1)
        loss = -torch.sum(logp_u * mask, dim=0) / (mask.sum(0) + 1e-6)

        # compute stats
        nan_mask = mask.clone()
        nan_mask[nan_mask == 0] = torch.nan
        logp_u_mean = -torch.nanmean((nan_mask * logp_u)).cpu().data
        stats = {"loss_u": logp_u_mean}
        return loss, stats
    
    def obs_loss(self, o, u, mask, hidden):
        """ Compute observation loss 
        
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            mask (torch.tensor): binary mask tensor. size=[T, batch_size]
            hidden (list): hidden outputs of forward method

        Returns:
            loss (torch.tensor): observation loss. size=[batch_size]
            stats (dict): observation loss stats
        """
        alpha_b, _ = hidden
        logp_o = self.obs_model.mixture_log_prob(alpha_b, o)
        loss = -torch.sum(logp_o * mask, dim=0) / (mask.sum(0) + 1e-6)

        # compute stats
        nan_mask = mask.clone()
        nan_mask[nan_mask == 0] = torch.nan
        logp_o_mean = -torch.nanmean((nan_mask * logp_o)).cpu().data
        stats = {"loss_o": logp_o_mean}
        return loss, stats
    
    def choose_action(self, o, sample_method="ace", num_samples=1):
        """ Choose action online for a single time step
        
        Args:
            o (torch.tensor): observation sequence. size[batch_size, obs_dim]
            u (torch.tensor): control sequence. size[batch_size, ctl_dim]
            sample_method (str, optional): sampling method. 
                choices=["bma", "ace", "ace"]. Default="ace"
            num_samples (int, optional): number of samples to draw. Default=1
        
        Returns:
            u_sample (torch.tensor): sampled controls. size=[num_samples, batch_size, ctl_dim]
            logp (torch.tensor): control log probability. size=[num_samples, batch_size]
        """
        [alpha_b, alpha_a], _ = self.forward(
            o.unsqueeze(0), self._prev_ctl, [self._b, self._a]
        )
        b_t, a_t = alpha_b[0], alpha_a[0]
        
        if sample_method == "bma":
            u_sample = self.ctl_model.bayesian_average(a_t)
        else:
            sample_mean = True if sample_method == "acm" else False
            u_sample = self.ctl_model.ancestral_sample(
                a_t.unsqueeze(0), num_samples, sample_mean
            ).squeeze(-3)
            logp = self.ctl_model.mixture_log_prob(a_t, u_sample)
        
        self._b, self._a = b_t, a_t
        self._prev_ctl = u_sample.sum(0)
        return u_sample, logp
    
    def choose_action_batch(self, o, u, sample_method="ace", num_samples=1, tau=0.1, hard=True, return_hidden=False, **kwargs):
        """ Choose action offline for a batch of sequences 
        
        Args:
            o (torch.tensor): observation sequence. size[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size[T, batch_size, ctl_dim]
            sample_method (str, optional): sampling method. 
                choices=["bma", "ace", "ace"]. Default="ace"
            num_samples (int, optional): number of samples to draw. Default=1
            tau (float, optional): gumbel softmax temperature. Default=0.1
            hard (bool, optional): if hard use straight-through gradient. Default=True
            return_hidden (bool, optional): if true return agent hidden state. Default=False

        Returns:
            u_sample (torch.tensor): sampled controls. size=[num_samples, T, batch_size, ctl_dim]
            logp (torch.tensor): control log probability. size=[num_samples, T, batch_size]
        """
        [alpha_b, alpha_a], hidden = self.forward(o, u)

        if sample_method == "bma":
            u_sample = self.ctl_model.bayesian_average(alpha_a)
        else:
            sample_mean = True if sample_method == "acm" else False
            u_sample = self.ctl_model.ancestral_sample(
                alpha_a, num_samples, sample_mean, tau, hard
            )
            logp = self.ctl_model.mixture_log_prob(alpha_a, u_sample)
        if return_hidden:
            return u_sample, logp, hidden
        else:
            return u_sample, logp

    def predict(self, o, u, sample_method="ace", num_samples=1):
        """ Offline prediction observations and control """
        [alpha_b, alpha_a], _ = self.forward(o, u)

        if sample_method == "bma":
            o_sample = self.obs_model.bayesian_average(alpha_b)
            u_sample = self.ctl_model.bayesian_average(alpha_a)

        else:
            sample_mean = True if sample_method == "acm" else False
            o_sample = self.obs_model.ancestral_sample(
                alpha_b, num_samples, sample_mean, tau=0.1, hard=True
            )
            u_sample = self.ctl_model.ancestral_sample(
                alpha_a, num_samples, sample_mean, tau=0.1, hard=True
            )
        return o_sample, u_sample