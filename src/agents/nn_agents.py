import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple, Optional
from src.agents.core import AbstractAgent
from src.distributions.nn_models import MLP
from src.distributions.mixture_models import ConditionalGaussian
from src.distributions.utils import kl_divergence

class MLPAgent(AbstractAgent):
    def __init__(
        self, obs_dim, ctl_dim, act_dim, hidden_dim, num_hidden, 
        activation="silu"
        ):
        super().__init__()
        self.obs_dim = obs_dim
        self.ctl_dim = ctl_dim
        self.act_dim = act_dim

        self.mlp = MLP(
            input_dim=obs_dim,
            output_dim=act_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation,
            batch_norm=False
        )

        self.ctl_model = ConditionalGaussian(ctl_dim, act_dim, cov="diag", batch_norm=True)
        
        # observation normalization stats
        self.obs_mean = nn.Parameter(torch.zeros(obs_dim), requires_grad=False)
        self.obs_variance = nn.Parameter(torch.ones(obs_dim), requires_grad=False)
    
    def __repr__(self):
        s = "{}(obs_dim={}, ctl_dim={}, act_dim={}, hidden_dim={}, num_hidden={}, activation={})".format(
            self.__class__.__name__, self.obs_dim, self.ctl_dim, self.act_dim,
            self.mlp.hidden_dim, self.mlp.num_hidden, self.mlp.activation,
        )
        return s

    def reset(self):
        """ Reset internal states for online inference """
        self._prev_ctl = None
        self._state = {
            "b": None, # dummy belief distribution
            "pi": None, # previous policy/action prior
        }
    
    def normalize_obs(self, obs):
        obs_norm = (obs - self.obs_mean) / self.obs_variance**0.5
        return obs_norm

    def forward(self, o, **kwargs):
        o = self.normalize_obs(o)
        pi = torch.softmax(self.mlp(o), dim=-1)
        return pi

    def choose_action(self, o, sample_method="ace", num_samples=1):
        """ Choose action online for a single time step
        
        Args:
            o (torch.tensor): observation sequence. size[batch_size, obs_dim]
            sample_method (str, optional): sampling method. 
                choices=["bma", "ace", "acm"]. Default="ace"
            num_samples (int, optional): number of samples to draw. Default=1
        
        Returns:
            u_sample (torch.tensor): sampled controls. size=[num_samples, batch_size, ctl_dim]
            logp (torch.tensor): control log probability. size=[num_samples, batch_size]
        """
        pi_t = self.forward(o)

        if sample_method == "bma":
            u_sample = self.ctl_model.bayesian_average(pi_t)
        else:
            sample_mean = True if sample_method == "acm" else False
            u_sample = self.ctl_model.ancestral_sample(
                pi_t.unsqueeze(0), num_samples, sample_mean
            ).squeeze(-3)
            logp = self.ctl_model.mixture_log_prob(pi_t, u_sample)
        
        self._state["b"] = None
        self._state["pi"] = pi_t
        return u_sample, logp
    
    def choose_action_batch(
        self, o, u, sample_method="ace", num_samples=1, tau=0.1, hard=True, return_hidden=False, **kwargs
        ):
        """ Choose action offline for a batch of sequences 
        
        Args:
            o (torch.tensor): observation sequence. size[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size[T, batch_size, ctl_dim]
            sample_method (str, optional): sampling method. 
                choices=["bma", "ace", "acm"]. Default="ace"
            num_samples (int, optional): number of samples to draw. Default=1
            tau (float, optional): gumbel softmax temperature. Default=0.1
            hard (bool, optional): if hard use straight-through gradient. Default=True
            return_hidden (bool, optional): if true return agent hidden state. Default=False

        Returns:
            u_sample (torch.tensor): sampled controls. size=[num_samples, T, batch_size, ctl_dim]
            logp (torch.tensor): control log probability. size=[num_samples, T, batch_size]
        """
        alpha_pi = self.forward(o)

        if sample_method == "bma":
            u_sample = self.ctl_model.bayesian_average(alpha_pi)
        else:
            sample_mean = True if sample_method == "acm" else False
            u_sample = self.ctl_model.ancestral_sample(
                alpha_pi, num_samples, sample_mean, tau, hard
            )
            logp = self.ctl_model.mixture_log_prob(alpha_pi, u_sample)
        if return_hidden:
            return u_sample, logp, None
        else:
            return u_sample, logp

    def act_loss(self, o, u, mask, forward_out):
        """ Compute action loss by matching forward kl of discrete actions
        
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            mask (torch.tensor): binary mask sequence. size=[T, batch_size]
            forward_out (list): outputs of forward method

        Returns:
            loss (torch.tensor): action loss. size=[batch_size]
            stats (dict): action loss stats
        """
        alpha_pi = forward_out
        logp_u = self.ctl_model.log_prob(u)
        pi_target = torch.softmax(logp_u, dim=-1)
        logp_pi = kl_divergence(pi_target, alpha_pi)
        loss = torch.sum(logp_pi * mask, dim=0) / (mask.sum(0) + 1e-6)

        # compute stats
        nan_mask = mask.clone()
        nan_mask[nan_mask != 0] = 1.
        nan_mask[nan_mask == 0] = torch.nan
        logp_u_mean = torch.nanmean((nan_mask * logp_pi)).cpu().data
        stats = {"loss_u": logp_u_mean}
        return loss, stats

    def obs_loss(self, o, u, mask, forward_out):
        loss = torch.zeros(1)
        stats = {"loss_o": 0.}
        return loss, stats


class RNNAgent(AbstractAgent):
    def __init__(
        self, obs_dim, ctl_dim, act_dim, hidden_dim, num_hidden, gru_layers, activation="silu"
        ):
        super().__init__()
        self.obs_dim = obs_dim
        self.ctl_dim = ctl_dim
        self.act_dim = act_dim
        
        self.h0 = nn.Parameter(torch.randn(gru_layers, hidden_dim))
        self.gru = nn.GRU(obs_dim, hidden_dim, gru_layers)
        self.mlp = MLP(gru_layers * hidden_dim, act_dim, hidden_dim, num_hidden, activation)
        nn.init.xavier_normal_(self.h0, gain=1.)

        self.ctl_model = ConditionalGaussian(ctl_dim, act_dim, cov="diag", batch_norm=True)
        
        # observation normalization stats
        self.obs_mean = nn.Parameter(torch.zeros(obs_dim), requires_grad=False)
        self.obs_variance = nn.Parameter(torch.ones(obs_dim), requires_grad=False)
    
    def __repr__(self):
        s = "{}(obs_dim={}, ctl_dim={}, act_dim={}, hidden_dim={}, num_hidden={}, activation={})".format(
            self.__class__.__name__, self.obs_dim, self.ctl_dim, self.act_dim,
            self.mlp.hidden_dim, self.mlp.num_hidden, self.mlp.activation
        )
        return s

    def reset(self):
        """ Reset internal states for online inference """
        self._prev_ctl = None
        self._state = {
            "b": None, # dummy belief distribution
            "pi": None, # previous policy/action prior
        }
    
    def normalize_obs(self, obs):
        obs_norm = (obs - self.obs_mean) / self.obs_variance**0.5
        return obs_norm
    
    def init_hidden(self, batch_size):
        h0 = torch.repeat_interleave(self.h0.unsqueeze(-2), batch_size, -2)
        return h0
    
    def forward(
        self, o: Tensor, u: Union[Tensor, None], 
        hidden: Optional[Union[Tuple[Tensor, Tensor], list]]=[None], **kwargs
        ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        o = self.normalize_obs(o)
        
        if None in hidden:
            b = self.init_hidden(o.shape[1])
        else:
            b = hidden[0].view(1, o.shape[1], -1)

        alpha_b, _ = self.gru(o, b)

        alpha_pi = torch.softmax(self.mlp.forward(alpha_b), dim=-1)
        return [alpha_b, alpha_pi], [alpha_b, alpha_pi]

    def choose_action(self, o, sample_method="ace", num_samples=1):
        """ Choose action online for a single time step
        
        Args:
            o (torch.tensor): observation sequence. size[batch_size, obs_dim]
            sample_method (str, optional): sampling method. 
                choices=["bma", "ace", "acm"]. Default="ace"
            num_samples (int, optional): number of samples to draw. Default=1
        
        Returns:
            u_sample (torch.tensor): sampled controls. size=[num_samples, batch_size, ctl_dim]
            logp (torch.tensor): control log probability. size=[num_samples, batch_size]
        """
        b_t, pi_t = self._state["b"], self._state["pi"]
        [alpha_b, alpha_pi], _ = self.forward(o.unsqueeze(0), self._prev_ctl, [b_t])
        b_t, pi_t = alpha_b[0], alpha_pi[0]

        if sample_method == "bma":
            u_sample = self.ctl_model.bayesian_average(pi_t)
        else:
            sample_mean = True if sample_method == "acm" else False
            u_sample = self.ctl_model.ancestral_sample(
                pi_t.unsqueeze(0), num_samples, sample_mean
            ).squeeze(-3)
            logp = self.ctl_model.mixture_log_prob(pi_t, u_sample)
        
        self._prev_ctl = u_sample
        self._state["b"] = b_t
        self._state["pi"] = pi_t
        return u_sample, logp
    
    def choose_action_batch(
        self, o, u, sample_method="ace", num_samples=1, tau=0.1, hard=True, return_hidden=False, **kwargs
        ):
        """ Choose action offline for a batch of sequences 
        
        Args:
            o (torch.tensor): observation sequence. size[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size[T, batch_size, ctl_dim]
            sample_method (str, optional): sampling method. 
                choices=["bma", "ace", "acm"]. Default="ace"
            num_samples (int, optional): number of samples to draw. Default=1
            tau (float, optional): gumbel softmax temperature. Default=0.1
            hard (bool, optional): if hard use straight-through gradient. Default=True
            return_hidden (bool, optional): if true return agent hidden state. Default=False

        Returns:
            u_sample (torch.tensor): sampled controls. size=[num_samples, T, batch_size, ctl_dim]
            logp (torch.tensor): control log probability. size=[num_samples, T, batch_size]
        """
        [alpha_b, alpha_pi], hidden = self.forward(o, u)

        if sample_method == "bma":
            u_sample = self.ctl_model.bayesian_average(alpha_pi)
        else:
            sample_mean = True if sample_method == "acm" else False
            u_sample = self.ctl_model.ancestral_sample(
                alpha_pi, num_samples, sample_mean, tau, hard
            )
            logp = self.ctl_model.mixture_log_prob(alpha_pi, u_sample)
        if return_hidden:
            return u_sample, logp, hidden
        else:
            return u_sample, logp

    def act_loss(self, o, u, mask, hidden):
        """ Compute action loss by matching forward kl of discrete actions
        
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            mask (torch.tensor): binary mask sequence. size=[T, batch_size]
            forward_out (list): outputs of forward method

        Returns:
            loss (torch.tensor): action loss. size=[batch_size]
            stats (dict): action loss stats
        """
        _, alpha_pi = hidden
        logp_u = self.ctl_model.log_prob(u)
        pi_target = torch.softmax(logp_u, dim=-1)
        logp_pi = kl_divergence(pi_target, alpha_pi)
        loss = torch.sum(logp_pi * mask, dim=0) / (mask.sum(0) + 1e-6)

        # compute stats
        nan_mask = mask.clone()
        nan_mask[nan_mask != 0] = 1.
        nan_mask[nan_mask == 0] = torch.nan
        logp_u_mean = torch.nanmean((nan_mask * logp_pi)).cpu().data
        stats = {"loss_u": logp_u_mean}
        return loss, stats

    def obs_loss(self, o, u, mask, hidden, **kwargs):
        loss = torch.zeros(1)
        stats = {"loss_o": 0.}
        return loss, stats