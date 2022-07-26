from random import sample
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as torch_dist
from src.agents.core import AbstractAgent
from src.distributions.hmm import QMDPLayer
from src.distributions.mixture_models import ConditionalGaussian
from src.distributions.nn_models import GRUMLP
from src.distributions.utils import kl_divergence, rectify

from typing import Union, Tuple, Optional
from torch import Tensor

class LFVINAgent(AbstractAgent):
    """ Latent factor value iteraction network agent with 
    conditinal gaussian observation and control models and 
    QMDP hidden layer
    """
    def __init__(
        self, state_dim, act_dim, obs_dim, ctl_dim, rank, horizon,
        num_factors, hidden_dim, num_hidden, gru_layers, activation,
        obs_cov="full", ctl_cov="full", use_tanh=False, ctl_lim=None, 
        ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.ctl_dim = ctl_dim
        self.horizon = horizon
        self.num_factors = num_factors
        
        self.rnn = QMDPLayer(state_dim, act_dim, rank, horizon, place_holder=True)
        self.obs_model = ConditionalGaussian(
            obs_dim, state_dim, cov=obs_cov, batch_norm=True, 
            use_tanh=False, limits=None, place_holder=True
        )
        self.ctl_model = ConditionalGaussian(
            ctl_dim, act_dim, cov=ctl_cov, batch_norm=True, 
            use_tanh=use_tanh, limits=ctl_lim, place_holder=True
        )
        self.c = torch.randn(1, state_dim)
        self._pi0 = torch.randn(1, act_dim, state_dim)
        
        self.parameter_size = [
            self.c.shape, self._pi0.shape,
            self.rnn.b0.shape, self.rnn.u.shape, self.rnn.v.shape, self.rnn.w.shape, self.rnn.tau.shape,
            self.obs_model.mu.shape, self.obs_model.lv.shape, self.obs_model.tl.shape,
            self.ctl_model.mu.shape, self.ctl_model.lv.shape, self.ctl_model.tl.shape
        ]
        
        self.encoder = GRUMLP(
            input_dim=obs_dim + ctl_dim,
            output_dim=num_factors * 2,
            hidden_dim=hidden_dim,
            gru_layers=gru_layers,
            mlp_layers=num_hidden,
            activation=activation
        )
        self.decoder = nn.Linear(num_factors, sum([np.prod(s) for s in self.parameter_size]))
    
    def reset(self):
        """ Reset internal states for online inference """
        self._b = None # torch.ones(1, self.state_dim)
        self._a = None # previous action distribution
        self._prev_ctl = None
        self._theta = None # parameter vecot
        self._ent = None # posterior entropy

    @property
    def target_dist(self):
        return torch.softmax(self.c, dim=-1)
    
    @property
    def pi0(self):
        """ Prior policy """
        return torch.softmax(self._pi0, dim=-1)
    
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
        kl = kl_divergence(transition, c.unsqueeze(-2).unsqueeze(-2))
        eh = torch.sum(transition * entropy.unsqueeze(-2).unsqueeze(-2), dim=-1)
        log_pi0 = torch.log(self.pi0 + 1e-6)
        r = -kl - eh + log_pi0
        return r
    
    def transform_params(self, theta):
        theta_ = torch.split(theta, [np.prod(s) for s in self.parameter_size], dim=-1)
        
        self.c = theta_[0].view([-1] + list(self.parameter_size[0])[1:])
        self._pi0 = theta_[1].view([-1] + list(self.parameter_size[1])[1:])
        self.rnn.b0 = theta_[2].view([-1] + list(self.parameter_size[2])[1:])
        self.rnn.u = theta_[3].view([-1] + list(self.parameter_size[3])[1:])
        self.rnn.v = theta_[4].view([-1] + list(self.parameter_size[4])[1:])
        self.rnn.w = theta_[5].view([-1] + list(self.parameter_size[5])[1:])
        self.rnn.tau = theta_[6].view([-1] + list(self.parameter_size[6])[1:])
        self.obs_model.mu = theta_[7].view([-1] + list(self.parameter_size[7])[1:])
        self.obs_model.lv = theta_[8].view([-1] + list(self.parameter_size[8])[1:])
        self.obs_model.tl = theta_[9].view([-1] + list(self.parameter_size[9])[1:])
        self.ctl_model.mu = theta_[10].view([-1] + list(self.parameter_size[10])[1:])
        self.ctl_model.lv = theta_[11].view([-1] + list(self.parameter_size[11])[1:])
        self.ctl_model.tl = theta_[12].view([-1] + list(self.parameter_size[12])[1:])
    
    def encode(self, o, u):
        """ Sample from variational posterior """
        z_params = self.encoder(torch.cat([o, u], dim=-1))
        mu, lv = torch.chunk(z_params, 2, dim=-1)
        z_dist = torch_dist.Normal(mu, rectify(lv))
        z = z_dist.rsample()
        ent = z_dist.entropy().sum(-1, keepdim=True)
        theta = self.decoder(z)
        return theta, ent
    
    def sample_theta(self):
        z = torch_dist.Normal(
            torch.zeros(1, self.num_factors), torch.ones(1, self.num_factors)
        ).sample()
        theta = self.decoder(z)
        return theta

    def forward(
        self, o: Tensor, u: Union[Tensor, None], 
        hidden: Optional[Union[Tuple[Tensor, Tensor, Tensor, Tensor], None]]=None,
        sample_theta: Optional[bool]=False
        ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """ 
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            hidden ([tuple[torch.tensor] * 4, None], optional). initial hidden state.
            sample_theta (bool, optional): whether to sample theta from prior. If not encode observations. 
        
        Returns:
            alpha_b (torch.tensor): state belief distributions. size=[T, batch_size, state_dim]
            alpha_a (torch.tensor): action predictive distributions. size=[T, batch_size, act_dim]
            theta (torch.tensor): agent parameter vector. size=[batch_size, num_params]
            ent (torch.tensor): variational distribution entropy. size=[batch_size, 1]
        """
        b, a, theta, ent = None, None, None, None
        if hidden is not None:
            b, a, theta, ent = hidden
        
        if theta is None:
            if sample_theta:
                theta = self.sample_theta()
            else:
                theta, ent = self.encode(o, u)
        self.transform_params(theta)

        logp_o = self.obs_model.log_prob(o)
        logp_u = None if u is None else self.ctl_model.log_prob(u)
        reward = self.reward
        alpha_b, alpha_a = self.rnn(logp_o, logp_u, reward, b, a)
        return [alpha_b, alpha_a], [alpha_b, alpha_a, theta, ent] # second tuple used in bptt
    
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
        _, alpha_a, _, ent = forward_out[1]
        logp_u = self.ctl_model.mixture_log_prob(alpha_a, u)
        loss = -torch.sum(logp_u * mask, dim=0) / (mask.sum(0) + 1e-6) - torch.mean(ent)/len(o)

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
        alpha_b, _, _, _ = forward_out[1]
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
        [alpha_b, alpha_a], [_, _, theta, _] = self.forward(
            o.unsqueeze(0), self._prev_ctl, [self._b, self._a, self._theta, self._ent], 
            sample_theta=True
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
        self._theta = theta
        return u_sample, logp
    
    def choose_action_batch(self, o, u, sample_method="ace", num_samples=1, tau=0.1, hard=True, return_hidden=False):
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
        [alpha_b, alpha_a], _ = self.forward(o, u)

        if sample_method == "bma":
            u_sample = self.ctl_model.bayesian_average(alpha_a)
        else:
            sample_mean = True if sample_method == "acm" else False
            u_sample = self.ctl_model.ancestral_sample(
                alpha_a, num_samples, sample_mean, tau, hard
            )
            logp = self.ctl_model.mixture_log_prob(alpha_a, u_sample)
        if return_hidden:
            return u_sample, logp, [alpha_b, alpha_a]
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