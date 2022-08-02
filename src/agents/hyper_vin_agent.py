import torch
import torch.nn as nn
import torch.distributions as torch_dist
from src.agents.core import AbstractAgent
from src.agents.qmdp_layers import HyperQMDPLayer
from src.distributions.mixture_models import ConditionalGaussian
from src.distributions.mixture_models import HyperConditionalGaussian
from src.distributions.nn_models import GRUMLP
from src.distributions.utils import kl_divergence, rectify

from typing import Union, Tuple, Optional
from torch import Tensor

class HyperVINAgent(AbstractAgent):
    """ Hyper value iteraction network agent with 
    conditinal gaussian observation and control models and 
    QMDP hidden layer
    """
    def __init__(
        self, state_dim, act_dim, obs_dim, ctl_dim, rank, horizon,
        hyper_dim, hidden_dim, num_hidden, gru_layers, activation,
        obs_cov="full", ctl_cov="full", use_tanh=False, ctl_lim=None, 
        ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.ctl_dim = ctl_dim
        self.horizon = horizon
        self.hyper_dim = hyper_dim
        
        self.rnn = HyperQMDPLayer(state_dim, act_dim, rank, horizon, hyper_dim)
        # self.obs_model = HyperConditionalGaussian(
        #     obs_dim, state_dim, hyper_dim, cov=obs_cov, batch_norm=True, 
        #     use_tanh=False, limits=None
        # )
        
        # use shared obs model for now
        self.obs_model = ConditionalGaussian(
            obs_dim, state_dim, cov=obs_cov, batch_norm=True, 
            use_tanh=False, limits=None
        )
        self.ctl_model = ConditionalGaussian(
            ctl_dim, act_dim, cov=ctl_cov, batch_norm=True, 
            use_tanh=use_tanh, limits=ctl_lim
        )
        self._c = nn.Linear(hyper_dim, state_dim)
        self._pi0 = nn.Linear(hyper_dim, act_dim * state_dim)
        
        # hyper prior
        self.mu = nn.Parameter(torch.randn(1, hyper_dim))
        self.lv = nn.Parameter(torch.randn(1, hyper_dim))

        self.encoder = GRUMLP(
            input_dim=obs_dim + ctl_dim,
            output_dim=hyper_dim * 2,
            hidden_dim=hidden_dim,
            gru_layers=gru_layers,
            mlp_layers=num_hidden,
            activation=activation
        )

        self.obs_mean = nn.Parameter(torch.zeros(obs_dim), requires_grad=False)
        self.obs_variance = nn.Parameter(torch.ones(obs_dim), requires_grad=False)
    
    def reset(self):
        """ Reset internal states for online inference """
        self._b = None # prior belief
        self._a = None # previous action distribution
        self._prev_ctl = None # previous control
        self._z = None # hyper vector
        self._ent = None # hyper posterior entropy

    @property
    def target_dist(self):
        z = torch.ones(1, self.hyper_dim).to(self.device)
        return self.compute_target_dist(z)
    
    @property
    def pi0(self):
        """ Prior policy """
        z = torch.ones(1, self.hyper_dim).to(self.device)
        return self.compute_prior_policy(z)
    
    @property
    def transition(self):
        z = torch.ones(1, self.hyper_dim).to(self.device)
        return self.rnn.compute_transition(z)
    
    @property
    def value(self):
        value = self.rnn.compute_value(self.transition, self.reward)
        return value
    
    @property
    def policy(self):
        """ Optimal planned policy """
        z = torch.ones(1, self.hyper_dim).to(self.device)
        b = torch.eye(self.state_dim).to(self.device)
        pi = self.rnn.plan(b, z, self.value)
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
        z = torch.ones(1, self.hyper_dim).to(self.device)
        # entropy = self.obs_model.entropy(z)
        entropy = self.obs_model.entropy()
        c = self.target_dist
        kl = kl_divergence(torch.eye(self.state_dim), c)
        return -kl - entropy
    
    @property
    def reward(self):
        z = torch.ones(1, self.hyper_dim).to(self.device)
        return self.compute_reward(z)
    
    def compute_target_dist(self, z):
        return torch.softmax(self._c(z), dim=-1)
    
    def compute_prior_policy(self, z):
        pi0 = self._pi0(z).view(-1, self.act_dim, self.state_dim)
        return torch.softmax(pi0, dim=-2)

    def compute_reward(self, z):
        """ State action reward """
        transition = self.rnn.compute_transition(z)
        # entropy = self.obs_model.entropy(z)
        entropy = self.obs_model.entropy()

        c = self.compute_target_dist(z)
        pi0 = self.compute_prior_policy(z)
        kl = kl_divergence(transition, c.unsqueeze(-2).unsqueeze(-2))
        eh = torch.sum(transition * entropy.unsqueeze(-2).unsqueeze(-2), dim=-1)
        log_pi0 = torch.log(pi0 + 1e-6)
        r = -kl - eh + log_pi0
        return r

    def encode(self, o, u):
        """ Sample from variational posterior """
        o_norm = (o - self.obs_mean) / self.obs_variance**0.5
        z_params = self.encoder(torch.cat([o_norm, u], dim=-1))
        mu, lv = torch.chunk(z_params, 2, dim=-1)
        
        z_dist = torch_dist.Normal(mu, rectify(lv))
        z = z_dist.rsample()
        
        prior_dist = torch_dist.Normal(self.mu, rectify(self.lv))
        kl = torch_dist.kl.kl_divergence(z_dist, prior_dist).sum(-1, keepdim=True)
        return z, kl
    
    def sample_z(self):
        z = torch_dist.Normal(
            self.mu, rectify(self.lv)
        ).rsample()
        return z

    def forward(
        self, o: Tensor, u: Union[Tensor, None], 
        hidden: Optional[Union[Tuple[Tensor, Tensor, Tensor, Tensor], None]]=None,
        sample_z: Optional[bool]=False
        ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """ 
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            hidden ([tuple[torch.tensor] * 4, None], optional). initial hidden state.
            sample_z (bool, optional): whether to sample z from prior. If not encode observations. 
        
        Returns:
            alpha_b (torch.tensor): state belief distributions. size=[T, batch_size, state_dim]
            alpha_a (torch.tensor): action predictive distributions. size=[T, batch_size, act_dim]
            z (torch.tensor): agent parameter vector. size=[batch_size, num_params]
            ent (torch.tensor): variational distribution entropy. size=[batch_size, 1]
        """
        b, a, z, ent = None, None, None, None
        if hidden is not None:
            b, a, z, ent = hidden
        
        if z is None:
            if sample_z:
                z = self.sample_z()
            else:
                z, ent = self.encode(o, u)

        # logp_o = self.obs_model.log_prob(o, z)
        logp_o = self.obs_model.log_prob(o)
        logp_u = None if u is None else self.ctl_model.log_prob(u)
        reward = self.compute_reward(z)
        alpha_b, alpha_a = self.rnn(logp_o, logp_u, reward, z, b, a)
        return [alpha_b, alpha_a], [alpha_b, alpha_a, z, ent] # second tuple used in bptt
    
    def act_loss(self, o, u, mask, hidden):
        """ Compute action loss 
        
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            mask (torch.tensor): binary mask sequence. size=[T, batch_size]
            hidden (list): outputs of forward method

        Returns:
            loss (torch.tensor): action loss. size=[batch_size]
            stats (dict): action loss stats
        """
        _, alpha_a, _, kl = hidden
        logp_u = self.ctl_model.mixture_log_prob(alpha_a, u)
        loss = -torch.sum(logp_u * mask, dim=0) / (mask.sum(0) + 1e-6) + torch.mean(kl)/len(o)
        
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
            forward_out (list): outputs of forward method

        Returns:
            loss (torch.tensor): observation loss. size=[batch_size]
            stats (dict): observation loss stats
        """
        alpha_b, _, z, _ = hidden
        # logp_o = self.obs_model.mixture_log_prob(alpha_b, o, z)
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
        [alpha_b, alpha_a], [_, _, z, _] = self.forward(
            o.unsqueeze(0), self._prev_ctl, [self._b, self._a, self._z, self._ent], 
            sample_z=True
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
        self._z = z
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

    # def predict(self, o, u, sample_method="ace", num_samples=1):
    #     """ Offline prediction observations and control """
    #     [alpha_b, alpha_a], _ = self.forward(o, u)

    #     if sample_method == "bma":
    #         o_sample = self.obs_model.bayesian_average(alpha_b)
    #         u_sample = self.ctl_model.bayesian_average(alpha_a)

    #     else:
    #         sample_mean = True if sample_method == "acm" else False
    #         o_sample = self.obs_model.ancestral_sample(
    #             alpha_b, num_samples, sample_mean, tau=0.1, hard=True
    #         )
    #         u_sample = self.ctl_model.ancestral_sample(
    #             alpha_a, num_samples, sample_mean, tau=0.1, hard=True
    #         )
    #     return o_sample, u_sample