import torch
import torch.nn as nn
import torch.distributions as torch_dist
from torch import Tensor
from typing import Union, Tuple, Optional

from src.agents.core import AbstractAgent
from src.agents.hyper_qmdp_layer import HyperQMDPLayer
from src.distributions.mixture_models import ConditionalGaussian
from src.distributions.hyper_mixture_models import HyperConditionalGaussian, HyperConditionalFlow
from src.distributions.nn_models import GRUMLP
from src.distributions.utils import kl_divergence, rectify

""" TODO: parameterize all factors as row orthogonal matrix """
class HyperVINAgent(AbstractAgent):
    """ Hypernet version of the VIN agent """
    def __init__(
        self, state_dim, act_dim, obs_dim, ctl_dim, rank, horizon,
        hyper_dim, hidden_dim, num_hidden, gru_layers, activation,
        alpha=1., beta=1., obs_model="flow", obs_cov="full", ctl_cov="full", rwd="efe",
        hyper_cov=True, train_prior=False
        ):
        """
        Args:
            state_dim (int): hidden state dimension
            act_dim (int): discrete action dimension
            obs_dim (int): observation dimension
            ctl_dim (int): control dimension
            rank (int): transition matrix embedding dimension. If rank=0 transition matrix is full rank
            horizon (int): maximum planning horizon
            hyper_dim (int): hyper variable dimension
            hidden_dim (int): inference network hidden dimension
            num_hidden (int): inference network hidden layers
            gru_layer (int): inference network gru layers
            activation (str): inference network activation
            alpha (float, optional): observation entropy weight. Default=1.
            beta (float, optional): prior policy likelihood weight. Default=0.
            obs_model (str, optional): observation model type. choices=["gmm", "flow"]. Default="flow"
            obs_cov (str, optional): observation covariance. Default="full"
            ctl_cov (str, optional): control covariance. Default="full"
            rwd (str, optional): reward function type. choices=["efe", "ece"]. Default="efe"
            hyper_cov (bool, optional): whether to use hyper variable for cov. Default=True
            train_prior (bool, optional): whether to train hyper prior. Default=False
        """
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.ctl_dim = ctl_dim
        self.horizon = horizon
        self.alpha = alpha # obs entropy temperature
        self.beta = beta # policy prior temperature
        self.rwd = rwd
        self.hyper_dim = hyper_dim
        self.hyper_cov = hyper_cov
        
        self.rnn = HyperQMDPLayer(state_dim, act_dim, rank, horizon, hyper_dim)
        if obs_model == "flow":
            self.obs_model = HyperConditionalFlow(
                obs_dim, state_dim, hyper_dim, cov=obs_cov, hyper_cov=hyper_cov, batch_norm=True
            )
        else:
            self.obs_model = HyperConditionalGaussian(
                obs_dim, state_dim, hyper_dim, cov=obs_cov, hyper_cov=hyper_cov, batch_norm=True
            )
        self.ctl_model = ConditionalGaussian(ctl_dim, act_dim, cov=ctl_cov, batch_norm=True)
        
        self._c = nn.Linear(hyper_dim, state_dim)
        self._pi0 = nn.Parameter(torch.randn(1, act_dim, state_dim))
        self._gamma = nn.Linear(hyper_dim, 1)
        
        self._c.weight.data = 0.1 * torch.randn(self._c.weight.data.shape)
        self._gamma.weight.data = 0.1 * torch.randn(self._gamma.weight.data.shape)
        nn.init.xavier_normal_(self._pi0, gain=1.)
        
        # hyper prior
        self.prior_mu = nn.Parameter(torch.zeros(1, hyper_dim), requires_grad=train_prior)
        self.prior_lv = nn.Parameter(torch.zeros(1, hyper_dim), requires_grad=train_prior)
        
        # inference network
        self.encoder = GRUMLP(
            input_dim=obs_dim + ctl_dim,
            output_dim=hyper_dim * 2,
            hidden_dim=hidden_dim,
            gru_layers=gru_layers,
            mlp_layers=num_hidden,
            activation=activation
        )
        
        # obs stats for encoder input normalization
        self.obs_mean = nn.Parameter(torch.zeros(obs_dim), requires_grad=False)
        self.obs_variance = nn.Parameter(torch.ones(obs_dim), requires_grad=False)
    
    def reset(self, z=None):
        """ Reset internal states for online inference 
        
        Args:
            z (torch.tensor): hyper variable. size=[1, hyper_dim]
        """
        if z is None:
            z = self.sample_z()
            
        self._prev_ctl = None # previous control
        self._value = None # precomputed value
        self._state = {
            "b": None, # previous belief distribution
            "pi": None, # previous policy/action prior
            "z": z, # hyper vector
        }

    def compute_target_dist(self, z):
        return torch.softmax(self._c(z), dim=-1)
    
    def compute_efe(self, z, detach=False):
        """ Compute expected free energy """
        transition = self.rnn.compute_transition(z)
        entropy = self.obs_model.entropy(z) / self.obs_dim

        if detach:
            transition = transition.data
            entropy = entropy.data

        c = self.compute_target_dist(z)
        kl = kl_divergence(transition, c.unsqueeze(-2).unsqueeze(-2))
        eh = torch.sum(transition * entropy.unsqueeze(-2).unsqueeze(-2), dim=-1)
        r = -kl - self.alpha * eh
        return r
    
    def compute_ece(self, z, num_samples=200, detach=False):
        """ Compute expected cross entropy """
        # sample observation
        transition = self.rnn.compute_transition(z)
        o = self.obs_model.sample(z, (num_samples,)).transpose(1, 2)
        self.obs_model.bn.training = False
        logp_o = self.obs_model.log_prob(o, z)
        self.obs_model.bn.training = self.training 
        
        if detach:
            transition = transition.data
            logp_o = logp_o.data
        
        # compute expected reward
        target_dist = self.compute_target_dist(z)
        logp_s = torch.log(target_dist + 1e-6)
        log_r = torch.logsumexp(logp_s + logp_o, dim=-1).transpose(1, 2)
        r = torch.einsum("nkij, nj -> nki", transition, log_r.mean(0))
        return r

    def compute_prior_policy(self, z):
        gamma = rectify(self._gamma(z)).unsqueeze(-1)
        return torch.softmax(gamma * self._pi0, dim=-2)
    
    def compute_value(self, z):
        transition = self.rnn.compute_transition(z)
        value = self.rnn.compute_value(transition, self.compute_reward(z))
        return value
    
    def compute_policy(self, z):        
        value = self.compute_value(z)
        b = torch.eye(self.state_dim).to(self.device)
        pi = self.rnn.plan(b, z, value)
        return pi

    def compute_reward(self, z, detach=False):
        """ State action reward """
        pi0 = self.compute_prior_policy(z)
        log_pi0 = torch.log(pi0 + 1e-6)

        if self.rwd == "efe":
            r = self.compute_efe(z, detach=detach) + self.beta * log_pi0
        else:
            r = self.compute_ece(z, detach=detach) + self.beta * log_pi0
        return r
    
    def get_prior_dist(self):
        return torch_dist.Normal(self.prior_mu, rectify(self.prior_lv))        

    def sample_z(self, sample_shape=torch.Size()):
        """ Sample from hyper prior """
        return self.get_prior_dist().rsample(sample_shape)
    
    def get_posterior_dist(self, o, u, mask):
        o_norm = mask.unsqueeze(-1) * (o - self.obs_mean) / self.obs_variance**0.5
        z_params = self.encoder(torch.cat([o_norm, u], dim=-1), mask)
        mu, lv = torch.chunk(z_params, 2, dim=-1)
        
        z_dist = torch_dist.Normal(mu, rectify(lv))
        return z_dist

    def encode(self, o, u, mask):
        """ Sample from variational posterior """
        z_dist = self.get_posterior_dist(o, u, mask)
        z = z_dist.rsample()
        return z
    
    def forward(
        self, o: Tensor, u: Union[Tensor, None], z: Tensor,
        hidden: Optional[Union[Tuple[Tensor, Tensor], list]]=[None],
        value: Optional[Union[Tensor, None]]=None, 
        detach: Optional[bool]=False, **kwargs
        ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """ 
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            z (torch.tensor): hyper variable. size=[batch_size, hyper_dim]
            hidden ([tuple[torch.tensor] * 2, None], optional). initial hidden state.
            value (tuple[torch.tensor, None], optional): precomputed value matrix. 
                size=[horizon, batch_size, act_dim, state_dim]. Default=None
            detach (bool, optional): whether to detach model training. Default=False
        
        Returns:
            alpha_b (torch.tensor): state belief distributions. size=[T, batch_size, state_dim]
            alpha_a (torch.tensor): action predictive distributions. size=[T, batch_size, act_dim]
        """
        if value is None:
            value = self.compute_value(z)
        
        if None in hidden:
            b, pi = self.rnn.init_hidden(z, value)
        else:
            b, pi = hidden
        
        logp_o = self.obs_model.log_prob(o, z)
        logp_u = torch.log(pi + 1e-6).unsqueeze(0)
        if u is not None:
            logp_u = torch.cat([logp_u, self.ctl_model.log_prob(u)])

        alpha_b, alpha_pi = self.rnn.forward(logp_o, logp_u, value, z, b, detach=detach)
        return [alpha_b, alpha_pi, value], [alpha_b, alpha_pi]
    
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
        [alpha_b, alpha_pi, value], _ = self.forward(
            o.unsqueeze(0), self._prev_ctl, self._state["z"], [b_t, pi_t]
        )
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
        self._value = value
        self._state["b"] = b_t
        self._state["pi"] = pi_t
        return u_sample, logp
    
    def choose_action_batch(
        self, o, u, z=None, sample_method="ace", num_samples=1, tau=0.1, 
        hard=True, return_hidden=False
        ):
        """ Choose action offline for a batch of sequences 
        
        Args:
            o (torch.tensor): observation sequence. size[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size[T, batch_size, ctl_dim]
            z (torch.tensor): hyper variable. size=[batch_size, hyper_dim]
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
        if z is None:
            z = self.sample_z(o.shape[1])

        [alpha_b, alpha_a, _], hidden = self.forward(o, u, z)

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

    def act_loss(self, o, u, z, mask, hidden):
        """ Compute action loss by matching forward kl of discrete actions
        
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            z (torch.tensor): hyper variable. size=[batch_size, hyper_dim]
            mask (torch.tensor): binary mask sequence. size=[T, batch_size]
            hidden (list): hidden outputs of forward method

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

    def obs_loss(self, o, u, z, mask, hidden, pred_steps=1):
        """ Compute multi-step observation prediction loss 
        
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            z (torch.tensor): hyper variable. size=[batch_size, hyper_dim]
            mask (torch.tensor): binary mask tensor. size=[T, batch_size]
            hidden (list): hidden outputs of forward method
            pred_steps (int, optional): number of steps ahead to predict. Default=1.

        Returns:
            loss (torch.tensor): observation loss. size=[batch_size]
            stats (dict): observation loss stats
        """
        alpha_b, _ = hidden
        
        # multi step prediction
        logp_u = self.ctl_model.log_prob(u)
        s_pred = [alpha_b[:-pred_steps]] + [torch.empty(0)] * pred_steps
        for i in range(pred_steps):
            s_pred[i+1] = self.rnn.predict_one_step(logp_u[i:-pred_steps+i], s_pred[i], z)
        
        logp_o = self.obs_model.mixture_log_prob(s_pred[-1], o[pred_steps:], z)
        loss = -torch.sum(logp_o * mask[pred_steps:], dim=0) / (mask[pred_steps:].sum(0) + 1e-6)

        # compute stats
        nan_mask = mask[pred_steps:].clone()
        nan_mask[nan_mask != 0] = 1.
        nan_mask[nan_mask == 0] = torch.nan
        logp_o_mean = -torch.nanmean((nan_mask * logp_o)).cpu().data
        stats = {"loss_o": logp_o_mean}
        return loss, stats

    def compute_mutual_information(self, o, u, mask):
        """ Compute conditiona mutual information H(z, u|o) """
        prior_dist = self.get_prior_dist()
        z = prior_dist.rsample((o.shape[1],)).squeeze(-2)
        u_sample = self.choose_action_batch(o, u, z)[0].squeeze(0)
        post_dist = self.get_posterior_dist(o, u_sample, mask)
        mi = post_dist.entropy().sum(-1).mean() - prior_dist.entropy().sum(-1)
        return mi

    def compute_hessian_penalty(self, o, u, z, mask, hidden):
        """ Compute hessian penalty using finite difference """
        def masked_mean(x, mask, dim):
            return torch.sum(x * mask, dim=dim) / (mask.sum(dim) + 1e-6)
        
        eps = 1e-3
        rademacher_vec = torch.randint(0, 2, size=z.shape)
        rademacher_vec[rademacher_vec == 0] = -1
        v = eps * rademacher_vec
        
        mask_ = mask.unsqueeze(-1)
        _, alpha_pi = hidden
        _, [_, alpha_pi_1] = self.forward(o, u, z + v)
        _, [_, alpha_pi_2] = self.forward(o, u, z - v)
        finite_diff = (alpha_pi_1 - 2 * alpha_pi + alpha_pi_2) / eps**2 * mask_
        
        penalty = masked_mean(finite_diff**2, mask_, dim=1) - masked_mean(finite_diff, mask_, dim=1)**2
        return penalty.max()