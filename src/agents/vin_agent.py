import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple, Optional

from src.agents.core import AbstractAgent
from src.agents.qmdp_layer import QMDPLayer
from src.distributions.mixture_models import ConditionalGaussian, ConditionalFlow
from src.distributions.utils import kl_divergence

class VINAgent(AbstractAgent):
    """ Active inference agent implemented as a Value Interation Network """
    def __init__(
        self, state_dim, act_dim, obs_dim, ctl_dim, rank, horizon, 
        alpha=1., beta=1., obs_model="flow", obs_cov="full", ctl_cov="full", 
        rwd="efe", detach=False
        ):
        """
        Args:
            state_dim (int): hidden state dimension
            act_dim (int): discrete action dimension
            obs_dim (int): observation dimension
            ctl_dim (int): control dimension
            rank (int): transition matrix embedding dimension. If rank=0 transition matrix is full rank
            horizon (int): maximum planning horizon
            alpha (float, optional): observation entropy weight. Default=1.
            beta (float, optional): prior policy likelihood weight. Default=0.
            obs_model (str, optional): observation model type. choices=["gmm", "flow"]. Default="flow"
            obs_cov (str, optional): observation covariance. Default="full"
            ctl_cov (str, optional): control covariance. Default="full"
            rwd (str, optional): reward function type. choices=["efe", "ece"]. Default="efe"
            detach (bool, optional): whether to stop model gradient in reward and policy computation. Default=False
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
        self.detach = detach
        
        self.rnn = QMDPLayer(state_dim, act_dim, rank, horizon, detach=detach)
        if obs_model == "flow":
            self.obs_model = ConditionalFlow(obs_dim, state_dim, cov=obs_cov, batch_norm=True)
        else:
            self.obs_model = ConditionalGaussian(obs_dim, state_dim, cov=obs_cov, batch_norm=True)
        
        self.ctl_model = ConditionalGaussian(ctl_dim, act_dim, cov=ctl_cov, batch_norm=True)

        self.c = nn.Parameter(torch.randn(1, state_dim))
        self._pi0 = nn.Parameter(torch.randn(1, act_dim, state_dim)) 
        
        nn.init.xavier_normal_(self.c, gain=1.)
        nn.init.xavier_normal_(self._pi0, gain=1.)
    
    def reset(self):
        """ Reset internal states for online inference """
        self._prev_ctl = None # previous control
        self._value = None # precomputed value
        self._state = {
            "b": None, # previous belief distribution
            "pi": None, # previous policy/action prior
        }
    
    def compute_target_dist(self):
        return torch.softmax(self.c, dim=-1)

    def compute_pi0(self):
        return torch.softmax(self._pi0, dim=-2)

    def compute_value(self):
        transition = self.rnn.compute_transition()
        reward = self.compute_reward()
        value = self.rnn.compute_value(transition, reward)
        return value

    def compute_policy(self, b):
        value = self.compute_value()
        pi = self.rnn.plan(b, value)
        return pi
    
    def compute_efe(self):
        """ Compute negative expected free energy """
        transition = self.rnn.compute_transition()
        entropy = self.obs_model.entropy() / self.obs_dim

        if self.detach:
            transition = transition.data
            entropy = entropy.data
        
        c = self.compute_target_dist()
        kl = kl_divergence(transition, c.unsqueeze(-2).unsqueeze(-2))
        eh = torch.sum(transition * entropy.unsqueeze(-2).unsqueeze(-2), dim=-1)
        r = -kl - self.alpha * eh
        return r

    def compute_ece(self, num_samples=200):
        """ Compute expected cross entropy """
        # sample observation
        transition = self.rnn.compute_transition()
        o = self.obs_model.sample((num_samples,))
        self.obs_model.bn.training = False # temp solution to work with batch norm
        logp_o = self.obs_model.log_prob(o)
        self.obs_model.bn.training = self.training 
        
        if self.detach:
            transition = transition.data
            logp_o = logp_o.data
        
        # compute expected reward
        logp_s = torch.log(self.compute_target_dist() + 1e-6)
        log_r = torch.logsumexp(logp_s + logp_o, dim=-1)
        r = torch.einsum("nkij, nj -> nki", transition, log_r.mean(0))
        return r
    
    def compute_ig(self, num_samples):
        """ Compute expected information gain """
        # sample observation
        transition = self.rnn.compute_transition()
        o = self.obs_model.sample((num_samples,))
        self.obs_model.bn.training = False
        logp_o = self.obs_model.log_prob(o)
        self.obs_model.bn.training = self.training 
        
        if self.detach:
            transition = transition.data
            logp_o = logp_o.data
        
        # second to last dim is the ancestral sampling source dim
        log_transition = torch.log(transition + 1e-6).unsqueeze(0).unsqueeze(-2)
        b_next = torch.softmax(
            log_transition + logp_o.unsqueeze(2).unsqueeze(2), dim=-1
        )
        kl = kl_divergence(b_next, transition.unsqueeze(0).unsqueeze(-2))
        ig = torch.einsum("nkij, nkij -> nki", transition, kl.sum(0)) / self.obs_dim
        return ig
    
    def compute_reward(self):
        log_pi0 = torch.log(self.compute_pi0() + 1e-6)
        if self.rwd == "efe":
            r = self.compute_efe() + self.beta * log_pi0
        else:
            r = self.compute_ece() + self.beta * log_pi0
        return r

    def forward(
        self, o: Tensor, u: Union[Tensor, None], 
        hidden: Optional[Union[Tuple[Tensor, Tensor], list]]=[None], 
        value: Optional[Union[Tensor, None]]=None, **kwargs
        ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """ 
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            hidden ([tuple[torch.tensor, torch.tensor], None], optional). initial hidden state.
            value (tuple[torch.tensor, None], optional): precomputed value matrix. 
                size=[horizon, batch_size, act_dim, state_dim]. Default=None
        
        Returns:
            alpha_b (torch.tensor): state belief distributions. size=[T, batch_size, state_dim]
            alpha_a (torch.tensor): action predictive distributions. size=[T, batch_size, act_dim]
            value (torch.tensor): value matrix. size=[horizon, batch_size, act_dim, state_dim]
        """
        batch_size = o.shape[1]
        
        if value is None:
            value = self.compute_value()
        
        if None in hidden:
            b, pi = self.rnn.init_hidden(batch_size, value)
        else:
            b, pi = hidden
        
        logp_o = self.obs_model.log_prob(o)
        logp_u = torch.log(pi + 1e-6).unsqueeze(0)
        if u is not None:
            logp_u = torch.cat([logp_u, self.ctl_model.log_prob(u)])
        
        alpha_b, alpha_pi = self.rnn.forward(logp_o, logp_u, value, b)
        return [alpha_b, alpha_pi, value], [alpha_b, alpha_pi] # second tuple used in bptt
    
    def choose_action(self, o, sample_method="ace", num_samples=1):
        """ Choose action online for a single time step
        
        Args:
            o (torch.tensor): observation sequence. size[batch_size, obs_dim]
            u (torch.tensor): control sequence. size[batch_size, ctl_dim]
            sample_method (str, optional): sampling method. 
                choices=["bma", "ace", "acm"]. Default="ace"
            num_samples (int, optional): number of samples to draw. Default=1
        
        Returns:
            u_sample (torch.tensor): sampled controls. size=[num_samples, batch_size, ctl_dim]
            logp (torch.tensor): control log probability. size=[num_samples, batch_size]
        """
        b_t, pi_t = self._state["b"], self._state["pi"]
        [alpha_b, alpha_pi, value], _ = self.forward(
            o.unsqueeze(0), self._prev_ctl, [b_t, pi_t], self._value
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
        [alpha_b, alpha_pi, _], hidden = self.forward(o, u)

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

    def obs_loss(self, o, u, mask, hidden, pred_steps=1):
        """ Compute multi-step observation prediction loss 
        
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
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
            s_pred[i+1] = self.rnn.predict_one_step(logp_u[i:-pred_steps+i], s_pred[i])
        
        logp_o = self.obs_model.mixture_log_prob(s_pred[-1], o[pred_steps:])
        loss = -torch.sum(logp_o * mask[pred_steps:], dim=0) / (mask[pred_steps:].sum(0) + 1e-6)

        # compute stats
        nan_mask = mask[pred_steps:].clone()
        nan_mask[nan_mask != 0] = 1.
        nan_mask[nan_mask == 0] = torch.nan
        logp_o_mean = -torch.nanmean((nan_mask * logp_o)).cpu().data
        stats = {"loss_o": logp_o_mean}
        return loss, stats
    
    def compute_mutual_information(self, o, u, mask, hidden):
        """ Compute mutual information between state and observation """
        alpha_b, _ = hidden

        # compute variational prior
        logp_u = self.ctl_model.log_prob(u)
        s_next = self.rnn.predict_one_step(logp_u, alpha_b)
        logp_s = torch.log(s_next + 1e-6)

        # sample observation
        o_sample = self.obs_model.ancestral_sample(s_next).squeeze(0)
        
        # compute posterior
        logp_o = self.obs_model.log_prob(o_sample)
        log_b_next = torch.log_softmax(logp_o + logp_s, dim=-1)
        
        cross_ent = torch.sum(s_next * log_b_next, dim=-1)
        ent = -torch.sum(s_next * logp_s, dim=-1)
        
        mi = cross_ent + ent
        mi = torch.sum(mi * mask, dim=0) / (mask.sum(0) + 1e-6)
        return mi

    def predict(self, o, u, sample_method="ace", num_samples=1):
        """ Offline prediction observations and control """
        [alpha_b, alpha_pi, _], _ = self.forward(o, u)
        
        # one step transition
        logp_u = self.ctl_model.log_prob(u)
        s_next = self.rnn.predict_one_step(logp_u, alpha_b)
        
        if sample_method == "bma":
            o_sample = self.obs_model.bayesian_average(s_next)
            u_sample = self.ctl_model.bayesian_average(alpha_pi)

        else:
            sample_mean = True if sample_method == "acm" else False
            o_sample = self.obs_model.ancestral_sample(
                s_next, num_samples, sample_mean, tau=0.1, hard=True
            )
            u_sample = self.ctl_model.ancestral_sample(
                alpha_pi, num_samples, sample_mean, tau=0.1, hard=True
            )
        return o_sample, u_sample