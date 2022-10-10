import torch
import torch.nn as nn
from src.agents.core import AbstractAgent
from src.agents.qmdp_layers import QMDPLayer, QMDPLayer2
from src.distributions.mixture_models import ConditionalGaussian
from src.distributions.utils import kl_divergence

from typing import Union, Tuple, Optional
from torch import Tensor

class VINAgent(AbstractAgent):
    """ Value iteraction network agent using global action prior with 
    conditinal gaussian observation and control models and 
    QMDP hidden layer
    """
    def __init__(
        self, state_dim, act_dim, obs_dim, ctl_dim, rank, horizon, alpha=1., beta=1.,
        obs_cov="full", ctl_cov="full", rwd="efe", use_tanh=False, ctl_lim=None, detach=True
        ):
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
        self.obs_model = ConditionalGaussian(
            obs_dim, state_dim, cov=obs_cov, batch_norm=True, 
            use_tanh=False, limits=None
        )
        self.ctl_model = ConditionalGaussian(
            ctl_dim, act_dim, cov=ctl_cov, batch_norm=True, 
            use_tanh=use_tanh, limits=ctl_lim
        )
        self.c = nn.Parameter(torch.randn(1, state_dim))
        self._pi0 = nn.Parameter(torch.randn(1, act_dim, state_dim))
        
        nn.init.xavier_normal_(self.c, gain=1.)
        nn.init.xavier_normal_(self._pi0, gain=1.)

        # self.parameter_size = [
        #     self.c.shape, self._pi0.shape,
        #     self.rnn.b0.shape, self.rnn.u.shape, self.rnn.v.shape, self.rnn.w.shape, self.rnn.tau.shape,
        #     self.obs_model.mu.shape, self.obs_model.lv.shape, self.obs_model.tl.shape,
        #     self.ctl_model.mu.shape, self.ctl_model.lv.shape, self.ctl_model.tl.shape
        # ]
    
    def reset(self):
        """ Reset internal states for online inference """
        self._prev_ctl = None
        self._state = {
            "b": None, # previous belief distribution
            "pi": None, # previous policy/action prior
        }

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
        reward = self.compute_reward()
        value = self.rnn.compute_value(self.transition, reward)
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
        entropy = self.obs_model.entropy() / self.obs_dim
        c = self.target_dist
        kl = kl_divergence(torch.eye(self.state_dim), c)
        return -kl - self.alpha * entropy
    
    def compute_efe(self):
        """ Compute negative expected free energy """
        transition = self.rnn.transition
        entropy = self.obs_model.entropy() / self.obs_dim

        if self.detach:
            transition = transition.data
            entropy = entropy.data
        
        c = self.target_dist
        kl = kl_divergence(transition, c.unsqueeze(-2).unsqueeze(-2))
        eh = torch.sum(transition * entropy.unsqueeze(-2).unsqueeze(-2), dim=-1)
        r = -kl - self.alpha * eh
        return r

    def compute_ece(self, num_samples=200):
        """ Compute expected cross entropy """
        # sample observation
        transition = self.rnn.transition
        o = self.obs_model.sample((num_samples,))
        self.obs_model.bn.training = False # temp solution to work with batch norm
        logp_o = self.obs_model.log_prob(o)
        self.obs_model.bn.training = self.training 
        
        if self.detach:
            transition = transition.data
            logp_o = logp_o.data
        
        # compute expected reward
        logp_s = torch.log(self.target_dist + 1e-6)
        log_r = torch.logsumexp(logp_s + logp_o, dim=-1)
        r = torch.einsum("nkij, nj -> nki", transition, log_r.mean(0))
        return r
    
    def compute_ig(self, num_samples):
        """ Compute expected information gain """
        # sample observation
        transition = self.rnn.transition
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
        log_pi0 = torch.log(self.pi0 + 1e-6)
        if self.rwd == "efe":
            r = self.compute_efe() + self.beta * log_pi0
        else:
            r = self.compute_ece() + self.beta * log_pi0
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
        batch_size = o.shape[1]
        b, pi = None, None
        if hidden is not None:
            b, pi = hidden

        logp_o = self.obs_model.log_prob(o)
        logp_u = torch.zeros(1, batch_size, self.act_dim).to(self.device) if u is None else self.ctl_model.log_prob(u)
        reward = self.compute_reward()
        alpha_b, alpha_pi = self.rnn(logp_o, logp_u, reward, b)
        return [alpha_b, alpha_pi], [alpha_b, alpha_pi] # second tuple used in bptt
    
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
        _, alpha_pi = hidden
        logp_u = self.ctl_model.mixture_log_prob(alpha_pi, u)
        loss = -torch.sum(logp_u * mask, dim=0) / (mask.sum(0) + 1e-6)

        # compute stats
        nan_mask = mask.clone()
        nan_mask[nan_mask != 0] = 1.
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
        
        # one step transition
        logp_u = self.ctl_model.log_prob(u)
        s_next = self.rnn.predict_one_step(logp_u, alpha_b)
        
        logp_o = self.obs_model.mixture_log_prob(s_next[:-1], o[1:])
        loss = -torch.sum(logp_o * mask[1:], dim=0) / (mask[1:].sum(0) + 1e-6)

        # compute stats
        nan_mask = mask[1:].clone()
        nan_mask[nan_mask != 0] = 1.
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
        b_t, pi_t = self._state["b"], self._state["pi"]
        [alpha_b, alpha_pi], _ = self.forward(
            o.unsqueeze(0), self._prev_ctl, [b_t, pi_t]
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
        self._state["b"] = b_t
        self._state["pi"] = pi_t
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

    def predict(self, o, u, sample_method="ace", num_samples=1):
        """ Offline prediction observations and control """
        [alpha_b, alpha_pi], _ = self.forward(o, u)
        
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


# class VINAgent2(AbstractAgent):
#     """ Non causal Value iteraction network agent using policy as prior with 
#     conditinal gaussian observation and control models and 
#     QMDP hidden layer
#     """
#     def __init__(
#         self, state_dim, act_dim, obs_dim, ctl_dim, rank, horizon, alpha=1., beta=1.,
#         obs_cov="full", ctl_cov="full", use_tanh=False, ctl_lim=None, detach=True
#         ):
#         super().__init__()
#         self.state_dim = state_dim
#         self.act_dim = act_dim
#         self.obs_dim = obs_dim
#         self.ctl_dim = ctl_dim
#         self.horizon = horizon
#         self.alpha = alpha
#         self.beta = beta
#         self.detach = detach
        
#         self.rnn = QMDPLayer2(state_dim, act_dim, rank, horizon, detach=detach)
#         self.obs_model = ConditionalGaussian(
#             obs_dim, state_dim, cov=obs_cov, batch_norm=True, 
#             use_tanh=False, limits=None
#         )
#         self.ctl_model = ConditionalGaussian(
#             ctl_dim, act_dim, cov=ctl_cov, batch_norm=True, 
#             use_tanh=use_tanh, limits=ctl_lim
#         )
#         self.c = nn.Parameter(torch.randn(1, state_dim))
#         self._pi0 = nn.Parameter(torch.randn(1, act_dim, state_dim))
        
#         nn.init.xavier_normal_(self.c, gain=1.)
#         nn.init.xavier_normal_(self._pi0, gain=1.)

#         # self.parameter_size = [
#         #     self.c.shape, self._pi0.shape,
#         #     self.rnn.b0.shape, self.rnn.u.shape, self.rnn.v.shape, self.rnn.w.shape, self.rnn.tau.shape,
#         #     self.obs_model.mu.shape, self.obs_model.lv.shape, self.obs_model.tl.shape,
#         #     self.ctl_model.mu.shape, self.ctl_model.lv.shape, self.ctl_model.tl.shape
#         # ]
    
#     def reset(self):
#         """ Reset internal states for online inference """
#         self._prev_ctl = None
#         self._state = {
#             "a": None, # previous action distribution
#             "b": None, # previous belief distribution
#             "pi": None, # previous policy/action prior
#         }

#     @property
#     def target_dist(self):
#         return torch.softmax(self.c, dim=-1)
    
#     @property
#     def pi0(self):
#         """ Prior policy """
#         return torch.softmax(self._pi0, dim=-2)
    
#     @property
#     def transition(self):
#         return self.rnn.transition
    
#     @property
#     def value(self):
#         value = self.rnn.compute_value(self.transition, self.reward)
#         return value
    
#     @property
#     def policy(self):
#         """ Optimal planned policy """
#         b = torch.eye(self.state_dim)
#         pi = self.rnn.plan(b, self.value)
#         return pi
    
#     @property
#     def passive_dynamics(self):
#         """ Optimal controlled dynamics """
#         policy = self.policy.T.unsqueeze(-1)
#         transition = self.transition.squeeze(0)
#         return torch.sum(transition * policy, dim=-3)
    
#     @property
#     def efe(self):
#         """ Negative expected free energy """
#         entropy = self.obs_model.entropy()
#         c = self.target_dist
#         kl = kl_divergence(torch.eye(self.state_dim), c)
#         return -kl - self.alpha * entropy
    
#     @property
#     def reward(self):
#         """ State action reward """
#         transition = self.rnn.transition
#         entropy = self.obs_model.entropy()

#         if self.detach:
#             transition = transition.data
        
#         c = self.target_dist
#         kl = kl_divergence(transition, c.unsqueeze(-2).unsqueeze(-2))
#         eh = torch.sum(transition * entropy.unsqueeze(-2).unsqueeze(-2), dim=-1).data
#         log_pi0 = torch.log(self.pi0 + 1e-6)
#         r = -kl - self.alpha * eh + self.beta * log_pi0
#         return r

#     def forward(
#         self, o: Tensor, u: Union[Tensor, None], 
#         hidden: Optional[Union[Tuple[Tensor, Tensor], None]]=None, **kwargs
#         ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
#         """ 
#         Args:
#             o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
#             u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
#             hidden ([tuple[torch.tensor, torch.tensor], None], optional). initial hidden state.
        
#         Returns:
#             alpha_b (torch.tensor): state belief distributions. size=[T, batch_size, state_dim]
#             alpha_a (torch.tensor): action predictive distributions. size=[T, batch_size, act_dim]
#         """
#         batch_size = o.shape[1]
#         a, b, pi = None, None, None
#         if hidden is not None:
#             a, b, pi = hidden

#         logp_o = self.obs_model.log_prob(o)
#         logp_u = torch.zeros(1, batch_size, self.act_dim).to(self.device) if u is None else self.ctl_model.log_prob(u)
#         reward = self.reward
#         alpha_a, alpha_b, alpha_pi = self.rnn(logp_o, logp_u, reward, b, pi)
#         return [alpha_a, alpha_b, alpha_pi], [alpha_a, alpha_b, alpha_pi] # second tuple used in bptt
    
#     def act_loss(self, o, u, mask, hidden):
#         """ Compute action loss 
        
#         Args:
#             o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
#             u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
#             mask (torch.tensor): binary mask sequence. size=[T, batch_size]
#             hidden (list): hidden outputs of forward method

#         Returns:
#             loss (torch.tensor): action loss. size=[batch_size]
#             stats (dict): action loss stats
#         """
#         _, _, alpha_pi = hidden
#         logp_u = self.ctl_model.mixture_log_prob(alpha_pi, u)
#         loss = -torch.sum(logp_u * mask, dim=0) / (mask.sum(0) + 1e-6)

#         # compute stats
#         nan_mask = mask.clone()
#         nan_mask[nan_mask != 0] = 1.
#         nan_mask[nan_mask == 0] = torch.nan
#         logp_u_mean = -torch.nanmean((nan_mask * logp_u)).cpu().data
#         stats = {"loss_u": logp_u_mean}
#         return loss, stats
    
#     def obs_loss(self, o, u, mask, hidden):
#         """ Compute observation loss 
        
#         Args:
#             o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
#             u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
#             mask (torch.tensor): binary mask tensor. size=[T, batch_size]
#             hidden (list): hidden outputs of forward method

#         Returns:
#             loss (torch.tensor): observation loss. size=[batch_size]
#             stats (dict): observation loss stats
#         """
#         alpha_a, alpha_b, _ = hidden
        
#         # one step transition
#         transition = self.rnn.transition
#         pi_transition = torch.einsum("nkij, tnk -> tnij", transition, alpha_a.data)
#         s_next = torch.einsum("tnij, tni -> tnj", pi_transition, alpha_b)
        
#         logp_o = self.obs_model.mixture_log_prob(s_next[:-1], o[1:])
#         loss = -torch.sum(logp_o * mask[1:], dim=0) / (mask[1:].sum(0) + 1e-6)

#         # compute stats
#         nan_mask = mask[1:].clone()
#         nan_mask[nan_mask != 0] = 1.
#         nan_mask[nan_mask == 0] = torch.nan
#         logp_o_mean = -torch.nanmean((nan_mask * logp_o)).cpu().data
#         stats = {"loss_o": logp_o_mean}
#         return loss, stats
    
#     def choose_action(self, o, sample_method="ace", num_samples=1):
#         """ Choose action online for a single time step
        
#         Args:
#             o (torch.tensor): observation sequence. size[batch_size, obs_dim]
#             u (torch.tensor): control sequence. size[batch_size, ctl_dim]
#             sample_method (str, optional): sampling method. 
#                 choices=["bma", "ace", "ace"]. Default="ace"
#             num_samples (int, optional): number of samples to draw. Default=1
        
#         Returns:
#             u_sample (torch.tensor): sampled controls. size=[num_samples, batch_size, ctl_dim]
#             logp (torch.tensor): control log probability. size=[num_samples, batch_size]
#         """
#         a_t, b_t, pi_t = self._state["a"], self._state["b"], self._state["pi"]
#         [alpha_a, alpha_b, alpha_pi], _ = self.forward(
#             o.unsqueeze(0), self._prev_ctl, [a_t, b_t, pi_t]
#         )
#         a_t, b_t, pi_t = alpha_a[0], alpha_b[0], alpha_pi[0]
        
#         if sample_method == "bma":
#             u_sample = self.ctl_model.bayesian_average(pi_t)
#         else:
#             sample_mean = True if sample_method == "acm" else False
#             u_sample = self.ctl_model.ancestral_sample(
#                 pi_t.unsqueeze(0), num_samples, sample_mean
#             ).squeeze(-3)
#             logp = self.ctl_model.mixture_log_prob(pi_t, u_sample)
        
#         self._prev_ctl = u_sample.sum(0)
#         self._state["a"] = a_t
#         self._state["b"] = b_t
#         self._state["pi"] = pi_t
#         return u_sample, logp
    
#     def choose_action_batch(self, o, u, sample_method="ace", num_samples=1, tau=0.1, hard=True, return_hidden=False, **kwargs):
#         """ Choose action offline for a batch of sequences 
        
#         Args:
#             o (torch.tensor): observation sequence. size[T, batch_size, obs_dim]
#             u (torch.tensor): control sequence. size[T, batch_size, ctl_dim]
#             sample_method (str, optional): sampling method. 
#                 choices=["bma", "ace", "ace"]. Default="ace"
#             num_samples (int, optional): number of samples to draw. Default=1
#             tau (float, optional): gumbel softmax temperature. Default=0.1
#             hard (bool, optional): if hard use straight-through gradient. Default=True
#             return_hidden (bool, optional): if true return agent hidden state. Default=False

#         Returns:
#             u_sample (torch.tensor): sampled controls. size=[num_samples, T, batch_size, ctl_dim]
#             logp (torch.tensor): control log probability. size=[num_samples, T, batch_size]
#         """
#         [alpha_a, alpha_b, alpha_pi], hidden = self.forward(o, u)

#         if sample_method == "bma":
#             u_sample = self.ctl_model.bayesian_average(alpha_pi)
#         else:
#             sample_mean = True if sample_method == "acm" else False
#             u_sample = self.ctl_model.ancestral_sample(
#                 alpha_pi, num_samples, sample_mean, tau, hard
#             )
#             logp = self.ctl_model.mixture_log_prob(alpha_pi, u_sample)
#         if return_hidden:
#             return u_sample, logp, hidden
#         else:
#             return u_sample, logp

#     def predict(self, o, u, sample_method="ace", num_samples=1):
#         """ Offline prediction observations and control """
#         [alpha_a, alpha_b, alpha_pi], _ = self.forward(o, u)
        
#         # one step transition
#         transition = self.rnn.transition
#         pi_transition = torch.einsum("nkij, tnk -> tnij", transition, alpha_a)
#         s_next = torch.einsum("tnij, tni -> tnj", pi_transition, alpha_b)
        
#         if sample_method == "bma":
#             o_sample = self.obs_model.bayesian_average(s_next)
#             u_sample = self.ctl_model.bayesian_average(alpha_a)

#         else:
#             sample_mean = True if sample_method == "acm" else False
#             o_sample = self.obs_model.ancestral_sample(
#                 s_next, num_samples, sample_mean, tau=0.1, hard=True
#             )
#             u_sample = self.ctl_model.ancestral_sample(
#                 alpha_pi, num_samples, sample_mean, tau=0.1, hard=True
#             )
#         return o_sample, u_sample