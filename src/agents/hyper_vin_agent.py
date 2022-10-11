import torch
import torch.nn as nn
import torch.distributions as torch_dist
from src.agents.core import AbstractAgent
from src.agents.hyper_qmdp_layers import HyperQMDPLayer
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
        alpha=1., beta=1., obs_cov="full", ctl_cov="full", rwd="efe",
        use_tanh=False, ctl_lim=None, train_prior=False
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
        self.hyper_dim = hyper_dim
        
        self.rnn = HyperQMDPLayer(state_dim, act_dim, rank, horizon, hyper_dim)
        self.obs_model = HyperConditionalGaussian(
            obs_dim, state_dim, hyper_dim, cov=obs_cov, batch_norm=True, 
            use_tanh=False, limits=None
        )
        self.ctl_model = ConditionalGaussian(
            ctl_dim, act_dim, cov=ctl_cov, batch_norm=True, 
            use_tanh=use_tanh, limits=ctl_lim
        )
        self._c = nn.Linear(hyper_dim, state_dim)
        self._pi0 = nn.Parameter(torch.randn(1, act_dim, state_dim))
        self._gamma = nn.Linear(hyper_dim, 1)
        
        self._c.weight.data = 0.1 * torch.randn(self._c.weight.data.shape)
        self._gamma.weight.data = 0.1 * torch.randn(self._gamma.weight.data.shape)
        nn.init.xavier_normal_(self._pi0, gain=1.)
        
        # hyper prior
        self.mu = nn.Parameter(torch.zeros(1, hyper_dim), requires_grad=train_prior)
        self.lv = nn.Parameter(torch.zeros(1, hyper_dim), requires_grad=train_prior)

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
            z (torch.tensor): latent variable. size=[1, hyper_dim]
        """
        if z is None:
            z = self.sample_z()
            
        self._prev_ctl = None # previous control
        self._state = {
            "b": None, # previous belief distribution
            "pi": None, # previous policy/action prior
            "z": z, # hyper vector
        }

    @property
    def target_dist(self):
        z = torch.zeros(1, self.hyper_dim).to(self.device)
        return self.compute_target_dist(z)
    
    @property
    def pi0(self):
        """ Prior policy """
        z = torch.zeros(1, self.hyper_dim).to(self.device)
        return self.compute_prior_policy(z)
    
    @property
    def transition(self):
        z = torch.zeros(1, self.hyper_dim).to(self.device)
        return self.rnn.compute_transition(z)
    
    @property
    def value(self):
        value = self.rnn.compute_value(self.transition, self.reward)
        return value
    
    @property
    def policy(self):
        """ Optimal planned policy """
        z = torch.zeros(1, self.hyper_dim).to(self.device)
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
        z = torch.zeros(1, self.hyper_dim).to(self.device)
        entropy = self.obs_model.entropy(z)
        c = self.target_dist
        kl = kl_divergence(torch.eye(self.state_dim), c)
        return -kl - entropy
    
    @property
    def reward(self):
        z = torch.zeros(1, self.hyper_dim).to(self.device)
        return self.compute_reward(z)
    
    def parameter_entropy(self, z):
        eps = 1e-6
        c_ent = torch.log(torch.abs(self._c.weight) + eps).sum() / self.hyper_dim
        pi0_ent = torch.log(torch.abs(self._pi0.weight) + eps).sum() / self.hyper_dim
        rnn_ent = self.rnn.parameter_entropy(z)
        obs_ent = self.obs_model.parameter_entropy(z)

        ent = c_ent + self.beta * pi0_ent + rnn_ent + obs_ent
        return ent

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
    
    def compute_policy(self, z):        
        b = torch.eye(self.state_dim).to(self.device)
        pi = self.rnn.plan(b, z, self.value)
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
        return torch_dist.Normal(self.mu, rectify(self.lv))        

    def sample_z(self, sample_shape=torch.Size()):
        """ Sample from latent prior """
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
        hidden: Optional[Union[Tuple[Tensor, Tensor], None]]=None,
        detach: Optional[bool]=False
        ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """ 
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            z (torch.tensor): latent variable. size=[batch_size, hyper_dim]
            hidden ([tuple[torch.tensor] * 2, None], optional). initial hidden state.
            detach (bool, optional): whether to detach model training. Default=False
        
        Returns:
            alpha_b (torch.tensor): state belief distributions. size=[T, batch_size, state_dim]
            alpha_a (torch.tensor): action predictive distributions. size=[T, batch_size, act_dim]
        """
        batch_size = o.shape[1]
        b, pi = None, None
        if hidden is not None:
            b, pi = hidden

        logp_o = self.obs_model.log_prob(o, z)
        logp_u = torch.zeros(1, batch_size, self.act_dim).to(self.device) if u is None else self.ctl_model.log_prob(u)
        reward = self.compute_reward(z)
        alpha_b, alpha_pi = self.rnn.forward(logp_o, logp_u, reward, z, b, detach=detach)
        return [alpha_b, alpha_pi], [alpha_b, alpha_pi] # second tuple used in bptt
    
    def act_loss(self, o, u, z, mask, hidden):
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
        _, alpha_pi = hidden
        logp_u = self.ctl_model.mixture_log_prob(alpha_pi, u)
        loss = -torch.sum(logp_u * mask, dim=0) / (mask.sum(0) + 1e-6) #+ torch.mean(kl)/len(o)

        # compute stats
        nan_mask = mask.clone()
        nan_mask[nan_mask != 0] = 1.
        nan_mask[nan_mask == 0] = torch.nan
        logp_u_mean = -torch.nanmean((nan_mask * logp_u)).cpu().data
        stats = {"loss_u": logp_u_mean}
        return loss, stats
    
    def obs_loss(self, o, u, z, mask, hidden):
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
        
        """ TODO: move this to rnn updates """
        # one step transition
        logp_u = self.ctl_model.log_prob(u)
        s_next = self.rnn.predict_one_step(logp_u, alpha_b, z)

        logp_o = self.obs_model.mixture_log_prob(s_next[:-1], o[1:], z)
        loss = -torch.sum(logp_o * mask[1:], dim=0) / (mask[1:].sum(0) + 1e-6)
        
        # compute stats
        nan_mask = mask[1:].clone()
        nan_mask[nan_mask !=0] = 1.
        nan_mask[nan_mask == 0] = torch.nan
        logp_o_mean = -torch.nanmean((nan_mask * logp_o)).cpu().data
        stats = {"loss_o": logp_o_mean}
        return loss, stats
    
    def choose_action(self, o, sample_method="ace", num_samples=1):
        """ Choose action online for a single time step
        
        Args:
            o (torch.tensor): observation sequence. size[batch_size, obs_dim]
            sample_method (str, optional): sampling method. 
                choices=["bma", "ace", "ace"]. Default="ace"
            num_samples (int, optional): number of samples to draw. Default=1
        
        Returns:
            u_sample (torch.tensor): sampled controls. size=[num_samples, batch_size, ctl_dim]
            logp (torch.tensor): control log probability. size=[num_samples, batch_size]
        """
        b_t, pi_t = self._state["b"], self._state["pi"]
        [alpha_b, alpha_pi], _ = self.forward(
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
            z (torch.tensor): latent variable. size=[batch_size, hyper_dim]
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
        if z is None:
            z = self.sample_z(o.shape[1])

        [alpha_b, alpha_a], hidden = self.forward(o, u, z)

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


# class HyperVINAgent(AbstractAgent):
#     """ Hyper value iteraction network agent with 
#     conditinal gaussian observation and control models and 
#     QMDP hidden layer
#     """
#     def __init__(
#         self, state_dim, act_dim, obs_dim, ctl_dim, rank, horizon,
#         hyper_dim, hidden_dim, num_hidden, gru_layers, activation,
#         alpha=1., beta=1., obs_cov="full", ctl_cov="full", 
#         use_tanh=False, ctl_lim=None, 
#         ):
#         super().__init__()
#         self.state_dim = state_dim
#         self.act_dim = act_dim
#         self.obs_dim = obs_dim
#         self.ctl_dim = ctl_dim
#         self.horizon = horizon
#         self.alpha = alpha # obs entropy temperature
#         self.beta = beta # policy prior temperature
#         self.hyper_dim = hyper_dim
        
#         self.rnn = HyperQMDPLayer(state_dim, act_dim, rank, horizon, hyper_dim)
#         # self.obs_model = HyperConditionalGaussian(
#         #     obs_dim, state_dim, hyper_dim, cov=obs_cov, batch_norm=True, 
#         #     use_tanh=False, limits=None
#         # )
        
#         # use shared obs model for now
#         self.obs_model = ConditionalGaussian(
#             obs_dim, state_dim, cov=obs_cov, batch_norm=True, 
#             use_tanh=False, limits=None
#         )
#         self.ctl_model = ConditionalGaussian(
#             ctl_dim, act_dim, cov=ctl_cov, batch_norm=True, 
#             use_tanh=use_tanh, limits=ctl_lim
#         )
#         self._c = nn.Linear(hyper_dim, state_dim)
#         self._pi0 = nn.Linear(hyper_dim, act_dim * state_dim)
        
#         # hyper prior
#         self.mu = nn.Parameter(torch.randn(1, hyper_dim))
#         self.lv = nn.Parameter(torch.randn(1, hyper_dim))

#         self.encoder = GRUMLP(
#             input_dim=obs_dim + ctl_dim,
#             output_dim=hyper_dim * 2,
#             hidden_dim=hidden_dim,
#             gru_layers=gru_layers,
#             mlp_layers=num_hidden,
#             activation=activation
#         )

#         self.obs_mean = nn.Parameter(torch.zeros(obs_dim), requires_grad=False)
#         self.obs_variance = nn.Parameter(torch.ones(obs_dim), requires_grad=False)
    
#     def reset(self):
#         """ Reset internal states for online inference """
#         self._prev_ctl = None # previous control
#         # self._b = None # prior belief
#         # self._a = None # previous action distribution
#         # self._z = None # hyper vector
#         # self._ent = None # hyper posterior entropy
#         self._state = {
#             "z": None, # hyper vector
#             "ent": None, # hyper posterior entropy
#             "b": None, # previous belief distribution
#             "pi": None, # previous policy/action prior
#         }

#     @property
#     def target_dist(self):
#         z = torch.ones(1, self.hyper_dim).to(self.device)
#         return self.compute_target_dist(z)
    
#     @property
#     def pi0(self):
#         """ Prior policy """
#         z = torch.ones(1, self.hyper_dim).to(self.device)
#         return self.compute_prior_policy(z)
    
#     @property
#     def transition(self):
#         z = torch.ones(1, self.hyper_dim).to(self.device)
#         return self.rnn.compute_transition(z)
    
#     @property
#     def value(self):
#         value = self.rnn.compute_value(self.transition, self.reward)
#         return value
    
#     @property
#     def policy(self):
#         """ Optimal planned policy """
#         z = torch.ones(1, self.hyper_dim).to(self.device)
#         b = torch.eye(self.state_dim).to(self.device)
#         pi = self.rnn.plan(b, z, self.value)
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
#         z = torch.ones(1, self.hyper_dim).to(self.device)
#         # entropy = self.obs_model.entropy(z)
#         entropy = self.obs_model.entropy()
#         c = self.target_dist
#         kl = kl_divergence(torch.eye(self.state_dim), c)
#         return -kl - entropy
    
#     @property
#     def reward(self):
#         z = torch.ones(1, self.hyper_dim).to(self.device)
#         return self.compute_reward(z)
    
#     def compute_target_dist(self, z):
#         return torch.softmax(self._c(z), dim=-1)
    
#     def compute_prior_policy(self, z):
#         pi0 = self._pi0(z).view(-1, self.act_dim, self.state_dim)
#         return torch.softmax(pi0, dim=-2)

#     def compute_reward(self, z):
#         """ State action reward """
#         transition = self.rnn.compute_transition(z)
#         # entropy = self.obs_model.entropy(z)
#         entropy = self.obs_model.entropy()

#         c = self.compute_target_dist(z)
#         pi0 = self.compute_prior_policy(z)
#         kl = kl_divergence(transition, c.unsqueeze(-2).unsqueeze(-2))
#         eh = torch.sum(transition * entropy.unsqueeze(-2).unsqueeze(-2), dim=-1).data
#         log_pi0 = torch.log(pi0 + 1e-6)
#         r = -kl - self.alpha * eh + self.beta * log_pi0
#         return r

#     def encode(self, o, u):
#         """ Sample from variational posterior """
#         o_norm = (o - self.obs_mean) / self.obs_variance**0.5
#         z_params = self.encoder(torch.cat([o_norm, u], dim=-1))
#         mu, lv = torch.chunk(z_params, 2, dim=-1)
        
#         z_dist = torch_dist.Normal(mu, rectify(lv))
#         z = z_dist.rsample()
        
#         prior_dist = torch_dist.Normal(self.mu, rectify(self.lv))
#         kl = torch_dist.kl.kl_divergence(z_dist, prior_dist).sum(-1, keepdim=True)
#         return z, kl
    
#     def sample_z(self):
#         z = torch_dist.Normal(
#             self.mu, rectify(self.lv)
#         ).rsample()
#         return z

#     def forward(
#         self, o: Tensor, u: Union[Tensor, None], 
#         hidden: Optional[Union[Tuple[Tensor, Tensor, Tensor, Tensor], None]]=None,
#         sample_z: Optional[bool]=False
#         ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
#         """ 
#         Args:
#             o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
#             u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
#             hidden ([tuple[torch.tensor] * 4, None], optional). initial hidden state.
#             sample_z (bool, optional): whether to sample z from prior. If not encode observations. 
        
#         Returns:
#             alpha_b (torch.tensor): state belief distributions. size=[T, batch_size, state_dim]
#             alpha_a (torch.tensor): action predictive distributions. size=[T, batch_size, act_dim]
#             z (torch.tensor): agent parameter vector. size=[batch_size, num_params]
#             ent (torch.tensor): variational distribution entropy. size=[batch_size, 1]
#         """
#         b, a, z, ent = None, None, None, None
#         if hidden is not None:
#             b, a, z, ent = hidden
        
#         if z is None:
#             if sample_z:
#                 z = self.sample_z()
#             else:
#                 z, ent = self.encode(o, u)

#         # logp_o = self.obs_model.log_prob(o, z)
#         logp_o = self.obs_model.log_prob(o)
#         logp_u = None if u is None else self.ctl_model.log_prob(u)
#         reward = self.compute_reward(z)
#         alpha_b, alpha_a = self.rnn(logp_o, logp_u, reward, z, b, a)
#         return [alpha_b, alpha_a], [alpha_b, alpha_a, z, ent] # second tuple used in bptt
    
#     def act_loss(self, o, u, mask, hidden):
#         """ Compute action loss 
        
#         Args:
#             o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
#             u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
#             mask (torch.tensor): binary mask sequence. size=[T, batch_size]
#             hidden (list): outputs of forward method

#         Returns:
#             loss (torch.tensor): action loss. size=[batch_size]
#             stats (dict): action loss stats
#         """
#         _, alpha_a, _, kl = hidden
#         logp_u = self.ctl_model.mixture_log_prob(alpha_a, u)
#         loss = -torch.sum(logp_u * mask, dim=0) / (mask.sum(0) + 1e-6) + torch.mean(kl)/len(o)
        
#         # compute stats
#         nan_mask = mask.clone()
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
#             forward_out (list): outputs of forward method

#         Returns:
#             loss (torch.tensor): observation loss. size=[batch_size]
#             stats (dict): observation loss stats
#         """
#         alpha_b, _, z, _ = hidden
#         # logp_o = self.obs_model.mixture_log_prob(alpha_b, o, z)
#         logp_o = self.obs_model.mixture_log_prob(alpha_b, o)
#         loss = -torch.sum(logp_o * mask, dim=0) / (mask.sum(0) + 1e-6)
        
#         # compute stats
#         nan_mask = mask.clone()
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
#         [alpha_b, alpha_a], [_, _, z, _] = self.forward(
#             o.unsqueeze(0), self._prev_ctl, [self._b, self._a, self._z, self._ent], 
#             sample_z=True
#         )
#         b_t, a_t = alpha_b[0], alpha_a[0]
        
#         if sample_method == "bma":
#             u_sample = self.ctl_model.bayesian_average(a_t)
#         else:
#             sample_mean = True if sample_method == "acm" else False
#             u_sample = self.ctl_model.ancestral_sample(
#                 a_t.unsqueeze(0), num_samples, sample_mean
#             ).squeeze(-3)
#             logp = self.ctl_model.mixture_log_prob(a_t, u_sample)
        
#         self._b, self._a = b_t, a_t
#         self._prev_ctl = u_sample.sum(0)
#         self._z = z
#         return u_sample, logp
    
#     def choose_action_batch(
#         self, o, u, sample_method="ace", num_samples=1, tau=0.1, 
#         hard=True, sample_z=False, return_hidden=False
#         ):
#         """ Choose action offline for a batch of sequences 
        
#         Args:
#             o (torch.tensor): observation sequence. size[T, batch_size, obs_dim]
#             u (torch.tensor): control sequence. size[T, batch_size, ctl_dim]
#             sample_method (str, optional): sampling method. 
#                 choices=["bma", "ace", "ace"]. Default="ace"
#             num_samples (int, optional): number of samples to draw. Default=1
#             tau (float, optional): gumbel softmax temperature. Default=0.1
#             hard (bool, optional): if hard use straight-through gradient. Default=True
#             sample_z (bool, optional): whether to sample hyper vector. Default=False
#             return_hidden (bool, optional): if true return agent hidden state. Default=False

#         Returns:
#             u_sample (torch.tensor): sampled controls. size=[num_samples, T, batch_size, ctl_dim]
#             logp (torch.tensor): control log probability. size=[num_samples, T, batch_size]
#         """
#         [alpha_b, alpha_a], hidden = self.forward(o, u, sample_z=sample_z)

#         if sample_method == "bma":
#             u_sample = self.ctl_model.bayesian_average(alpha_a)
#         else:
#             sample_mean = True if sample_method == "acm" else False
#             u_sample = self.ctl_model.ancestral_sample(
#                 alpha_a, num_samples, sample_mean, tau, hard
#             )
#             logp = self.ctl_model.mixture_log_prob(alpha_a, u_sample)
        
#         if return_hidden:
#             return u_sample, logp, hidden
#         else:
#             return u_sample, logp