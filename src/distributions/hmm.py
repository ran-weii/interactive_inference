import torch
import torch.nn as nn
import torch.jit as jit
from src.distributions.nn_models import Model
from src.distributions.mixture_models import ConditionalGaussian
from src.distributions.transition_models import DiscreteMC
# from src.distributions.mixture_models import ConditionalGaussian, EmbeddedConditionalGaussian
# from src.distributions.transition_models import DiscreteMC, EmbeddedDiscreteMC

import math
from typing import Union, List, Tuple
from torch import Tensor

class ContinuousGaussianHMM(Model):
    """Input-output hidden markov model with:
        discrete transitions, continuous actions, gaussian observations
    """
    def __init__(
        self, state_dim, act_dim, obs_dim, ctl_dim, rank,
        obs_cov="full", ctl_cov="full", use_tanh=False, ctl_lim=None
        ):
        """
        Args:
            state_dim (int): state dimension
            act_dim (int): discrete action dimension
            obs_dim (int): observation dimension
            ctl_dim (int): control dimension
            rank (int): transition matrix rank
            obs_cov (str): observation covariance type. choices=["full", "diag], default="full"
            ctl_cov (str): control covariance type. choices=["full", "diag], default="full"
            use_tanh (bool, optional):
            ctl_lim (torch.tensor, optional):
        """
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.ctl_dim = ctl_dim
        self.eps = 1e-6
        
        self.obs_model = ConditionalGaussian(
            obs_dim, state_dim, cov=obs_cov, batch_norm=True, 
            use_tanh=False, limits=None
        )
        self.transition_model = DiscreteMC(state_dim, act_dim, rank)
        self.ctl_model = ConditionalGaussian(
            ctl_dim, act_dim, cov=ctl_cov, batch_norm=True, 
            use_tanh=use_tanh, limits=ctl_lim
        )
        self.pi0 = nn.Parameter(torch.randn(1, state_dim, act_dim))
        nn.init.xavier_normal_(self.pi0, gain=1.)
    
    @property
    def prior_policy(self):
        return torch.softmax(self.pi0 + self.eps, dim=-1)
    
    def obs_entropy(self):
        return self.obs_model.entropy()

    def obs_log_prob(self, x):
        return self.obs_model.log_prob(x)

    def ctl_log_prob(self, u):
        return self.ctl_model.log_prob(u)
    
    def obs_mixture_log_prob(self, pi, x):
        return self.obs_model.mixture_log_prob(pi, x)

    def ctl_mixture_log_prob(self, pi, u):
        return self.ctl_model.mixture_log_prob(pi, u)
    
    def obs_ancestral_sample(self, z, num_samples):
        return self.obs_model.ancestral_sample(z, num_samples)
    
    def ctl_ancestral_sample(self, z, num_samples):
        return self.ctl_model.ancestral_sample(z, num_samples)

    def alpha(self, b, x, u=None, a=None, logp_x=None, logp_u=None):
        """ Compute forward message for a single time step

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
            b_t (torch.tensor): state posterior. size=[batch_size, state_dim]
            a_t (torch.tensor): action posterior, return None if u is None. 
                size=[batch_size, act_dim]
        """
        # compute state likelihood
        if u is None:
            logp_z = torch.log(self.transition_model.initial_state + self.eps)
            logp_z = logp_z * torch.ones_like(b).to(self.device)
            a_t = None
        else:
            if a is None:
                pi0 = self.prior_policy
                a = torch.sum(b.unsqueeze(-1) * pi0, dim=-1)
            a_t = self.ctl_model.infer(a, u, logp_u)
            logp_z = torch.log(self.transition_model._forward(b, a_t) + self.eps)
        
        if logp_x is None:
            logp_x = self.obs_model.log_prob(x)
        b_t = torch.softmax(logp_x + logp_z, dim=-1)
        return b_t, a_t
    
    def _forward(self, x, u):
        """ Forward algorithm

        Args:
            x (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]

        Returns:
            alpha_b (torch.tensor): state forward messages. size=[T, batch_size, state_dim]
            alpha_a (torch.tensor): action forward messages. size=[T-1, batch_size, act_dim]
        """
        batch_size = x.shape[1]
        T = x.shape[0]
        
        logp_x = self.obs_model.log_prob(x) # supplying this increase test likelihood
        logp_u = self.ctl_model.log_prob(u) # supplying this increase test likelihood
        alpha_b = [torch.empty(0)] * (T + 1)
        alpha_b[0] = torch.ones(batch_size, self.state_dim).to(self.device) # filler initial belief
        alpha_a = [torch.empty(0)] * (T)
        for t in range(T):
            x_t = x[t]
            u_t = None if t == 0 else u[t-1]
            logp_x_t = logp_x[t]
            logp_u_t = None if t == 0 else logp_u[t-1]
            alpha_b[t+1], alpha_a[t] = self.alpha(
                alpha_b[t], x_t, u_t, logp_x=logp_x_t, logp_u=logp_u_t
            )
        
        alpha_b = torch.stack(alpha_b)[1:]
        alpha_a = torch.stack(alpha_a[1:])
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
        alpha_b, alpha_a = self._forward(x, u)
        
        batch_size = x.shape[1]
        if prior:
            z0 = self.transition_model.initial_state
            z0 = z0 * torch.ones(batch_size, 1).to(self.device)
            z = torch.cat([z0.unsqueeze(0), alpha_b[:-1]], dim=0)
        else:
            z = alpha_b
        
        if not inference:
            logp_o = self.obs_model.mixture_log_prob(z, x)
            logp_u = self.ctl_model.mixture_log_prob(alpha_a, u[:-1])
            return logp_o, logp_u
        else:
            # model average prediction
            if sample_method == "ace":
                x_sample = self.obs_model.ancestral_sample(z, num_samples)
                u_sample = self.ctl_model.ancestral_sample(alpha_a, num_samples)
            else:
                x_sample = self.obs_model.bayesian_average(z)
                u_sample = self.ctl_model.bayesian_average(alpha_a)
            
            return x_sample, u_sample, alpha_b, alpha_a


class QMDPLayer(jit.ScriptModule):
    def __init__(self, state_dim, act_dim, rank, horizon):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.rank = rank
        self.horizon = horizon
        self.eps = 1e-6
        
        self.b0 = nn.Parameter(torch.randn(1, state_dim))
        self.u = nn.Parameter(torch.randn(1, rank, state_dim)) # source tensor
        self.v = nn.Parameter(torch.randn(1, rank, state_dim)) # sink tensor
        self.w = nn.Parameter(torch.randn(1, rank, act_dim)) # action tensor 
        self.tau = nn.Parameter(torch.randn(1, 1))
        
        nn.init.xavier_normal_(self.b0, gain=1.)
        nn.init.xavier_normal_(self.u, gain=1.)
        nn.init.xavier_normal_(self.v, gain=1.)
        nn.init.xavier_normal_(self.w, gain=1.)
        nn.init.uniform_(self.tau, a=-1, b=1)
    
    def __repr__(self):
        s = "{}(state_dim={}, act_dim={}, rank={}, horizon={})".format(
            self.__class__.__name__, self.state_dim, self.act_dim, self.rank, self.horizon
        )
        return s

    @property
    def transition(self):
        """ Return transition matrix. size=[1, act_dim, state_dim, state_dim] """
        w = torch.einsum("nri, nrj, nrk -> nkij", self.u, self.v, self.w)
        return torch.softmax(w, dim=-1)
    
    @jit.script_method
    def compute_value(self, transition: Tensor, reward: Tensor) -> Tensor:
        """ Compute expected value using value iteration

        Args:
            transition (torch.tensor): transition matrix. size=[batch_size, act_dim, state_dim, state_dim]
            reward (torch.tensor): reward matrix. size=[batch_size, act_dim, state_dim]
        
        Returns:
            q (torch.tensor): state q value. size=[horizon, batch_size, act_dim, state_dim]
        """        
        q = [torch.empty(0)] * (self.horizon + 1)
        q[0] = reward
        for t in range(self.horizon):
            v_next = reward + torch.logsumexp(q[t], dim=-2, keepdim=True)
            q[t+1] = torch.einsum("nkij, nkj -> nki", transition, v_next)
        return torch.stack(q[1:])
    
    @jit.script_method
    def plan(self, b: Tensor, value: Tensor) -> Tensor:
        """ Compute the belief action distribution 
        
        Args:
            b (torch.tensor): current belief. size=[batch_size, state_dim]
            value (torch.tensor): state q value. size=[horizon, batch_size, act_dim, state_dim]

        Returns:
            a (torch.tensor): action distribution. size=[batch_size, act_dim]
        """
        tau = torch.exp(self.tau.clip(math.log(1e-6), math.log(1e3)))
        tau = poisson_pdf(tau, self.horizon)
        if tau.shape[0] != b.shape[0]:
            tau = torch.repeat_interleave(tau, b.shape[0], 0)
        
        a = torch.softmax(torch.einsum("ni, hnki -> hnk", b, value), dim=-1)
        a = torch.einsum("hnk, nh -> nk", a, tau)
        return a

    @jit.script_method
    def update_belief(self, logp_o: Tensor, transition: Tensor, b: Tensor, a: Tensor) -> Tensor:
        """ Compute state posterior
        
        Args:
            logp_o (torch.tensor): log probability of current observation. size=[batch_size, state_dim]
            transition (torch.tensor): transition matrix. size=[batch_size, act_dim, state_dim, state_dim]
            b (torch.tensor): prior belief. size=[batch_size, state_dim]
            a (torch.tensor): action posterior. size=[batch_size, act_dim]

        Returns:
            b_post (torch.tensor): state posterior. size=[batch_size, state_dim]
        """
        s_next = torch.einsum("nkij, ni, nk -> nj", transition, b, a)
        logp_s = torch.log(s_next + self.eps)
        b_post = torch.softmax(logp_s + logp_o, dim=-1)
        return b_post
    
    @jit.script_method
    def update_action(self, logp_u: Tensor, a: Tensor) -> Tensor:
        """ Compute action posterior 
        
        Args:
            logp_u (torch.tensor): log probability of previous control. size=[batch_size, act_dim]
            a (torch.tensor): action prior. size=[batch_size, act_dim]

        Returns:
            a_post (torch.tensor): action posterior. size=[batch_size, act_dim]
        """ 
        logp_a = torch.log(a + self.eps)
        a_post = torch.softmax(logp_a + logp_u, dim=-1)
        return a_post
    
    @jit.script_method
    def update_cell(
        self, logp_o: Tensor, logp_u: Tensor, b: Tensor, a: Tensor, 
        transition: Tensor, value: Tensor
        ) -> Tuple[Tensor, Tensor]:
        """ Compute action and state posterior. Then compute the next action distribution """
        a_post = self.update_action(logp_u, a)
        b_post = self.update_belief(logp_o, transition, b, a_post)
        a_next = self.plan(b_post, value)
        return (b_post, a_next)
    
    @jit.script_method
    def init_hidden(self, logp_o: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        b0 = torch.softmax(self.b0, dim=-1)
        logp_s = torch.log(b0 + self.eps)
        b = torch.softmax(logp_s + logp_o, dim=-1)
        a = self.plan(b, value)
        return b, a
    
    def forward(
        self, logp_o: Tensor, logp_u: Union[Tensor, None], reward: Tensor,
        b: Union[Tensor, None], a: Union[Tensor, None]
        ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            logp_o (torch.tensor): sequence of observation probabilities. size=[T, batch_size, state_dim]
            logp_u (torch.tensor): sequence of control probabilities. Should be offset -1 if b is None.
                size=[T, batch_size, act_dim]
            reward (torch.tensor): reward matrix. size=[batch_size, act_dim, state_dim]
            b ([torch.tensor, None], optional): prior belief. size=[batch_size, state_dim]
            a ([torch.tensor, None], optional): prior action. size=[batch_size, act_dim]
        
        Returns:
            alpha_b (torch.tensor): sequence of posterior belief. size=[T, batch_size, state_dim]
            alpha_a (torch.tensor): sequence of action distribution. size=[T, batch_size, act_dim]
        """
        transition = self.transition
        value = self.compute_value(transition, reward)
        t_offset = 0 if b is not None else -1 # offset for offline prediction
        T = len(logp_o)

        alpha_b = [b] + [torch.empty(0)] * (T)
        alpha_a = [a] + [torch.empty(0)] * (T)
        for t in range(T):
            if alpha_b[t] is None:
                alpha_b[t+1], alpha_a[t+1] = self.init_hidden(logp_o[t], value)
            else:
                alpha_b[t+1], alpha_a[t+1] = self.update_cell(
                    logp_o[t], logp_u[t+t_offset], 
                    alpha_b[t], alpha_a[t], transition, value
                )
        return torch.stack(alpha_b[1:]), torch.stack(alpha_a[1:])

    # @jit.script_method
    # def forward(
    #     self, logp_o: Tensor, logp_u: Tensor, b: Tensor, a: Tensor, reward: Tensor
    #     ) -> Tuple[Tensor, Tensor]:
    #     transition = self.transition
    #     value = self.compute_value(transition, reward)
    #     T = len(logp_o)
        
    #     alpha_b = [b] + [torch.empty(0)] * (T)
    #     alpha_a = [a] + [torch.empty(0)] * (T)
    #     for t in range(T):
    #         alpha_b[t+1], alpha_a[t+1] = self.update_cell(
    #             logp_o[t], logp_u[t], alpha_b[t], alpha_a[t], transition, value
    #         )
    #     return torch.stack(alpha_b[1:]), torch.stack(alpha_a[1:])


def poisson_pdf(rate: Tensor, K: int) -> Tensor:
    """ 
    Args:
        rate (torch.tensor): poission arrival rate [batch_size, 1]
        K (int): number of bins

    Returns:
        pdf (torch.tensor): truncated poisson pdf [batch_size, K]
    """
    Ks = 1 + torch.arange(K).to(rate.device)
    poisson_logp = Ks.xlogy(rate) - rate - (Ks + 1).lgamma()
    pdf = torch.softmax(poisson_logp, dim=-1)
    return pdf

# class EmbeddedContinuousGaussianHMM(Model):
#     """ Input-output hidden markov model with:
#         discrete transitions, continuous actions, gaussian observations
#         observation and transition models use embeddings
#     """
#     def __init__(
#         self, state_dim, act_dim, obs_dim, ctl_dim, state_embed_dim, act_embed_dim, 
#         rank=0, obs_cov="full", ctl_cov="full"):
#         """
#         Args:
#             state_dim (int): state dimension
#             act_dim (int): discrete action dimension
#             obs_dim (int): observation dimension
#             ctl_dim (int): control dimension
#             state_embed_dim (int): state embedding dimension
#             act_embed_dim (int): action embedding dimension
#             rank (int): transition model core rank. Default=0
#             obs_cov (str): observation covariance type. choices=["full", "diag], default="full"
#             ctl_cov (str): control covariance type. choices=["full", "diag], default="full"
#         """
#         super().__init__()
#         self.state_dim = state_dim
#         self.act_dim = act_dim
#         self.obs_dim = obs_dim
#         self.ctl_dim = ctl_dim
#         self.eps = 1e-6
        
#         self.obs_model = EmbeddedConditionalGaussian(
#             obs_dim, state_dim, state_embed_dim, cov=obs_cov, batch_norm=True
#         )
#         self.transition_model = EmbeddedDiscreteMC(
#             state_dim, act_dim, state_embed_dim, act_embed_dim, rank=rank
#         )
#         self.ctl_model = EmbeddedConditionalGaussian(
#             ctl_dim, act_dim, act_embed_dim, cov=ctl_cov, batch_norm=True
#         )
#         self.act_prior = nn.Parameter(torch.randn(state_dim, act_dim))
#         nn.init.xavier_normal_(self.act_prior, gain=1.)
    
#     @property
#     def state_embedding(self):
#         return self.transition_model.w.state_embedding

#     @property
#     def act_embedding(self):
#         return self.transition_model.w.act_embedding

#     def get_initial_state(self):
#         return self.transition_model.get_initial_state()

#     def get_transition_matrix(self, a):
#         """
#         Args:
#             a (torch.tensor): action vector. size=[batch_size, act_dim]

#         Returns:
#             transition_matrix (torch.tensor): action conditioned transition matrix.
#                 size=[batch_size, state_dim, state_dim]
#         """
#         a_ = a.unsqueeze(-1).unsqueeze(-1)
#         transition_matrix = self.transition_model.get_transition_matrix()
#         transition_matrix = torch.sum(transition_matrix * a_, dim=-3)
#         return transition_matrix
    
#     def obs_entropy(self):
#         return self.obs_model.entropy(self.state_embedding)

#     def obs_log_prob(self, x):
#         return self.obs_model.log_prob(self.state_embedding, x)

#     def ctl_log_prob(self, u):
#         return self.ctl_model.log_prob(self.act_embedding, u)
    
#     def obs_mixture_log_prob(self, pi, x):
#         return self.obs_model.mixture_log_prob(pi, self.state_embedding, x)

#     def ctl_mixture_log_prob(self, pi, u):
#         return self.ctl_model.mixture_log_prob(pi, self.act_embedding, u)

#     def alpha(self, b, x, u=None, a=None, logp_x=None, logp_u=None):
#         """ Compute forward message for a single time step

#         Args:
#             b (torch.tensor): prior belief. size=[batch_size, state_dim]
#             x (torch.tensor): observation vector. size=[batch_size, obs_dim]
#             u (torch.tensor, None, optional): control vector. size=[batch_size, ctl_dim]. 
#                 None for initial state. Default=None
#             a (torch.tensor, None, optional): action prior, to be supplied by a planner.
#                 size=[batch_size, act_dim]. Default=None
#             logp_x (torch.tensor, None, optional): observation likelihood. Supplied during training. 
#                 size=[batch_size, state_dim]
#             logp_u (torch.tensor, None, optional): control likelihood. Supplied during training. 
#                 size=[batch_size, act_dim]

#         Returns:
#             b_t (torch.tensor): state posterior. size=[batch_size, state_dim]
#             a_t (torch.tensor): action posterior, return None if u is None. 
#                 size=[batch_size, act_dim]
#         """
#         # compute state likelihood
#         if u is None:
#             logp_z = torch.log(self.get_initial_state() + self.eps)
#             logp_z = logp_z * torch.ones_like(b).to(self.device)
#             a_t = None
#         else:
#             if a is None:
#                 a = torch.softmax(self.act_prior + self.eps, dim=-1).unsqueeze(0)
#                 a = torch.sum(b.unsqueeze(-1) * a, dim=-2)
#             a_t = self.ctl_model.infer(a, u, self.act_embedding, logp_u)
#             transition = self.get_transition_matrix(a_t)
#             logp_z = torch.sum(transition * b.unsqueeze(-1) + self.eps, dim=-2).log()
        
#         if logp_x is None:
#             logp_x = self.obs_model.log_prob(self.state_embedding, x)
#         b_t = torch.softmax(logp_x + logp_z, dim=-1)
#         return b_t, a_t
    
#     def _forward(self, x, u):
#         """ Forward algorithm

#         Args:
#             x (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
#             u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]

#         Returns:
#             alpha_b (torch.tensor): state forward messages. size=[T, batch_size, state_dim]
#             alpha_a (torch.tensor): action forward messages. size=[T-1, batch_size, act_dim]
#         """
#         batch_size = x.shape[1]
#         T = x.shape[0]
        
#         logp_x = self.obs_model.log_prob(self.state_embedding, x) # supplying this increase test likelihood
#         logp_u = self.ctl_model.log_prob(self.act_embedding, u) # supplying this increase test likelihood
#         alpha_b = [torch.empty(0)] * (T + 1)
#         alpha_b[0] = torch.ones(batch_size, self.state_dim).to(self.device) # filler initial belief
#         alpha_a = [torch.empty(0)] * (T)
#         for t in range(T):
#             x_t = x[t]
#             u_t = None if t == 0 else u[t-1]
#             logp_x_t = logp_x[t]
#             logp_u_t = None if t == 0 else logp_u[t-1]
#             alpha_b[t+1], alpha_a[t] = self.alpha(
#                 alpha_b[t], x_t, u_t, logp_x=logp_x_t, logp_u=logp_u_t
#             )
        
#         alpha_b = torch.stack(alpha_b[1:])
#         alpha_a = torch.stack(alpha_a[1:])
#         return alpha_b, alpha_a

#     def predict(self, x, u, prior=True, inference=True, sample_method="ace", num_samples=1):
#         """ Predict observations and controls

#         Args:
#             x (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
#             a (torch.tensor): action sequence. size=[T, batch_size, ctl_dim]
#             prior (bool, optional): whether to use prior predictive. 
#                 If false use posterior predictive. Default=True
#             inference (bool, optional): inference mode return likelihood. 
#                 If false return samples. Default=True
#             sample_method (str, optional): sampling method. choices=["ace", "bma"], Default="ace"
#             num_samples (int, optional): number of samples to return. Default=1

#         Returns:
#             logp_o (torch.tensor): observation likelihood. size=[T, batch_size]
#             logp_u (torch.tensor): control likelihood. size=[T-1, batch_size]
#             x_sample (torch.tensor): sampled observation sequence. size=[num_samples, T, batch_size, obs_dim]
#             u_sample (torch.tensor): sampled control sequence. size=[num_samples, T-1, batch_size, ctl_dim]
#             alpha_b (torch.tensor): state forward messages. size=[T, batch_size, state_dim]
#             alpha_a (torch.tensor): action forward messages. size=[T-1, batch_size, act_dim]
#         """
#         alpha_b, alpha_a = self._forward(x, u)
        
#         batch_size = x.shape[1]
#         if prior:
#             z0 = self.transition_model.get_initial_state()
#             z0 = z0 * torch.ones(batch_size, 1).to(self.device)
#             z = torch.cat([z0.unsqueeze(0), alpha_b[:-1]], dim=0)
#         else:
#             z = alpha_b
        
#         if not inference:
#             logp_o = self.obs_model.mixture_log_prob(z, self.state_embedding, x)
#             logp_u = self.ctl_model.mixture_log_prob(alpha_a, self.act_embedding, u[:-1])
#             return logp_o, logp_u
#         else:
#             # model average prediction
#             if sample_method == "ace":
#                 x_sample = self.obs_model.ancestral_sample(z, self.state_embedding, num_samples)
#                 u_sample = self.ctl_model.ancestral_sample(alpha_a, self.act_embedding, num_samples)
#             else:
#                 x_sample = self.obs_model.bayesian_average(z, self.state_embedding)
#                 u_sample = self.ctl_model.bayesian_average(alpha_a, self.act_embedding)
            
#             return x_sample, u_sample, alpha_b, alpha_a