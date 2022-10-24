import torch
import torch.nn as nn

from typing import Union, Tuple, Optional
from torch import Tensor

from src.agents.qmdp_layers import poisson_pdf
from src.distributions.utils import rectify

class HyperQMDPLayer(nn.Module):
    """ Causal hypernet version of qmdp layer """
    def __init__(self, state_dim, act_dim, rank, horizon, hyper_dim):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.rank = rank
        self.horizon = horizon
        self.hyper_dim = hyper_dim
        self.eps = 1e-6
        
        self._b0 = nn.Linear(hyper_dim, state_dim)
        self._tau = nn.Linear(hyper_dim, 1)
        self._gamma = nn.Linear(hyper_dim, act_dim * state_dim) # transition precision

        self._b0.weight.data = 0.1 * torch.randn(self._b0.weight.data.shape)
        self._tau.weight.data = 0.1 * torch.randn(self._tau.weight.data.shape)
        self._gamma.weight.data = 0.1 * torch.randn(self._gamma.weight.data.shape)
        
        if rank != 0:
            self._u = nn.Parameter(torch.randn(1, rank, state_dim)) # source tensor
            self._v = nn.Parameter(torch.randn(1, rank, state_dim)) # sink tensor
            self._w = nn.Parameter(torch.randn(1, rank, act_dim)) # action tensor 
        else:
            self._u = nn.Parameter(torch.randn(1, 1)) # dummy tensor
            self._v = nn.Parameter(torch.randn(1, 1)) # dummy tensor
            self._w = nn.Parameter(torch.randn(1, act_dim, state_dim, state_dim)) # transition tensor
        
        nn.init.xavier_normal_(self._u, gain=1.)
        nn.init.xavier_normal_(self._v, gain=1.)
        nn.init.xavier_normal_(self._w, gain=1.)

    def __repr__(self):
        s = "{}(state_dim={}, act_dim={}, rank={}, horizon={}, hyper_dim={})".format(
            self.__class__.__name__, self.state_dim, self.act_dim, 
            self.rank, self.horizon, self.hyper_dim
        )
        return s

    @property
    def transition(self):
        """ Return transition matrix. size=[1, act_dim, state_dim, state_dim] """
        z = torch.zeros(1, self.hyper_dim).to(self._b0.bias.device)
        return self.compute_transition(z)
    
    def parameter_entropy(self, z):
        eps = 1e-6
        b0_ent = torch.log(torch.abs(self._b0.weight) + eps).sum() / self.hyper_dim
        w_ent = torch.log(torch.abs(self._w.weight) + eps).sum() / self.hyper_dim
        tau_ent = torch.log(torch.abs(self._tau.weight) + eps).sum() / self.hyper_dim
        ent = b0_ent + w_ent + tau_ent
        
        if self.rank > 0:
            u_ent = torch.log(torch.abs(self._u.weight) + eps).sum() / self.hyper_dim
            v_ent = torch.log(torch.abs(self._v.weight) + eps).sum() / self.hyper_dim
            ent += u_ent + v_ent
        return ent

    def compute_transition(self, z):
        # compute base transition matrix
        if self.rank != 0:
            w = torch.einsum("nri, nrj, nrk -> nkij", self._u, self._v, self._w)
        else:
            w = self._w
        
        # compute transition temperature
        gamma = rectify(self._gamma(z)).view(-1, self.act_dim, self.state_dim, 1)
        return torch.softmax(gamma * w, dim=-1)
    
    def compute_value(self, transition: Tensor, reward: Tensor) -> Tensor:
        """ Compute expected value using value iteration

        Args:
            transition (torch.tensor): transition matrix. size=[batch_size, act_dim, state_dim, state_dim]
            reward (torch.tensor): reward matrix. size=[batch_size, act_dim, state_dim]
        
        Returns:
            q (torch.tensor): state q value. size=[horizon, batch_size, act_dim, state_dim]
        """        
        q = [torch.empty(0)] * (self.horizon)
        q[0] = reward
        for t in range(self.horizon - 1):
            v_next = torch.logsumexp(q[t], dim=-2, keepdim=True)
            q[t+1] = reward + torch.einsum("nkij, nkj -> nki", transition, v_next)
        return torch.stack(q)

    def compute_horizon_dist(self, z):
        tau = self._tau(z)
        h_dist = poisson_pdf(rectify(tau), self.horizon)
        return h_dist

    def plan(self, b: Tensor, z: Tensor, value: Tensor) -> Tensor:
        """ Compute the belief action distribution 
        
        Args:
            b (torch.tensor): current belief. size=[batch_size, state_dim]
            z (torch.tensor): hyper vector. size=[batch_size, hyper_dim]
            value (torch.tensor): state q value. size=[horizon, batch_size, act_dim, state_dim]

        Returns:
            a (torch.tensor): action distribution. size=[batch_size, act_dim]
        """
        tau = self.compute_horizon_dist(z)
        
        a = torch.softmax(torch.einsum("ni, hnki -> hnk", b, value), dim=-1)
        a = torch.einsum("hnk, nh -> nk", a, tau)
        return a
    
    def update_action(self, logp_u: Tensor) -> Tensor:
        """ Compute action posterior 
        
        Args:
            logp_u (torch.tensor): log probability of previous control. size=[batch_size, act_dim]

        Returns:
            a_post (torch.tensor): action posterior. size=[batch_size, act_dim]
        """ 
        a_post = torch.softmax(logp_u, dim=-1)
        return a_post
 
    def update_belief(self, logp_o: Tensor, a: Tensor, b: Tensor, transition: Tensor) -> Tensor:
        """ Compute state posterior
        
        Args:
            logp_o (torch.tensor): log probability of current observation. size=[batch_size, state_dim]
            a (torch.tensor): action posterior. size=[batch_size, act_dim]
            b (torch.tensor): prior belief. size=[batch_size, state_dim]
            transition (torch.tensor): transition matrix. size=[batch_size, act_dim, state_dim, state_dim]

        Returns:
            b_post (torch.tensor): state posterior. size=[batch_size, state_dim]
        """
        s_next = torch.einsum("nkij, ni, nk -> nj", transition, b, a)
        logp_s = torch.log(s_next + self.eps)
        b_post = torch.softmax(logp_s + logp_o, dim=-1)
        return b_post
    
    def update_cell(
        self, logp_o: Tensor, a: Tensor, b: Tensor, transition: Tensor
        ) -> Tensor:
        """ Compute action and state posterior. Then compute the next action distribution """
        b_post = self.update_belief(logp_o, a, b, transition)
        return b_post
    
    def init_hidden(self, z: Tensor) -> Tensor:
        b0 = torch.softmax(self._b0(z), dim=-1)
        return b0
    
    def predict_one_step(self, logp_u, b, z):
        transition = self.compute_transition(z)
        a = self.update_action(logp_u)
        s_next = torch.einsum("...kij, ...i, ...k -> ...j", transition, b, a)
        return s_next

    def forward(
        self, logp_o: Tensor, logp_u: Union[Tensor, None], reward: Tensor, 
        z: Tensor, b: Union[Tensor, None], detach: Optional[bool]=False
        ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            logp_o (torch.tensor): sequence of observation probabilities. size=[T, batch_size, state_dim]
            logp_u (torch.tensor): sequence of control probabilities. Should be offset -1 if b is None.
                size=[T, batch_size, act_dim]
            reward (torch.tensor): reward matrix. size=[batch_size, act_dim, state_dim]
            z (torch.tensor): hyper vector, size=[batch_size, hyper_dim]
            b ([torch.tensor, None], optional): prior belief. size=[batch_size, state_dim]
            detach (bool, optional): whether to detach model training. Default=False
        
        Returns:
            alpha_b (torch.tensor): sequence of posterior belief. size=[T, batch_size, state_dim]
            alpha_a (torch.tensor): sequence of action distribution. size=[T, batch_size, act_dim]
        """
        batch_size = logp_o.shape[1]
        transition = self.compute_transition(z)
        value = self.compute_value(transition, reward)
        T = len(logp_o)
        
        if b is None:
            b = self.init_hidden(z)
            logp_u = torch.cat([torch.zeros(1, batch_size, self.act_dim).to(self._b0.weight.device), logp_u], dim=0)

        alpha_a = self.update_action(logp_u)
        alpha_b = [b] + [torch.empty(0)] * (T)
        alpha_pi = [torch.empty(0)] * (T)
        for t in range(T):
            alpha_b[t+1] = self.update_cell(logp_o[t], alpha_a[t], alpha_b[t], transition)
            if detach:
                alpha_pi[t] = self.plan(alpha_b[t+1].data, z, value)
            else:
                alpha_pi[t] = self.plan(alpha_b[t+1], z, value)
        
        alpha_b = torch.stack(alpha_b[1:])
        alpha_pi = torch.stack(alpha_pi)
        return alpha_b, alpha_pi


# class HyperQMDPLayer(jit.ScriptModule):
#     """ Non causal hypernet version of qmdp layer """
#     def __init__(self, state_dim, act_dim, rank, horizon, hyper_dim):
#         super().__init__()
#         self.state_dim = state_dim
#         self.act_dim = act_dim
#         self.rank = rank
#         self.horizon = horizon
#         self.hyper_dim = hyper_dim
#         self.eps = 1e-6
        
#         self._b0 = nn.Linear(hyper_dim, state_dim)
#         self._u = nn.Linear(hyper_dim, rank * state_dim) # source tensor
#         self._v = nn.Linear(hyper_dim, rank * state_dim) # sink tensor
#         self._w = nn.Linear(hyper_dim, rank * act_dim) # action tensor
#         self._tau = nn.Linear(hyper_dim, 1)
    
#     def __repr__(self):
#         s = "{}(state_dim={}, act_dim={}, rank={}, horizon={}, hyper_dim={})".format(
#             self.__class__.__name__, self.state_dim, self.act_dim, 
#             self.rank, self.horizon, self.hyper_dim
#         )
#         return s

#     @property
#     def transition(self):
#         """ Return transition matrix. size=[1, act_dim, state_dim, state_dim] """
#         z = torch.zeros(1, self.hyper_dim).to(self._b0.bias.device)
#         return self.compute_transition(z)
    
#     @jit.script_method
#     def compute_transition(self, z):
#         u = self._u(z).view(-1, self.rank, self.state_dim)
#         v = self._v(z).view(-1, self.rank, self.state_dim)
#         w = self._w(z).view(-1, self.rank, self.act_dim)
#         core = torch.einsum("nri, nrj, nrk -> nkij", u, v, w)
#         return torch.softmax(core, dim=-1)

#     @jit.script_method
#     def compute_value(self, transition: Tensor, reward: Tensor) -> Tensor:
#         """ Compute expected value using value iteration

#         Args:
#             transition (torch.tensor): transition matrix. size=[batch_size, act_dim, state_dim, state_dim]
#             reward (torch.tensor): reward matrix. size=[batch_size, act_dim, state_dim]
        
#         Returns:
#             q (torch.tensor): state q value. size=[horizon, batch_size, act_dim, state_dim]
#         """        
#         q = [torch.empty(0)] * (self.horizon)
#         q[0] = reward
#         for t in range(self.horizon - 1):
#             v_next = torch.logsumexp(q[t], dim=-2, keepdim=True)
#             q[t+1] = reward + torch.einsum("nkij, nkj -> nki", transition, v_next)
#         return torch.stack(q)
    
#     @jit.script_method
#     def plan(self, b: Tensor, z: Tensor, value: Tensor) -> Tensor:
#         """ Compute the belief action distribution 
        
#         Args:
#             b (torch.tensor): current belief. size=[batch_size, state_dim]
#             z (torch.tensor): hyper vector. size=[batch_size, hyper_dim]
#             value (torch.tensor): state q value. size=[horizon, batch_size, act_dim, state_dim]

#         Returns:
#             a (torch.tensor): action distribution. size=[batch_size, act_dim]
#         """
#         tau = torch.exp(self._tau(z).clip(math.log(1e-6), math.log(1e3)))
#         tau = poisson_pdf(tau, self.horizon)
        
#         a = torch.softmax(torch.einsum("ni, hnki -> hnk", b, value), dim=-1)
#         a = torch.einsum("hnk, nh -> nk", a, tau)
#         return a

#     @jit.script_method
#     def update_belief(self, logp_o: Tensor, transition: Tensor, b: Tensor, a: Tensor) -> Tensor:
#         """ Compute state posterior
        
#         Args:
#             logp_o (torch.tensor): log probability of current observation. size=[batch_size, state_dim]
#             transition (torch.tensor): transition matrix. size=[batch_size, act_dim, state_dim, state_dim]
#             b (torch.tensor): prior belief. size=[batch_size, state_dim]
#             a (torch.tensor): action posterior. size=[batch_size, act_dim]

#         Returns:
#             b_post (torch.tensor): state posterior. size=[batch_size, state_dim]
#         """
#         s_next = torch.einsum("nkij, ni, nk -> nj", transition, b, a)
#         logp_s = torch.log(s_next + self.eps)
#         b_post = torch.softmax(logp_s + logp_o, dim=-1)
#         return b_post
    
#     @jit.script_method
#     def update_action(self, logp_u: Tensor, a: Tensor) -> Tensor:
#         """ Compute action posterior 
        
#         Args:
#             logp_u (torch.tensor): log probability of previous control. size=[batch_size, act_dim]
#             a (torch.tensor): action prior. size=[batch_size, act_dim]

#         Returns:
#             a_post (torch.tensor): action posterior. size=[batch_size, act_dim]
#         """ 
#         logp_a = torch.log(a + self.eps)
#         a_post = torch.softmax(logp_a + logp_u, dim=-1)
#         return a_post
    
#     @jit.script_method
#     def update_cell(
#         self, logp_o: Tensor, logp_u: Tensor, z: Tensor, b: Tensor, a: Tensor, 
#         transition: Tensor, value: Tensor
#         ) -> Tuple[Tensor, Tensor]:
#         """ Compute action and state posterior. Then compute the next action distribution """
#         a_post = self.update_action(logp_u, a)
#         b_post = self.update_belief(logp_o, transition, b, a_post)
#         a_next = self.plan(b_post, z, value)
#         return (b_post, a_next)
    
#     @jit.script_method
#     def init_hidden(self, logp_o: Tensor, z: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
#         b0 = torch.softmax(self._b0(z), dim=-1)
#         logp_s = torch.log(b0 + self.eps)
#         b = torch.softmax(logp_s + logp_o, dim=-1)
#         a = self.plan(b, z, value)
#         return b, a
    
#     def forward(
#         self, logp_o: Tensor, logp_u: Union[Tensor, None], reward: Tensor, 
#         z: Tensor, b: Union[Tensor, None], a: Union[Tensor, None]
#         ) -> Tuple[Tensor, Tensor]:
#         """
#         Args:
#             logp_o (torch.tensor): sequence of observation probabilities. size=[T, batch_size, state_dim]
#             logp_u (torch.tensor): sequence of control probabilities. Should be offset -1 if b is None.
#                 size=[T, batch_size, act_dim]
#             reward (torch.tensor): reward matrix. size=[batch_size, act_dim, state_dim]
#             z (torch.tensor): hyper vector, size=[batch_size, hyper_dim]
#             b ([torch.tensor, None], optional): prior belief. size=[batch_size, state_dim]
#             a ([torch.tensor, None], optional): prior action. size=[batch_size, act_dim]
        
#         Returns:
#             alpha_b (torch.tensor): sequence of posterior belief. size=[T, batch_size, state_dim]
#             alpha_a (torch.tensor): sequence of action distribution. size=[T, batch_size, act_dim]
#         """
#         transition = self.compute_transition(z)
#         value = self.compute_value(transition, reward)
#         t_offset = 0 if b is not None else -1 # offset for offline prediction
#         T = len(logp_o)
        
#         alpha_b = [b] + [torch.empty(0)] * (T)
#         alpha_a = [a] + [torch.empty(0)] * (T)
#         for t in range(T):
#             if alpha_b[t] is None:
#                 alpha_b[t+1], alpha_a[t+1] = self.init_hidden(logp_o[t], z, value)
#             else:
#                 alpha_b[t+1], alpha_a[t+1] = self.update_cell(
#                     logp_o[t], logp_u[t+t_offset], z,
#                     alpha_b[t], alpha_a[t], transition, value
#                 )
#         return torch.stack(alpha_b[1:]), torch.stack(alpha_a[1:])
