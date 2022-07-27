import torch
import torch.nn as nn
import torch.jit as jit

import math
from typing import Union, Tuple
from torch import Tensor

class QMDPLayer(jit.ScriptModule):
    def __init__(self, state_dim, act_dim, rank, horizon, place_holder=False):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.rank = rank
        self.horizon = horizon
        self.eps = 1e-6

        self.b0 = torch.randn(1, state_dim)
        self.u = torch.randn(1, rank, state_dim) # source tensor
        self.v = torch.randn(1, rank, state_dim) # sink tensor
        self.w = torch.randn(1, rank, act_dim) # action tensor 
        self.tau = torch.randn(1, 1)

        if not place_holder:
            self.b0 = nn.Parameter(self.b0)
            self.u = nn.Parameter(self.u) # source tensor
            self.v = nn.Parameter(self.v) # sink tensor
            self.w = nn.Parameter(self.w) # action tensor 
            self.tau = nn.Parameter(self.tau)
            # self.b0 = nn.Parameter(torch.randn(1, state_dim))
            # self.u = nn.Parameter(torch.randn(1, rank, state_dim)) # source tensor
            # self.v = nn.Parameter(torch.randn(1, rank, state_dim)) # sink tensor
            # self.w = nn.Parameter(torch.randn(1, rank, act_dim)) # action tensor 
            # self.tau = nn.Parameter(torch.randn(1, 1))

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
        q = [torch.empty(0)] * (self.horizon)
        q[0] = reward
        for t in range(self.horizon - 1):
            v_next = torch.logsumexp(q[t], dim=-2, keepdim=True)
            q[t+1] = reward + torch.einsum("nkij, nkj -> nki", transition, v_next)
        return torch.stack(q)
    
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


def poisson_pdf(rate: Tensor, K: int) -> Tensor:
    """ jit compatible version of poisson pdf
    
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
