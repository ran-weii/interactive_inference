import torch
import torch.nn as nn
import torch.jit as jit

import math
from typing import Union, Tuple
from torch import Tensor

def poisson_pdf(rate: Tensor, K: int) -> Tensor:
    """ jit compatible version of poisson pdf
    
    Args:
        rate (torch.tensor): poission arrival rate [batch_size, 1]
        K (int): number of bins
        beta (torch.tensor): precision parameter. Default=1.

    Returns:
        pdf (torch.tensor): truncated poisson pdf [batch_size, K]
    """
    Ks = 1 + torch.arange(K).to(rate.device)
    poisson_logp = Ks.xlogy(rate) - rate - (Ks + 1).lgamma()
    pdf = torch.softmax(poisson_logp, dim=-1)
    return pdf


class QMDPLayer(nn.Module):
    """ Causal qmdp layer using global action prior """
    def __init__(self, state_dim, act_dim, rank, horizon, detach=True):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.rank = rank
        self.horizon = horizon
        self.eps = 1e-6
        self.detach = detach

        self.b0 = nn.Parameter(torch.randn(1, state_dim)) # initial belief
        self.a0 = nn.Parameter(torch.randn(1, act_dim)) # total action prior
        self.tau = nn.Parameter(torch.randn(1, 1))

        nn.init.xavier_normal_(self.b0, gain=1.)
        nn.init.xavier_normal_(self.a0, gain=1.)
        nn.init.uniform_(self.tau, a=-1, b=1)
        
        # if rank != 0:
        #     self.u = nn.Parameter(torch.randn(1, rank, state_dim)) # source tensor
        #     self.v = nn.Parameter(torch.randn(1, rank, state_dim)) # sink tensor
        #     self.w = nn.Parameter(torch.randn(1, rank, act_dim)) # action tensor 
        # else:
        #     self.u = nn.Parameter(torch.randn(1, 1)) # dummy tensor
        #     self.v = nn.Parameter(torch.randn(1, 1)) # dummy tensor
        #     self.w = nn.Parameter(torch.randn(1, act_dim, state_dim, state_dim)) # action tensor 

        # householder embedding parameterization
        self.u = nn.Parameter(torch.randn(state_dim, rank)) # state embedding
        self.v = nn.Parameter(torch.randn(act_dim, rank)) # action embedding
        
        nn.init.xavier_normal_(self.u, gain=1.)
        nn.init.xavier_normal_(self.v, gain=1.)
        # nn.init.xavier_normal_(self.w, gain=1.)
    
    def __repr__(self):
        s = "{}(state_dim={}, act_dim={}, rank={}, horizon={}, detach={})".format(
            self.__class__.__name__, self.state_dim, self.act_dim, self.rank, self.horizon, self.detach
        )
        return s

    def compute_transition(self):
        def compute_householder_matrix():
            v_norm = self.v / torch.linalg.norm(self.v, dim=-1, keepdim=True)
            i = torch.eye(self.rank).unsqueeze(0)
            i = torch.repeat_interleave(i, self.act_dim, dim=0)
            q = i - 2 * torch.einsum("...m, ...n -> ...mn", v_norm, v_norm)
            return q
        
        q = compute_householder_matrix()
        u = torch.einsum("kmn, in -> kim", q, self.u)
        w = torch.einsum("kim, jm -> kij", u, self.u).unsqueeze(0)
        return torch.softmax(w, dim=-1)

    # def compute_transition(self):
    #     if self.rank != 0:
    #         w = torch.einsum("nri, nrj, nrk -> nkij", self.u, self.v, self.w)
    #     else:
    #         w = self.w
    #     return torch.softmax(w, dim=-1)
    
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
    
    def plan(self, b: Tensor, value: Tensor) -> Tensor:
        """ Compute the belief policy distribution 
        
        Args:
            b (torch.tensor): current belief. size=[batch_size, state_dim]
            value (torch.tensor): state q value. size=[horizon, batch_size, act_dim, state_dim]

        Returns:
            pi (torch.tensor): policy/action prior. size=[batch_size, act_dim]
        """
        tau = torch.exp(self.tau.clip(math.log(1e-6), math.log(1e3)))
        tau = poisson_pdf(tau, self.horizon)
        if tau.shape[0] != b.shape[0]:
            tau = torch.repeat_interleave(tau, b.shape[0], 0)
        
        pi = torch.softmax(torch.einsum("ni, hnki -> hnk", b, value), dim=-1)
        pi = torch.einsum("hnk, nh -> nk", pi, tau)
        return pi
    
    def update_action(self, logp_u: Tensor) -> Tensor:
        """ Compute action posterior 
        
        Args:
            logp_u (torch.tensor): log probability of previous control. size=[batch_size, act_dim]
            pi (torch.tensor): action prior/policy. size=[batch_size, act_dim]

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

    def init_hidden(self, batch_size, value: Tensor) -> Tuple[Tensor, Tensor]:
        b0 = torch.softmax(self.b0, dim=-1)
        b0 = torch.repeat_interleave(b0, batch_size, dim=-2)
        a0 = self.plan(b0, value)
        return b0, a0
    
    def predict_one_step(self, logp_u, b):
        transition = self.compute_transition()
        a = self.update_action(logp_u)
        s_next = torch.einsum("...kij, ...i, ...k -> ...j", transition, b, a)
        return s_next
    
    def forward(
        self, logp_o: Tensor, logp_u: Union[Tensor, None], value: Tensor, b: Union[Tensor, None], 
        ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            logp_o (torch.tensor): sequence of observation probabilities. size=[T, batch_size, state_dim]
            logp_u (torch.tensor): sequence of control probabilities. Should be offset -1 if b is None.
                size=[T, batch_size, act_dim]
            value (torch.tensor): value matrix. size=[horizon, batch_size, act_dim, state_dim]
            b ([torch.tensor, None], optional): prior belief. size=[batch_size, state_dim]
        
        Returns:
            alpha_b (torch.tensor): sequence of posterior belief. size=[T, batch_size, state_dim]
            alpha_a (torch.tensor): sequence of policies. size=[T, batch_size, act_dim]
        """
        transition = self.compute_transition()
        T = len(logp_o)
        
        alpha_a = self.update_action(logp_u) # action posterior
        alpha_b = [b] + [torch.empty(0)] * (T) # state posterior
        alpha_pi = [torch.empty(0)] * (T) # policy/action prior
        for t in range(T):
            alpha_b[t+1] = self.update_cell(
                logp_o[t], alpha_a[t], alpha_b[t], transition
            )
            if self.detach:
                alpha_pi[t] = self.plan(alpha_b[t+1].data, value)
            else:
                alpha_pi[t] = self.plan(alpha_b[t+1], value)
        
        alpha_b = torch.stack(alpha_b[1:])
        alpha_pi = torch.stack(alpha_pi)
        return alpha_b, alpha_pi


class QMDPLayer2(jit.ScriptModule):
    """ Non causal QMDP layer using agent policy as prior """
    def __init__(self, state_dim, act_dim, rank, horizon, detach=True):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.rank = rank
        self.horizon = horizon
        self.eps = 1e-6
        self.detach = detach

        self.b0 = nn.Parameter(torch.randn(1, state_dim)) # initial belief
        self.a0 = nn.Parameter(torch.randn(1, act_dim)) # total action prior
        self.u = nn.Parameter(torch.randn(1, rank, state_dim)) # source tensor
        self.v = nn.Parameter(torch.randn(1, rank, state_dim)) # sink tensor
        self.w = nn.Parameter(torch.randn(1, rank, act_dim)) # action tensor 
        self.tau = nn.Parameter(torch.randn(1, 1))

        nn.init.xavier_normal_(self.b0, gain=1.)
        nn.init.xavier_normal_(self.a0, gain=1.)
        nn.init.xavier_normal_(self.u, gain=1.)
        nn.init.xavier_normal_(self.v, gain=1.)
        nn.init.xavier_normal_(self.w, gain=1.)
        nn.init.uniform_(self.tau, a=-1, b=1)
    
    def __repr__(self):
        s = "{}(state_dim={}, act_dim={}, rank={}, horizon={}, detach={})".format(
            self.__class__.__name__, self.state_dim, self.act_dim, self.rank, self.horizon, self.detach
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
        """ Compute the belief policy distribution 
        
        Args:
            b (torch.tensor): current belief. size=[batch_size, state_dim]
            value (torch.tensor): state q value. size=[horizon, batch_size, act_dim, state_dim]

        Returns:
            pi (torch.tensor): policy/action prior. size=[batch_size, act_dim]
        """
        tau = torch.exp(self.tau.clip(math.log(1e-6), math.log(1e3)))
        tau = poisson_pdf(tau, self.horizon)
        if tau.shape[0] != b.shape[0]:
            tau = torch.repeat_interleave(tau, b.shape[0], 0)
        
        pi = torch.softmax(torch.einsum("ni, hnki -> hnk", b, value), dim=-1)
        pi = torch.einsum("hnk, nh -> nk", pi, tau)
        return pi

    @jit.script_method
    def update_action(self, logp_u: Tensor, pi: Tensor) -> Tensor:
        """ Compute action posterior 
        
        Args:
            logp_u (torch.tensor): log probability of previous control. size=[batch_size, act_dim]
            pi (torch.tensor): action prior/policy. size=[batch_size, act_dim]

        Returns:
            a_post (torch.tensor): action posterior. size=[batch_size, act_dim]
        """ 
        logp_pi = torch.log(pi + self.eps)
        a_post = torch.softmax(logp_pi + logp_u, dim=-1)
        return a_post
    
    @jit.script_method
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

    @jit.script_method
    def update_cell(
        self, logp_o: Tensor, logp_u: Tensor, pi: Tensor, b: Tensor, transition: Tensor
        ) -> Tuple[Tensor, Tensor]:
        """ Compute action and state posterior. Then compute the next action distribution """
        a_post = self.update_action(logp_u, pi)
        b_post = self.update_belief(logp_o, a_post, b, transition)
        return (a_post, b_post)
    
    @jit.script_method
    def init_hidden(self) -> Tuple[Tensor, Tensor]:
        b0 = torch.softmax(self.b0, dim=-1)
        pi0 = torch.softmax(self.a0, dim=-1)
        return b0, pi0

    def forward(
        self, logp_o: Tensor, logp_u: Union[Tensor, None], reward: Tensor,
        b: Union[Tensor, None], pi: Union[Tensor, None]
        ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            logp_o (torch.tensor): sequence of observation probabilities. size=[T, batch_size, state_dim]
            logp_u (torch.tensor): sequence of control probabilities. Should be offset -1 if b is None.
                size=[T, batch_size, act_dim]
            reward (torch.tensor): reward matrix. size=[batch_size, act_dim, state_dim]
            b ([torch.tensor, None], optional): prior belief. size=[batch_size, state_dim]
            pi ([torch.tensor, None], optional): prior action. size=[batch_size, act_dim]
        
        Returns:
            alpha_b (torch.tensor): sequence of posterior belief. size=[T, batch_size, state_dim]
            alpha_a (torch.tensor): sequence of action distribution. size=[T, batch_size, act_dim]
        """
        batch_size = logp_o.shape[1]
        transition = self.transition
        value = self.compute_value(transition, reward)
        T = len(logp_o)
        
        if b is None:
            b, pi = self.init_hidden()
            logp_u = torch.cat([torch.zeros(1, batch_size, self.act_dim).to(self.b0.device), logp_u], dim=0)
        
        alpha_a = [torch.empty(0)] * (T) # action posterior
        alpha_b = [b] + [torch.empty(0)] * (T) # state posterior
        alpha_pi = [pi] + [torch.empty(0)] * (T) # policy/action prior
        for t in range(T):
            if self.detach:
                alpha_a[t], alpha_b[t+1] = self.update_cell(
                    logp_o[t], logp_u[t], alpha_pi[t].data, alpha_b[t], transition
                )
                alpha_pi[t+1] = self.plan(alpha_b[t+1].data, value)
            else:
                alpha_a[t], alpha_b[t+1] = self.update_cell(
                    logp_o[t], logp_u[t], alpha_pi[t], alpha_b[t], transition
                )
                alpha_pi[t+1] = self.plan(alpha_b[t+1], value)
        
        alpha_a = torch.stack(alpha_a)
        alpha_b = torch.stack(alpha_b[1:])
        alpha_pi = torch.stack(alpha_pi[1:])
        return alpha_a, alpha_b, alpha_pi