import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple, Optional

from src.agents.qmdp_layer import compute_householder_matrix
from src.distributions.utils import poisson_pdf, rectify

class HyperQMDPLayer(nn.Module):
    """ Hypernet version of the QMDP layer """
    def __init__(self, state_dim, act_dim, rank, horizon, hyper_dim):
        """
        Args:
            state_dim (int): hidden state dimension
            act_dim (int): action dimension
            rank (int): transition matrix embedding dimension. If rank=0 use full rank parameterization
            horizon (int): maximum planning horizon
            hyper_dim (int): dimension of hyper variable
        """
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.rank = rank
        self.horizon = horizon
        self.hyper_dim = hyper_dim
        self.eps = 1e-6
        
        self._b0 = nn.Parameter(torch.randn(1, state_dim)) # initial belief
        self._tau = nn.Parameter(torch.randn(1, 1)) # horizon poisson rate
        nn.init.xavier_normal_(self._b0, gain=1.)
        nn.init.uniform_(self._tau, a=-1, b=1)

        self._b0_offset = nn.Linear(hyper_dim, state_dim, bias=False)
        self._tau_offset = nn.Linear(hyper_dim, 1, bias=False)
        self._w_offset = nn.Linear(hyper_dim, act_dim * state_dim * state_dim, bias=False)

        self._b0_offset.weight.data *= 0.1
        self._tau_offset.weight.data *= 0.1
        self._w_offset.weight.data *= 0.1
        
        if rank == 0: # full rank parameterization
            self._w = nn.Parameter(torch.randn(1, act_dim, state_dim, state_dim))
            
            nn.init.xavier_normal_(self._w, gain=1.)
        else: # householder embedding parameterization
            self._u = nn.Parameter(torch.randn(state_dim, rank)) # state embedding
            self._v = nn.Parameter(torch.randn(act_dim, rank)) # action embedding

            nn.init.xavier_normal_(self._u, gain=1.)
            nn.init.xavier_normal_(self._v, gain=1.)

    def __repr__(self):
        s = "{}(state_dim={}, act_dim={}, rank={}, horizon={}, hyper_dim={})".format(
            self.__class__.__name__, self.state_dim, self.act_dim, 
            self.rank, self.horizon, self.hyper_dim
        )
        return s
    
    def compute_base_transition(self):
        if self.rank == 0:
            w = self._w
        else:
            q = compute_householder_matrix(self._v)
            u = torch.einsum("kmn, in -> kim", q, self._u)
            w = torch.einsum("kim, jm -> kij", u, self._u).unsqueeze(0)
        return w

    def compute_transition(self, z):
        # compute base transition matrix
        w = self.compute_base_transition()
        w_offset = self._w_offset(z).view(-1, self.act_dim, self.state_dim, self.state_dim)
        return torch.softmax(w_offset + w, dim=-1)
    
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
        tau = self._tau + self._tau_offset(z)
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
        
        a = torch.softmax(torch.einsum("...ni, hnki -> ...hnk", b, value), dim=-1)
        a = torch.einsum("...hnk, nh -> ...nk", a, tau)
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
    
    def init_hidden(self, z: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        b0 = torch.softmax(self._b0 + self._b0_offset(z), dim=-1)
        a0 = self.plan(b0, z, value)
        return b0, a0
    
    def predict_one_step(self, logp_u, b, z):
        transition = self.compute_transition(z)
        a = self.update_action(logp_u)
        s_next = torch.einsum("...kij, ...i, ...k -> ...j", transition, b, a)
        return s_next
    
    def forward(
        self, logp_o: Tensor, logp_u: Union[Tensor, None], value: Tensor, 
        z: Tensor, b: Union[Tensor, None], detach: Optional[bool]=False
        ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            logp_o (torch.tensor): sequence of observation probabilities. size=[T, batch_size, state_dim]
            logp_u (torch.tensor): sequence of control probabilities. size=[T, batch_size, act_dim]
            value (torch.tensor): value matrix. size=[horizon, batch_size, act_dim, state_dim]
            z (torch.tensor): hyper vector, size=[batch_size, hyper_dim]
            b ([torch.tensor, None], optional): prior belief. size=[batch_size, state_dim]
            detach (bool, optional): whether to detach model training. Default=False
        
        Returns:
            alpha_b (torch.tensor): sequence of posterior belief. size=[T, batch_size, state_dim]
            alpha_a (torch.tensor): sequence of action distribution. size=[T, batch_size, act_dim]
        """
        transition = self.compute_transition(z)
        T = len(logp_o)

        alpha_a = self.update_action(logp_u)
        alpha_b = [b] + [torch.empty(0)] * (T)
        alpha_pi = [torch.empty(0)] * (T)
        for t in range(T):
            alpha_b[t+1] = self.update_belief(logp_o[t], alpha_a[t], alpha_b[t], transition)
            if detach:
                alpha_pi[t] = self.plan(alpha_b[t+1].data, z, value)
            else:
                alpha_pi[t] = self.plan(alpha_b[t+1], z, value)
        
        alpha_b = torch.stack(alpha_b[1:])
        alpha_pi = torch.stack(alpha_pi)
        return alpha_b, alpha_pi