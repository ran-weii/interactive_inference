import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple

from src.distributions.utils import poisson_pdf, rectify

def compute_householder_matrix(v):
    """ Compute householder matrix Q = I - 2vv^{T}
    
    Args:
        v (torch.tensor): embedding matrix. size=[batch_size, embed_dim]

    Returns:
        q (torch.tensor): householder matrix. size=[batch_size, embed_dim, embed_dim]
    """
    v_norm = v / torch.linalg.norm(v, dim=-1, keepdim=True)
    i = torch.diag_embed(torch.ones(v.shape, device=v.device))
    q = i - 2 * torch.einsum("...m, ...n -> ...mn", v_norm, v_norm)
    return q

class QMDPLayer(nn.Module):
    """ Custom recurrent neural network implementing the QMDP algorithm """
    def __init__(self, state_dim, act_dim, rank, horizon, detach=False):
        """
        Args:
            state_dim (int): hidden state dimension
            act_dim (int): action dimension
            rank (int): transition matrix embedding dimension. If rank=0 use full rank parameterization
            horizon (int): maximum planning horizon
            detach (bool, optional): whether to stop belief gradient in action computation. Default=False
        """
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.rank = rank
        self.horizon = horizon
        self.detach = detach
        self.eps = 1e-6
        
        self.b0 = nn.Parameter(torch.randn(1, state_dim)) # initial belief
        self.tau = nn.Parameter(torch.randn(1, 1)) # horizon poisson rate

        nn.init.xavier_normal_(self.b0, gain=1.)
        nn.init.uniform_(self.tau, a=-1, b=1)
        
        if rank == 0: # full rank parameterization
            self.w = nn.Parameter(torch.randn(1, act_dim, state_dim, state_dim)) # transition logits
            
            nn.init.xavier_normal_(self.w, gain=1.)
        else: # householder embedding parameterization
            self.u = nn.Parameter(torch.randn(state_dim, rank)) # state embedding
            self.v = nn.Parameter(torch.randn(act_dim, rank)) # action embedding

            nn.init.xavier_normal_(self.u, gain=1.)
            nn.init.xavier_normal_(self.v, gain=1.)
    
    def __repr__(self):
        s = "{}(state_dim={}, act_dim={}, rank={}, horizon={}, detach={})".format(
            self.__class__.__name__, self.state_dim, self.act_dim, self.rank, self.horizon, self.detach
        )
        return s

    def compute_transition(self):
        if self.rank == 0:
            w = self.w
        else:
            q = compute_householder_matrix(self.v)
            u = torch.einsum("kmn, in -> kim", q, self.u)
            w = torch.einsum("kim, jm -> kij", u, self.u).unsqueeze(0)
        return torch.softmax(w, dim=-1)
    
    def compute_value(self, transition: Tensor, reward: Tensor) -> Tensor:
        """ Compute expected value matrix using value iteration

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
        tau = poisson_pdf(rectify(self.tau), self.horizon)
        if tau.shape[0] != b.shape[0]:
            tau = torch.repeat_interleave(tau, b.shape[0], 0)
        
        pi = torch.softmax(torch.einsum("ni, hnki -> hnk", b, value), dim=-1)
        pi = torch.einsum("hnk, nh -> nk", pi, tau)
        return pi
    
    def update_action(self, logp_u: Tensor) -> Tensor:
        """ Compute action posterior 
        
        Args:
            logp_u (torch.tensor): control log probabilities. size=[batch_size, act_dim]

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

    def init_hidden(self, batch_size, value: Tensor) -> Tuple[Tensor, Tensor]:
        b0 = torch.softmax(self.b0, dim=-1).repeat_interleave(batch_size, dim=-2)
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
            logp_u (torch.tensor): sequence of control probabilities. size=[T, batch_size, act_dim]
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
            alpha_b[t+1] = self.update_belief(
                logp_o[t], alpha_a[t], alpha_b[t], transition
            )
            if self.detach:
                alpha_pi[t] = self.plan(alpha_b[t+1].data, value)
            else:
                alpha_pi[t] = self.plan(alpha_b[t+1], value)
        
        alpha_b = torch.stack(alpha_b[1:])
        alpha_pi = torch.stack(alpha_pi)
        return alpha_b, alpha_pi