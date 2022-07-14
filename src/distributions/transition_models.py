import torch
import torch.nn as nn

class CPDecomposition(nn.Module):
    """ 3D matrix created using CP tensor decomposition """
    def __init__(self, state_dim, act_dim, rank):
        super().__init__()
        self.e = nn.Parameter(torch.randn(1, act_dim, 1, 1, rank)) # action embedding
        self.w1 = nn.Parameter(torch.randn(1, 1, state_dim, 1, rank)) # source state embedding
        self.w2 = nn.Parameter(torch.randn(1, 1, 1, state_dim, rank)) # sink state embedding

        # nn.init.xavier_normal_(self.w1, gain=1.)
        # nn.init.xavier_normal_(self.w2, gain=1.)
        # nn.init.xavier_normal_(self.e, gain=1.)
    
    def forward(self):
        """ Return matrix size=[1, act_dim, state_dim, state_dim] """
        w = torch.sum(self.w1 * self.w2 * self.e, dim=-1)
        return w


class TuckerDecomposition(nn.Module):
    """ 3D matrix created using Tucker decomposition """
    def __init__(self, state_dim, act_dim, state_embed_dim, act_embed_dim, core_rank=0):
        super().__init__()
        self.core_rank = core_rank
        
        self.state_embedding = nn.Parameter(torch.randn(1, state_dim, state_embed_dim))
        self.act_embedding = nn.Parameter(torch.randn(1, act_dim, act_embed_dim))
        
        nn.init.xavier_normal_(self.state_embedding, gain=1.)
        nn.init.xavier_normal_(self.act_embedding, gain=1.)

        if core_rank == 0:
            self.w = nn.Parameter(torch.randn(1, act_embed_dim, state_embed_dim, state_embed_dim))
            nn.init.xavier_normal_(self.w, gain=1.)
        else:
            self.w = CPDecomposition(
                state_embed_dim, act_embed_dim, core_rank
            )
    
    @property
    def core(self):
        if self.core_rank == 0:
            w = self.w
        else:
            w = self.w()
        return w
    
    def forward(self):
        """ Return matrix size=[1, act_dim, state_dim, state_dim] """
        source_embedding = self.state_embedding.unsqueeze(-1).unsqueeze(-4)
        sink_embedding = self.state_embedding.transpose(-1, -2).unsqueeze(-3).unsqueeze(-3)
        act_embedding = self.act_embedding.unsqueeze(-1).unsqueeze(-1)
        core_tensor = self.core.unsqueeze(-3)
        
        out = torch.sum(source_embedding * core_tensor, dim=-2)
        out = torch.sum(out.unsqueeze(-1) * sink_embedding, dim=-2)
        out = torch.sum(out.unsqueeze(-4) * act_embedding, dim=-3)
        return out


class DiscreteMC(nn.Module):
    """ Discrete Markov chain for discrete actions using cp decomposition """
    def __init__(self, state_dim, act_dim, rank):
        """
        Args:
            state_dim (int): state dimension
            act_dim (int): discrete action dimension
            rank (int): transition matrix rank
        """
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.rank = rank
        
        self.s0 = nn.Parameter(torch.randn(1, state_dim)) # initial state
        self._u = nn.Parameter(torch.randn(1, act_dim, rank, state_dim)) # source tensor
        self._v = nn.Parameter(torch.randn(1, act_dim, rank, state_dim)) # sink tensor
        # nn.init.xavier_normal_(self.s0, gain=1.)
        # nn.init.xavier_normal_(self._u, gain=1.)
        # nn.init.xavier_normal_(self._v, gain=1.)
        
    def __repr__(self):
        s = "{}(rank={})".format(self.__class__.__name__, self.rank)
        return s
    
    @property
    def u(self):
        return torch.softmax(self._u, dim=-2)
    
    @property
    def v(self):
        return torch.softmax(self._v, dim=-1)
    
    @property
    def initial_state(self):
        """ Return initial state. size=[1, state_dim] """
        return torch.softmax(self.s0, dim=-1)

    @property
    def transition(self):
        """ Return transition matrix. size=[1, act_dim, state_dim, state_dim] """
        u = self.u.unsqueeze(-1)
        v = self.v.unsqueeze(-2)
        transition = torch.sum(u * v, dim=-3)
        return transition
    
    def _forward(self, b, a):
        """ Compute forward message: \sum_{s, a} P(s'|s, a)P(s)P(a) 
        
        Args:
            b (torch.tensor): prior belief. size=[batch_size, state_dim]
            a (torch.tensor): prior action. size=[batch_size, act_dim]

        Returns:
            b_next (torch.tensor): posterior belief. size=[batch_size, state_dim]
        """
        transition = self.transition
        b_ = b.unsqueeze(-2).unsqueeze(-1)
        a_ = a.unsqueeze(-1).unsqueeze(-1)

        b_next = torch.sum(transition * b_ * a_, dim=[-3, -2])
        return b_next
    
    def _tensor_forward(self, b, a):
        """ Compute forward message using tensor method: \sum_{s, a} P(s'|s, a)P(s)P(a) 
        
        Args:
            b (torch.tensor): prior belief. size=[batch_size, state_dim]
            a (torch.tensor): prior action. size=[batch_size, act_dim]

        Returns:
            b_next (torch.tensor): posterior belief. size=[batch_size, state_dim]
        """
        u = self.u
        v = self.v

        b_ = b.unsqueeze(-2).unsqueeze(-2)
        a_ = a.unsqueeze(-1)
        
        u_dot_b = torch.sum(u * b_, dim=-1, keepdim=True)
        b_next = torch.sum(v * u_dot_b, dim=-2)
        b_next = torch.sum(b_next * a_, dim=-2)
        return b_next

    def _backward(self, m):
        """ Compute backward message: \sum_{s'} P(s'|s, a)m(s') 
        
        Args:
            m (torch.tensor): previous message. size=[batch_size, state_dim]
        
        Return:
            m_next (torch.tensor): next message. size=[batch_size, state_dim]
        """
        transition = self.transition
        m_next = torch.sum(transition * m.unsqueeze(-2), dim=-1)
        return m_next
    
    def _tensor_backward(self, m):
        """ Compute backward message using tensor method: \sum_{s'} P(s'|s, a)m(s') 
        
        Args:
            m (torch.tensor): previous message. size=[batch_size, state_dim] 
                or [batch_size, act_dim, state_dim]
        
        Return:
            m_next (torch.tensor): next message. size=[batch_size, state_dim]
        """
        u = self.u
        v = self.v

        m_ = m.unsqueeze(-2)
        v_dot_m = torch.sum(v * m_, dim=-1, keepdim=True)
        m_next = torch.sum(u * v_dot_m, dim=-2)
        return m_next


# class EmbeddedDiscreteMC(nn.Module):
#     """ Discrete Markov chain for discrete actions with embeddings """
#     def __init__(self, state_dim, act_dim, state_embed_dim, act_embed_dim, rank=0):
#         """
#         Args:
#             state_dim (int): state dimension
#             act_dim (int): discrete action dimension
#             state_embed_dim (int): state embedding dimension
#             act_embed_dim (int): act embedding dimension
#             rank (int, optional): core tensor rank. Dense matrix if rank=0, 
#                 else use CP tensor decomposition. Default=0
#         """
#         super().__init__()
#         self.state_dim = state_dim
#         self.act_dim = act_dim
#         self.state_embed_dim = state_embed_dim
#         self.act_embed_dim = act_embed_dim
#         self.rank = rank
        
#         self.s0 = nn.Parameter(torch.randn(1, state_dim))
#         nn.init.xavier_normal_(self.s0, gain=1.)
        
#         self.w = TuckerDecomposition(
#             state_dim, act_dim, state_embed_dim, act_embed_dim, core_rank=rank
#         )

#     def __repr__(self):
#         s = "{}(state_embedding={}, act_embedding={}, core_rank={})".format(
#             self.__class__.__name__, self.state_embed_dim, 
#             self.act_embed_dim, self.rank
#         )
#         return s

#     def get_initial_state(self):
#         """
#         Returns:
#             s0 (torch.tensor): initial state distribution. size=[batch_size, state_dim]
#         """
#         s0 = torch.softmax(self.s0.view(-1, self.state_dim), dim=-1)
#         return s0
    
#     def get_transition_matrix(self):
#         """
#         Returns:
#             w (torch.tensor): transition matrix. size=[batch_size, act_dim, state_dim, state_dim]
#         """
#         w = self.w()
#         return torch.softmax(w, dim=-1)


class LogisticMC(nn.Module):
    """ Discrete Markov chain for continuous actions """
    def __init__(self, state_dim, act_dim):
        """
        Args:
            state_dim (int): state dimension
            act_dim (int): continuous action dimension
        """
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        
        self.s0 = nn.Parameter(torch.randn(1, state_dim))
        self.w = nn.Parameter(torch.randn(1, state_dim, state_dim))
        self.w_a = nn.Parameter(torch.randn(1, state_dim, state_dim, act_dim))
        self.b = nn.Parameter(torch.zeros(1, state_dim, 1))

        nn.init.xavier_normal_(self.s0, gain=1.)
        nn.init.xavier_normal_(self.w, gain=1.)

    def get_initial_state(self, params=None):
        """
        Returns:
            s0 (torch.tensor): initial state distribution. size=[batch_size, state_dim]
        """
        s0 = self.s0 if params is None else params
        s0 = torch.softmax(s0.view(-1, self.state_dim), dim=-1)
        return s0
    
    def get_transition_matrix(self, a, params=None):
        """
        Args:
            a (torch.tensor): action vector. size=[batch_size, act_dim]

        Returns:
            w (torch.tensor): transition matrix. size=[batch_size, act_dim, state_dim, state_dim]
        """
        w_a = self.w_a if params is None else params
        w_a = w_a.view(-1, self.state_dim, self.state_dim, self.act_dim)
        
        w_a = torch.sum(w_a * a.view(-1, 1, 1, self.act_dim), dim=-1)
        return torch.softmax(w_a + self.w + self.b, dim=-1)
