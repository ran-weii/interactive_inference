import torch
import torch.nn as nn

class CPDecomposition(nn.Module):
    """ 3D matrix created using CP tensor decomposition """
    def __init__(self, state_dim, act_dim, rank):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, rank, state_dim)) # source tensor
        self.v = nn.Parameter(torch.randn(1, rank, state_dim)) # sink tensor
        self.w = nn.Parameter(torch.randn(1, rank, act_dim)) # action tensor

        nn.init.xavier_normal_(self.u, gain=1.)
        nn.init.xavier_normal_(self.v, gain=1.)
        nn.init.xavier_normal_(self.w, gain=1.)
    
    def forward(self):
        """ Return matrix size=[1, act_dim, state_dim, state_dim] """
        w = torch.einsum("nri, nrj, nrk -> nkij", self.u, self.v, self.w)
        return torch.softmax(w, dim=-1)


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
        self.u = nn.Parameter(torch.randn(1, rank, state_dim)) # source tensor
        self.v = nn.Parameter(torch.randn(1, rank, state_dim)) # sink tensor
        self.w = nn.Parameter(torch.randn(1, rank, act_dim)) # action tensor

        nn.init.xavier_normal_(self.u, gain=1.)
        nn.init.xavier_normal_(self.v, gain=1.)
        nn.init.xavier_normal_(self.w, gain=1.)
        
    def __repr__(self):
        s = "{}(rank={})".format(self.__class__.__name__, self.rank)
        return s
    
    @property
    def initial_state(self):
        """ Return initial state. size=[1, state_dim] """
        return torch.softmax(self.s0, dim=-1)

    @property
    def transition(self):
        """ Return transition matrix. size=[1, act_dim, state_dim, state_dim] """
        w = torch.einsum("nri, nrj, nrk -> nkij", self.u, self.v, self.w)
        return torch.softmax(w, dim=-1)
    
    def _forward(self, b, a):
        """ Compute forward message: \sum_{s, a} P(s'|s, a)P(s)P(a) 
        
        Args:
            b (torch.tensor): prior belief. size=[batch_size, state_dim]
            a (torch.tensor): prior action. size=[batch_size, act_dim]

        Returns:
            b_next (torch.tensor): posterior belief. size=[batch_size, state_dim]
        """
        transition = self.transition
        b_next = torch.einsum("nkij, ni, nk -> nj", transition, b, a)
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
