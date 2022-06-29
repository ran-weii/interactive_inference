import torch
import torch.nn as nn

class CPDecomposition(nn.Module):
    """ 3D matrix created using CP tensor decomposition """
    def __init__(self, state_dim, act_dim, rank):
        super().__init__()
        self.e = nn.Parameter(torch.randn(1, act_dim, 1, 1, rank)) # action embedding
        self.w1 = nn.Parameter(torch.randn(1, 1, state_dim, 1, rank)) # source state embedding
        self.w2 = nn.Parameter(torch.randn(1, 1, 1, state_dim, rank)) # sink state embedding

        nn.init.xavier_normal_(self.w1, gain=1.)
        nn.init.xavier_normal_(self.w2, gain=1.)
        nn.init.xavier_normal_(self.e, gain=1.)
    
    def forward(self):
        """ Return matrix size=[1, act_dim, state_dim, state_dim] """
        w = torch.sum(self.w1 * self.w2 * self.e, dim=-1)
        return w


class TuckerDecomposition(nn.Module):
    """ 3D matrix created using Tucker decomposition """
    def __init__(self, state_dim, act_dim, state_embed_dim, act_embed_dim):
        super().__init__()
        self.state_embedding = nn.Parameter(torch.randn(1, state_dim, state_embed_dim))
        self.act_embedding = nn.Parameter(torch.randn(1, act_dim, act_embed_dim))
        self.core_tensor = nn.Parameter(torch.randn(1, act_embed_dim, state_embed_dim, state_embed_dim))
        
        nn.init.xavier_normal_(self.state_embedding, gain=1.)
        nn.init.xavier_normal_(self.act_embedding, gain=1.)
        nn.init.xavier_normal_(self.core_tensor, gain=1.)
    
    def forward(self):
        """ Return matrix size=[1, act_dim, state_dim, state_dim] """
        source_embedding = self.state_embedding.unsqueeze(-1).unsqueeze(-4)
        sink_embedding = self.state_embedding.transpose(-1, -2).unsqueeze(-3).unsqueeze(-3)
        act_embedding = self.act_embedding.unsqueeze(-1).unsqueeze(-1)
        core_tensor = self.core_tensor.unsqueeze(-3)
        
        out = torch.sum(source_embedding * core_tensor, dim=-2)
        out = torch.sum(out.unsqueeze(-1) * sink_embedding, dim=-2)
        out = torch.sum(out.unsqueeze(-4) * act_embedding, dim=-3)
        return out


class DiscreteMC(nn.Module):
    """ Discrete Markov chain for discrete actions """
    def __init__(self, state_dim, act_dim, rank=0):
        """
        Args:
            state_dim (int): state dimension
            act_dim (int): discrete action dimension
            rank (int, optional): matrix rank. Dense matrix if rank=0, 
                else use CP tensor decomposition. Default=0
        """
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.rank = rank
        
        self.s0 = nn.Parameter(torch.randn(1, state_dim))
        nn.init.xavier_normal_(self.s0, gain=1.)
        
        if rank == 0:
            self.w = nn.Parameter(torch.randn(1, act_dim, state_dim, state_dim))
            nn.init.xavier_normal_(self.w, gain=1.)
        else:
            self.w = CPDecomposition(state_dim, act_dim, rank)

    def __repr__(self):
        s = "{}(rank={})".format(self.__class__.__name__, self.rank)
        return s

    def get_initial_state(self):
        """
        Returns:
            s0 (torch.tensor): initial state distribution. size=[batch_size, state_dim]
        """
        s0 = torch.softmax(self.s0.view(-1, self.state_dim), dim=-1)
        return s0
    
    def get_transition_matrix(self):
        """
        Returns:
            w (torch.tensor): transition matrix. size=[batch_size, act_dim, state_dim, state_dim]
        """
        if self.rank == 0:
            w = self.w
        else:
            w = self.w()
        return torch.softmax(w, dim=-1)


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
