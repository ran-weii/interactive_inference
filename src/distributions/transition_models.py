import torch
import torch.nn as nn

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
            self.e = nn.Parameter(torch.randn(1, act_dim, 1, 1, rank))
            self.w1 = nn.Parameter(torch.randn(1, 1, state_dim, 1, rank))
            self.w2 = nn.Parameter(torch.randn(1, 1, 1, state_dim, rank))

            nn.init.xavier_normal_(self.w1, gain=1.)
            nn.init.xavier_normal_(self.w2, gain=1.)
            nn.init.xavier_normal_(self.e, gain=1.)
    
    def get_initial_state(self):
        """
        Returns:
            s0 (torch.tensor): initial state distribution [batch_size, state_dim]
        """
        s0 = torch.softmax(self.s0.view(-1, self.state_dim), dim=-1)
        return s0
    
    def get_transition_matrix(self):
        """
        Returns:
            w (torch.tensor): transition matrix [batch_size, act_dim, state_dim, state_dim]
        """
        if self.rank == 0:
            w = self.w
        else:
            w = torch.sum(self.w1 * self.w2 * self.e, dim=-1)
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
            s0 (torch.tensor): initial state distribution [batch_size, state_dim]
        """
        s0 = self.s0 if params is None else params
        s0 = torch.softmax(s0.view(-1, self.state_dim), dim=-1)
        return s0
    
    def get_transition_matrix(self, a, params=None):
        """
        Args:
            a (torch.tensor): action vector [batch_size, act_dim]

        Returns:
            w (torch.tensor): transition matrix [batch_size, act_dim, state_dim, state_dim]
        """
        w_a = self.w_a if params is None else params
        w_a = w_a.view(-1, self.state_dim, self.state_dim, self.act_dim)
        
        w_a = torch.sum(w_a * a.view(-1, 1, 1, self.act_dim), dim=-1)
        return torch.softmax(w_a + self.w + self.b, dim=-1)
