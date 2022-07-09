import torch
from src.distributions.nn_models import Model
from src.distributions.nn_models import MLP

class DoubleQNetwork(Model):
    """ Double Q network for fully observable use """
    def __init__(self, obs_dim, ctl_dim, hidden_dim, num_hidden, activation="silu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.ctl_dim = ctl_dim

        self.q1 = MLP(
            input_dim=obs_dim + ctl_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation
        )
        self.q2 = MLP(
            input_dim=obs_dim + ctl_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation
        )
    
    def forward(self, o, u):
        """ Compute q1 and q2 values
        
        Args:
            o (torch.tensor): observation. size=[batch_size, obs_dim]
            u (torch.tensor): action. size=[batch_size, ctl_dim]

        Returns:
            q1 (torch.tensor): q1 value. size=[batch_size, 1]
            q2 (torch.tensor): q2 value. size=[batch_size, 1]
        """
        x = torch.cat([o, u], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2
    