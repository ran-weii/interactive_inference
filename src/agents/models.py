import math
import torch
import torch.nn as nn
from src.distributions.models import ConditionalDistribution
from src.distributions.utils import poisson_pdf

class StructuredPerceptionModel(nn.Module):
    def __init__(self, z_dim, dist="mvn", cov="diag"):
        super().__init__()
        x_dim = 11
        self.conditional_dist = ConditionalDistribution(
            x_dim, z_dim, dist, cov, batch_norm=True
        )
        self.k = nn.Parameter(torch.randn(1), requires_grad=True)
        nn.init.uniform_(self.k, a=-0.5, b=0.5)
        
        self.fields = [
            "vx_ego", "vy_ego", 
            "left_bound_dist", "right_bound_dist", 
            "x_rel_ego", "y_rel_ego", "vx_rel_ego", 
            "vy_rel_ego", "psi_rad_rel"
        ]
    
    def entropy(self, theta=None):
        return self.conditional_dist.entropy(theta)
    
    def log_prob(self, o, theta=None):
        o_raw = torch.cat([o[k].unsqueeze(-1) for k in self.fields], dim=-1)
        heading_error = self.get_heading_error(o).unsqueeze(-1)
        loom = self.get_looming(o).unsqueeze(-1)
        obs = torch.cat([o_raw, loom, heading_error], dim=-1)
        return self.conditional_dist.log_prob(obs, theta)
    
    def get_heading_error(self, o):
        ego_heading = o["psi_rad"]
        cell_heading = o["cell_headings"]
        heading_error = cell_heading - ego_heading.unsqueeze(-1).unsqueeze(-1)
        
        heading_error[heading_error > math.pi] -= 2 * math.pi
        heading_error[heading_error < -math.pi] += 2 * math.pi
        
        # let clockwise be positive
        heading_error[:, :, :, 0] *= -1
        avg_heading_error = torch.mean(heading_error, dim=-1).clone()
        
        # compute poisson average
        num_cells = cell_heading.shape[-2]
        k_dist = poisson_pdf(
            self.k.clip(math.log(1e-6), math.log(1e6)).exp(), num_cells
        )
        pois_heading_error = torch.sum(
            avg_heading_error * k_dist.unsqueeze(0).unsqueeze(0), dim=-1
        )
        return pois_heading_error
    
    def get_looming(self, o):
        loom = o["vx_rel_ego"].clone() / (o["x_rel_ego"].clone() + 1e-6)
        return loom

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden, activation):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.activation = activation
        
        if activation == "relu":
            act = nn.ReLU()
        else:
            raise NotImplementedError

        layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_hidden):
            layers.append(act)
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(act)
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.ModuleList(layers)

    def __repr__(self):
        s = "{}(input_dim={}, output_dim={}, hidden_dim={}, num_hidden={}, activation={})".format(
            self.__class__.__name__, self.input_dim, self.output_dim, 
            self.hidden_dim, self.num_hidden, self.activation
        )
        return s

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PopArt(nn.Module):
    """ Linear layer with output normalization """
    def __init__(self, in_features, out_features, momentum=0.1, epsilon=1e-5):
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon

        self.lin = nn.Linear(in_features, out_features)
        self.register_buffer("mu", torch.zeros(out_features, requires_grad=False))
        self.register_buffer("sigma", torch.ones(out_features, requires_grad=False))
    
    @property
    def weight(self):
        return self.lin.weight.data * self.sigma.view(-1, 1)
    
    @property
    def bias(self):
        return self.lin.bias.data * self.sigma + self.mu

    def forward(self, x):
        y_norm = self.lin(x)
        y = y_norm * self.sigma + self.mu
        return y, y_norm

    def normalize(self, y):
        mu_old = self.mu.clone()
        sigma_old = self.sigma.clone()

        op_dims = [i for i in range(len(y.shape) - 1)]
        mean, std = y.mean(op_dims), y.std(op_dims)
        with torch.no_grad():
            self.mu.mul_(1 - self.momentum).add_(mean * self.momentum)
            self.sigma.mul_(1 - self.momentum).add_(std * self.momentum)

        self.lin.weight.data = self.lin.weight.data * sigma_old.view(-1, 1) / (self.sigma + self.epsilon).view(-1, 1)
        self.lin.bias.data = (sigma_old * self.lin.bias + mu_old - self.mu) / (self.sigma + self.epsilon)
        
        y_norm = (y - self.mu) / (self.sigma + self.epsilon)
        return y_norm