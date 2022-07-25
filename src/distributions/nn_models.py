import torch
import torch.nn as nn
from src.distributions.flows import BatchNormTransform

class Model(nn.Module):
    """ Constructor for nn models with device property """
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    @property
    def device(self):
        return next(self.parameters()).device


class MLP(Model):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden, activation, batch_norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.activation = activation
        self.batch_norm = batch_norm
        
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "silu":
            act = nn.SiLU()
        else:
            raise NotImplementedError

        layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_hidden):
            layers.append(act)
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(act)
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.ModuleList(layers)

        if batch_norm:
            self.bn = BatchNormTransform(input_dim, affine=False)

    def __repr__(self):
        s = "{}(input_dim={}, output_dim={}, hidden_dim={}, num_hidden={}, activation={}, batch_norm={})".format(
            self.__class__.__name__, self.input_dim, self.output_dim, 
            self.hidden_dim, self.num_hidden, self.activation, self.batch_norm
        )
        return s

    def forward(self, x):
        if self.batch_norm:
            x = self.bn(x)
            
        for layer in self.layers:
            x = layer(x)
        return x


class GRUMLP(Model):
    """ GRU + MLP """
    def __init__(self, input_dim, output_dim, hidden_dim, gru_layers, mlp_layers, activation):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        
        self.h0 = nn.Parameter(torch.randn(gru_layers, hidden_dim))
        self.gru = nn.GRU(input_dim, hidden_dim, gru_layers)
        self.mlp = MLP(gru_layers * hidden_dim, output_dim, hidden_dim, mlp_layers, activation)
        nn.init.xavier_normal_(self.h0, gain=1.)

    def forward(self, x):
        h0 = self.init_hidden(x.shape[1])
        out, hn = self.gru(x, h0)
        hn = hn.transpose(0, 1).flatten(1, -1)
        out = self.mlp(hn)
        return out
    
    def init_hidden(self, batch_size):
        h0 = torch.repeat_interleave(self.h0.unsqueeze(-2), batch_size, -2)
        return h0


class PopArt(Model):
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
        y = y_norm * self.sigma.data + self.mu.data
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