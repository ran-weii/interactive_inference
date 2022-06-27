import torch
import torch.nn as nn

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