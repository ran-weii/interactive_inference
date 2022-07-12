import math
import torch
import torch.nn.functional as F
import torch.distributions as torch_dist
from torch.distributions.transforms import TanhTransform
from src.agents.core import AbstractAgent
from src.distributions.nn_models import MLP
from src.distributions.flows import SimpleTransformedModule, TanhTransform
from src.distributions.utils import rectify

class MLPAgent(AbstractAgent):
    """ Gaussian policy network """
    def __init__(
        self, obs_dim, ctl_dim, hidden_dim, num_hidden, activation="silu", 
        use_tanh=True, ctl_limits=None,
        ):
        super().__init__()
        self.obs_dim = obs_dim
        self.ctl_dim = ctl_dim
        self.use_tanh = use_tanh
        self.ctl_limits = ctl_limits

        self.mlp = MLP(
            input_dim=obs_dim,
            output_dim=ctl_dim * 2,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation,
            batch_norm=False
        )
        
        if self.use_tanh:
            self.tanh_transform = TanhTransform(ctl_limits)
    
    def reset(self):
        self._b = None 
        self._prev_ctl = None

    def forward(self, o, u):
        dist_params = self.mlp(o)
        mu, lv = torch.chunk(dist_params, 2, dim=-1)
        return mu, lv
    
    def get_action_dist(self, mu, lv):
        distribution = torch_dist.Normal(mu, rectify(lv))
        if self.use_tanh:
            distribution = SimpleTransformedModule(distribution, [self.tanh_transform])
        return distribution

    def choose_action(self, o, u, sample_method="", num_samples=1):
        mu, lv = self.forward(o, u)
        dist = torch_dist.Normal(mu, rectify(lv))
        ctl = dist.rsample((num_samples,))
        logp = dist.log_prob(ctl).sum(-1, keepdim=True)
        if self.use_tanh:
            logp -= (2. * (math.log(2.) - ctl - F.softplus(-2. * ctl))).sum(-1, keepdim=True)
            ctl *= self.ctl_limits
        return ctl, logp

    def choose_action_batch(self, o, u, sample_method="", num_samples=1):
        mu, lv = self.forward(o, u)
        ctl = self.get_action_dist(mu, lv).rsample((num_samples,))
        return ctl
    
    def ctl_log_prob(self, o, u):
        mu, lv = self.forward(o, u)
        logp_u = self.get_action_dist(mu, lv).log_prob(u).sum(-1)
        return logp_u

    def act_loss(self, o, u, mask, forward_out):
        mu, lv = forward_out
        logp_u = self.get_action_dist(mu, lv).log_prob(u).sum(-1)
        loss = -torch.sum(logp_u * mask, dim=0) / mask.sum(0)
        
        # compute stats
        nan_mask = mask.clone()
        nan_mask[nan_mask == 0] = torch.nan
        logp_u_mean = -torch.nanmean((nan_mask * logp_u)).cpu().data
        stats = {"loss_u": logp_u_mean}
        return loss, stats

    def obs_loss(self, o, u, mask, forward_out):
        loss = torch.zeros(1)
        stats = {"loss_o": loss}
        return loss, stats