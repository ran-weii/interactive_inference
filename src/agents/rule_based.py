import torch
import torch.nn as nn
import torch.distributions as torch_dist
from src.agents.core import AbstractAgent
from src.distributions.utils import rectify

class IDM(AbstractAgent):
    """ Intelligent driver model for longitudinal control """
    def __init__(
        self, feature_set, v0=120., tau=1.5, s0=2., a=0.5, b=3., std=0.01
        ):
        """
        Args:
            v0 (float): desired speed in km/h. Default=60.
            tau (float): desired time headway in s. Default=0.5
            s0 (float): minimum gap in m. Default=3.
            a (float): desired acceleration in m/s/s. Default=1.5
            b (float): deceleration in m/s/s. Default=4.
            std (float): control std. Default=0.01
        """
        super().__init__()
        assert feature_set == ["ego_ds", "lv_s_rel", "lv_ds_rel"]
        self.eps = 1e-6

        self.v0 = nn.Parameter(torch.tensor([v0]).log())
        self.tau = nn.Parameter(torch.tensor([tau]).log())
        self.s0 = nn.Parameter(torch.tensor([s0]).log())
        self.a = nn.Parameter(torch.tensor([a]).log())
        self.b = nn.Parameter(torch.tensor([b]).log())
        self.lv = nn.Parameter(0.5 * torch.log(torch.tensor([std + self.eps])))
    
    def reset(self):
        self._b = None
        self._state = {
            "b": None, # dummy belief distribution
            "pi": None, # previous policy/action prior
        }
    
    def compute_action_dist(self, o):
        """ Compute action distribution 
        
        Returns:
            mu (torch.tensor): action mean. size=[batch_size, 1]
            lv (torch.tensor): action log variance. size=[batch_size, 1]
        """
        ds = o[..., 0]
        s_rel = o[..., 1]
        ds_rel = o[..., 2]
        
        # get constants
        v0 = rectify(self.v0)
        tau = rectify(self.tau)
        s0 = rectify(self.s0)
        a = rectify(self.a)
        b = rectify(self.b)
        
        # compute longitudinal acc
        s_des = s0 + ds * tau - 0.5 * ds * ds_rel / torch.sqrt(a * b + self.eps)
        mu = a * (1. - (ds / v0)**4 - (s_des / s_rel)**2).unsqueeze(-1)
        lv =  self.lv * torch.ones_like(mu, device=self.device)
        return mu, lv
    
    def forward(self, o):
        """ Compute action distribution for a sequence
        
        Returns:
            mu (torch.tensor): action mean. size=[T, batch_size, 2]
            lv (torch.tensor): action log variance. size=[T, batch_size, 2]
        """
        mu, lv = self.compute_action_dist(o)
        return mu, lv

    def choose_action(self, o, sample_method="ace", num_samples=1):
        """ Choose action online for a single time step
        
        Args:
            o (torch.tensor): observation sequence. size[batch_size, obs_dim]
            sample_method (str, optional): sampling method. 
                choices=["bma", "ace", "ace"]. Default="ace"
            num_samples (int, optional): number of samples to draw. Default=1
        
        Returns:
            u_sample (torch.tensor): sampled controls. size=[num_samples, batch_size, ctl_dim]
        """
        mu, lv = self.compute_action_dist(o)
        a = torch_dist.Normal(mu, rectify(lv)).sample((num_samples,))
        logp = torch_dist.Normal(mu, rectify(lv)).log_prob(a).sum(-1)

        self._state["b"] = None
        self._state["pi"] = mu.clone()
        return a, logp

    def choose_action_batch(self, o, u, sample_method="", num_samples=1, return_hidden=False, **kwargs):
        mu, lv = self.compute_action_dist(o)
        a = torch_dist.Normal(mu, rectify(lv)).sample((num_samples,))
        logp = torch_dist.Normal(mu, rectify(lv)).log_prob(a).sum(-1)

        if return_hidden:
            return a, logp, None
        else:
            return a, logp

    def act_loss(self, o, u, mask, forward_out):
        mu, lv = forward_out
        logp_u = torch_dist.Normal(mu, rectify(lv)).log_prob(u).sum(-1)
        
        loss = -torch.sum(logp_u * mask, dim=0) / (mask.sum(0) + 1e-6)
        
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