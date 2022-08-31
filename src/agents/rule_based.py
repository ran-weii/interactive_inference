import torch
import torch.nn as nn
import torch.distributions as torch_dist
from src.agents.core import AbstractAgent
from src.distributions.utils import rectify

class IDM(AbstractAgent):
    """ Intelligent driver model """
    def __init__(
        self, feature_set, v0=120., tau=1.5, s0=2., a=0.5, b=3., 
        k1=-0.01, k2=-0.2, std=torch.tensor([0.3, 0.003]).to(torch.float32)
        ):
        """
        Args:
            v0 (float): desired speed in km/h. Default=60.
            tau (float): desired time headway in s. Default=0.5
            s0 (float): minimum gap in m. Default=3.
            a (float): desired acceleration in m/s/s. Default=1.5
            b (float): deceleration in m/s/s. Default=4.
            k1 (float): lateral distance gain. Default=-0.01
            k2 (float): lateral velocity gain. Default=-0.2
            std (torch.tensor): control std. size=[1, 2]
        """
        super().__init__(None, None, None, None, None)
        self.eps = 1e-6
        self.feature_set = feature_set
        self.d_idx = feature_set.index("ego_d")
        self.ds_idx = feature_set.index("ego_ds")
        self.dd_idx = feature_set.index("ego_dd")
        self.s_rel_idx = feature_set.index("lv_s_rel")
        self.ds_rel_idx = feature_set.index("lv_ds_rel")
        self.psi_err_idx = feature_set.index("ego_psi_error_r")

        self.v0 = nn.Parameter(torch.tensor([v0]).log())
        self.tau = nn.Parameter(torch.tensor([tau]).log())
        self.s0 = nn.Parameter(torch.tensor([s0]).log())
        self.a = nn.Parameter(torch.tensor([a]).log())
        self.b = nn.Parameter(torch.tensor([b]).log())
        self.k1 = nn.Parameter(torch.tensor([k1]))
        self.k2 = nn.Parameter(torch.tensor([k2]))
        self.lv = nn.Parameter(0.5 * torch.log(std + self.eps))
    
    def reset(self):
        self._b = None
    
    def compute_action_dist(self, o, u):
        """ Compute action distribution 
        
        Returns:
            mu (torch.tensor): action mean. size=[batch_size, 2]
            lv (torch.tensor): action log variance. size=[batch_size, 2]
        """
        d = o[..., self.d_idx]
        ds = o[..., self.ds_idx]
        dd = o[..., self.dd_idx]
        s_rel = o[..., self.s_rel_idx]
        ds_rel = o[..., self.ds_rel_idx]
        
        # get constants
        v0 = rectify(self.v0)
        tau = rectify(self.tau)
        s0 = rectify(self.s0)
        a = rectify(self.a)
        b = rectify(self.b)
        
        # compute longitudinal acc
        delta_s = ds * tau + 0.5 * ds * ds_rel / torch.sqrt(a * b + self.eps)
        s_star = s0 + torch.max(torch.zeros_like(d).to(self.device), delta_s) # desired gap distance
        dds_des = a * (1. - (ds / v0)**4) # obstacle free desired speed
        dds_int = - a * (s_star / (s_rel + self.eps))**2
        dds = dds_des + dds_int

        # emergency braking
        ttc = s_rel / ds_rel
        is_emergency = ttc < 2.
        dds_emergency = -0.05 * s_rel * b
        dds[is_emergency] = dds_emergency[is_emergency]

        # compute lateral acc using a feedback controller
        ddd = self.k1 * d + self.k2 * dd

        dds = dds.clip(-4.5, 4.5)
        ddd = ddd.clip(-1, 1.)
        
        mu = torch.cat([dds.unsqueeze(-1), ddd.unsqueeze(-1)], dim=-1)
        lv =  self.lv * torch.ones_like(mu).to(self.device)
        return mu, lv
    
    def forward(self, o, u):
        """ Compute action distribution for a sequence
        
        Returns:
            mu (torch.tensor): action mean. size=[T, batch_size, 2]
            lv (torch.tensor): action log variance. size=[T, batch_size, 2]
        """
        mu, lv = self.compute_action_dist(o, u)
        return mu, lv

    def choose_action(self, o, u, sample_method="ace", num_samples=1):
        """ Choose action online for a single time step
        
        Args:
            o (torch.tensor): observation sequence. size[batch_size, obs_dim]
            u (torch.tensor): control sequence. size[batch_size, ctl_dim]
            sample_method (str, optional): sampling method. 
                choices=["bma", "ace", "ace"]. Default="ace"
            num_samples (int, optional): number of samples to draw. Default=1
        
        Returns:
            u_sample (torch.tensor): sampled controls. size=[num_samples, batch_size, ctl_dim]
        """
        mu, lv = self.compute_action_dist(o, u)
        mu = mu.clip(-5., 5.)
        a = torch_dist.Normal(mu, rectify(lv)).sample((num_samples,))
        return a

    def act_loss(self, o, u, mask, forward_out):
        mu, lv = forward_out
        logp_u = torch_dist.Normal(mu, rectify(lv)).log_prob(u).sum(-1)
        
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