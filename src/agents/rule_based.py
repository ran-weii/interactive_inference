import torch
import torch.nn as nn
import torch.distributions as torch_dist
from src.agents.baseline import AbstractAgent
from src.distributions.utils import rectify

class IDM(AbstractAgent):
    """ Intelligent driver model """
    def __init__(
        self, v0=80., T=1.7, s0=2., a=0.3, b=2., delta=4., 
        k1=-0.01, k2=-0.2, std=torch.ones(1, 2).to(torch.float32)
        ):
        """
        Args:
            v0 (float): desired speed in km/h. Default=80
            T (float): time headway in s. Default=1.7
            s0 (float): minimum gap in m. Default=2
            a (float): acceleration in m/s/s. Default=0.3
            b (float): deceleration in m/s/s. Default=2.
            delta (float): desired acceleration factor. Default=4.
            k1 (float): lateral distance gain. Default=-0.01
            k2 (float): lateral velocity gain. Default=-0.2
            std (torch.tensor): control std. size=[1, 2]
        """
        super().__init__(None, None, None, None, None)
        self.eps = 1e-6
        self.v0 = nn.Parameter(torch.tensor([v0]).log())
        self.T = nn.Parameter(torch.tensor([T]).log())
        self.s0 = nn.Parameter(torch.tensor([s0]).log())
        self.a = nn.Parameter(torch.tensor([a]).log())
        self.b = nn.Parameter(torch.tensor([b]).log())
        self.delta = nn.Parameter(torch.tensor([delta]).log())
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
        d = o[:, 0]
        ds, dd = o[:, 1], o[:, 2]
        s_rel, ds_rel = o[:, 5], o[:, 7]
        
        # get constants
        v0 = rectify(self.v0)
        T = rectify(self.T)
        s0 = rectify(self.s0)
        a = rectify(self.a)
        b = rectify(self.b)
        delta = rectify(self.delta)
        
        # compute longitudinal acc
        delta_s = ds * T + 0.5 * ds * ds_rel / torch.sqrt(a * b + self.eps)
        s_star = s0 + torch.max(torch.zeros_like(d), delta_s)
        dds = self.a * (torch.ones_like(d) - (ds / (v0 + self.eps))**delta - (s_star / (s_rel + self.eps))**2)
        
        # compute lateral acc using a feedback controller
        ddd = self.k1 * d + self.k2 * dd
        
        mu = torch.stack([dds, ddd]).T
        lv = torch.ones_like(mu) * self.lv
        return mu, lv
    
    def forward(self, o, u):
        """ Compute action distribution for a sequence
        
        Returns:
            mu (torch.tensor): action mean. size=[T, batch_size, 2]
            lv (torch.tensor): action log variance. size=[T, batch_size, 2]
        """
        T = o.shape[0]
        mu = [torch.empty(0)] * T
        lv = [torch.empty(0)] * T
        for t in range(T):
            mu[t], lv[t] = self.compute_action_dist(o[t], u[t])
        mu = torch.stack(mu)
        lv = torch.stack(lv)
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