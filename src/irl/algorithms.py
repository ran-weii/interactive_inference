import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from src.agents.active_inference import ActiveInference

class MLEIRL(nn.Module):
    def __init__(self, state_dim, obs_dim, act_dim, ctl_dim, H, 
        obs_dist="mvn", obs_cov="full", ctl_dist="mvn", ctl_cov="full",
        obs_penalty=0, lr=1e-3, decay=0, grad_clip=40):
        super().__init__()
        self.obs_penalty = obs_penalty
        self.lr = lr
        self.decay = decay
        self.grad_clip = grad_clip
        
        self.agent = ActiveInference(
            state_dim, obs_dim, act_dim, ctl_dim, H,
            obs_dist, obs_cov, ctl_dist, ctl_cov
        )
        self.optimizers = [torch.optim.Adam(
            self.agent.parameters(), lr=lr, weight_decay=decay
        )]
        
    def train_epoch(self, loader):
        self.train()
        
        epoch_stats = []
        num_samples = 0
        for i, batch in enumerate(loader):
            o, u, mask = batch
            out = self.agent(o, u)
            loss, stats = self.loss(out, mask)
            
            nn.utils.clip_grad_value_(self.parameters(), self.grad_clip)
            
            loss.backward()
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_stats.append(stats)
            num_samples += o.shape[1]
        
        df_stats = pd.DataFrame(epoch_stats).mean()
        return df_stats
    
    def test_epoch(self, loader):
        self.eval()
        
        epoch_stats = []
        for i, batch in enumerate(loader):
            o, u, mask = batch
            with torch.no_grad():
                out = self.agent(o, u)
                loss, stats = self.loss(out, mask)
                epoch_stats.append(stats)
            
        df_stats = pd.DataFrame(epoch_stats).mean()
        return df_stats
    
    def loss(self, agent_out, mask):
        [logp_pi, logp_obs] = agent_out
        
        loss_pi = -torch.sum(mask * logp_pi)
        loss_obs = -torch.sum(mask * logp_obs)
        loss = loss_pi + self.obs_penalty * loss_obs
        
        # compute stats
        nan_mask = mask.clone()
        nan_mask[nan_mask == 0] = float("nan")
        logp_pi_np = (nan_mask * logp_pi).data.numpy()
        logp_obs_np = (nan_mask * logp_obs).data.numpy()
        stats_dict = {
            "loss": loss.data.numpy(),
            "loss_pi": loss_pi.data.numpy(),
            "loss_obs": loss_obs.data.numpy(),
            "logp_pi_mean": np.nanmean(logp_pi_np),
            "logp_pi_std": np.nanstd(logp_pi_np),
            "logp_pi_min": np.nanmin(logp_pi_np),
            "logp_pi_max": np.nanmax(logp_pi_np),
            "logp_obs_mean": np.nanmean(logp_obs_np),
            "logp_obs_std": np.nanstd(logp_obs_np),
            "logp_obs_min": np.nanmin(logp_obs_np),
            "logp_obs_max": np.nanmax(logp_obs_np),
        }
        return loss, stats_dict
    
""" TODO: 
implement init parameters similar to test_active_inference
implement elbo loss
"""
class BayesianIRL(MLEIRL):
    def __init__(self, state_dim, obs_dim, act_dim, ctl_dim, H, 
        obs_dist="mvn", obs_cov="full", ctl_dist="mvn", ctl_cov="full",
        obs_penalty=0, lr=1e-3, decay=0, grad_clip=40):
        super().__init__(state_dim, obs_dim, act_dim, ctl_dim, H, 
        obs_dist, obs_cov, ctl_dist, ctl_cov,
        obs_penalty, lr, decay, grad_clip)
        
        self.init_parameters()
        
    def init_parameters(self):
        
        return 
    
    def loss(self, agent_out, mask):
        return 
    