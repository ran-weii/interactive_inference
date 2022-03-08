import torch 
import torch.nn as nn
from src.agents.active_inference import ActiveInference

""" TODO:
implement loss function
"""
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
        return epoch_stats
    
    def test_epoch(self, loader):
        self.eval()
        
        epoch_stats = []
        for i, batch in enumerate(loader):
            o, u, mask = batch
            with torch.no_grad():
                out = self.agent(o, u)
                loss, stats = self.loss(out, mask)
                epoch_stats.append(stats)
        return epoch_stats
    
    def loss(self, agent_out, mask):
        return 
    
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
    