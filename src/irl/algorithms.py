import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from src.agents.active_inference import ActiveInference
from src.agents.baseline import ExpertNetwork

class ImitationLearning(nn.Module):
    def __init__(self, act_dim, obs_dim, ctl_dim, 
            obs_penalty=0, lr=1e-3, decay=0, grad_clip=40):
        super().__init__()
        self.obs_penalty = obs_penalty
        self.lr = lr
        self.decay = decay
        self.grad_clip = grad_clip
        self.agent = ExpertNetwork(
            act_dim, obs_dim, ctl_dim, nb=True, prod=False
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
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            
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
    
    
class MLEIRL(nn.Module):
    def __init__(self, state_dim, act_dim, obs_dim, ctl_dim, H, 
        obs_dist="mvn", obs_cov="full", ctl_dist="mvn", ctl_cov="full",
        obs_penalty=0, lr=1e-3, decay=0, grad_clip=40):
        super().__init__()
        self.obs_penalty = obs_penalty
        self.lr = lr
        self.decay = decay
        self.grad_clip = grad_clip
        
        self.agent = ActiveInference(
            state_dim, act_dim, obs_dim, ctl_dim, H,
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
            
            # if torch.isnan(loss):
            #     raise ValueError("nan in loss")
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            
            # # nan gradient handle
            # num_nan = 0
            # for n, p in self.named_parameters():
            #     if p.grad is not None:
            #         num_nan += torch.isnan(p.grad.data).sum()
            # if num_nan > 0:
            #     print(f"{num_nan} nan in grad")
                
            for optimizer in self.optimizers:
                # if num_nan == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_stats.append(stats)
            num_samples += o.shape[1]
        
        df_stats = pd.DataFrame(epoch_stats).mean()
        return df_stats
    
    """ TODO: add offline metrics to test epoch? """
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
    

class BayesianIRL(MLEIRL):
    def __init__(self, state_dim, act_dim, obs_dim, ctl_dim, H, 
        obs_dist="mvn", obs_cov="full", ctl_dist="mvn", ctl_cov="full",
        obs_penalty=0, lr=1e-3, decay=0, grad_clip=40):
        super().__init__(state_dim, act_dim, obs_dim, ctl_dim, H, 
        obs_dist, obs_cov, ctl_dist, ctl_cov,
        obs_penalty, lr, decay, grad_clip)
        A_size = sum(self.agent.obs_model.parameter_size)
        B_size = self.agent.hmm.parameter_size[0]
        C_size = self.agent.state_dim
        D_size = self.agent.hmm.parameter_size[1]
        F_size = sum(self.agent.ctl_model.parameter_size)
        tau_size = 1
        self.parameter_keys = ["A", "B", "C", "D", "F", "tau"]
        self.parameter_size = [A_size, B_size, C_size, D_size, F_size, tau_size]
        
        self.init_parameters()
        self.optimizers = [torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=decay
        )]
        
    def init_parameters(self):
        [A_size, B_size, C_size, D_size, F_size, tau_size] = self.parameter_size
        self.A_mu = nn.Parameter(torch.randn(A_size), requires_grad=True)
        self.A_lv = nn.Parameter(torch.randn(A_size), requires_grad=True)
        self.B_mu = nn.Parameter(torch.randn(B_size), requires_grad=True)
        self.B_lv = nn.Parameter(torch.randn(B_size), requires_grad=True)
        self.C_mu = nn.Parameter(torch.randn(C_size), requires_grad=True)
        self.C_lv = nn.Parameter(torch.randn(C_size), requires_grad=True)
        self.D_mu = nn.Parameter(torch.randn(D_size), requires_grad=True)
        self.D_lv = nn.Parameter(torch.randn(D_size), requires_grad=True)
        self.F_mu = nn.Parameter(torch.randn(F_size), requires_grad=True)
        self.F_lv = nn.Parameter(torch.randn(F_size), requires_grad=True)
        self.tau_mu = nn.Parameter(torch.randn(tau_size), requires_grad=True)
        self.tau_lv = nn.Parameter(torch.randn(tau_size), requires_grad=True)
        
        nn.init.normal_(self.A_lv, mean=0, std=1)
        nn.init.normal_(self.B_lv, mean=0, std=1)
        nn.init.normal_(self.C_lv, mean=0, std=1)
        nn.init.normal_(self.D_lv, mean=0, std=1)
        nn.init.normal_(self.F_lv, mean=0, std=1)
        nn.init.normal_(self.tau_lv, mean=0, std=1)
    
    def load_map_parameters(self):
        """ Load MAP parameters to agent """
        A_mu, A_lv, A_tl, A_sk = self.agent.obs_model.transform_parameters(self.A_mu.unsqueeze(0))
        self.agent.obs_model.mu.data = A_mu.data.view(self.agent.obs_model.mu.shape)
        self.agent.obs_model.lv.data = A_lv.data.view(self.agent.obs_model.lv.shape)
        self.agent.obs_model.tl.data = A_tl.data.view(self.agent.obs_model.tl.shape)
        self.agent.obs_model.sk.data = A_sk.data.view(self.agent.obs_model.sk.shape)
        
        self.agent.hmm.B.data = self.B_mu.data.view(self.agent.hmm.B.shape)
        self.agent.C.data = self.C_mu.data.view(self.agent.C.shape)
        self.agent.hmm.D.data = self.D_mu.data.view(self.agent.hmm.D.shape)
        
        F_mu, F_lv, F_tl, F_sk = self.agent.ctl_model.transform_parameters(self.F_mu.unsqueeze(0))
        self.agent.ctl_model.mu.data = F_mu.data.view(self.agent.ctl_model.mu.shape)
        self.agent.ctl_model.lv.data = F_lv.data.view(self.agent.ctl_model.lv.shape)
        self.agent.ctl_model.tl.data = F_tl.data.view(self.agent.ctl_model.tl.shape)
        self.agent.ctl_model.sk.data = F_sk.data.view(self.agent.ctl_model.sk.shape)
        
        self.agent.tau.data = self.tau_mu.data.view(self.agent.tau.shape)
    
    def train_epoch(self, loader):
        self.train()
        
        epoch_stats = []
        num_samples = 0
        for i, batch in enumerate(loader):
            o, u, mask = batch
            theta = self.encode(o.shape[1])
            out = self.agent(o, u, theta=theta)
            loss, stats = self.loss(out, mask)
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                
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
                theta = self.encode(o.shape[1])
                out = self.agent(o, u, theta=theta)
                loss, stats = self.loss(out, mask)
                epoch_stats.append(stats)
            
        df_stats = pd.DataFrame(epoch_stats).mean()
        return df_stats
    
    def encode(self, batch_size):
        """ Draw samples from posterior distribution """
        mu = torch.cat(
            [self.A_mu, self.B_mu, self.C_mu, self.D_mu, self.F_mu, self.tau_mu], dim=-1
        )
        lv = torch.cat(
            [self.A_lv, self.B_lv, self.C_lv, self.D_lv, self.F_lv, self.tau_lv], dim=-1
        )
        cov = torch.diag_embed(lv.exp())
        
        theta = torch.distributions.MultivariateNormal(mu, cov).rsample((batch_size,))
        theta = torch.split(theta, self.parameter_size, dim=-1)
        theta_dict = {k: theta[i] for (i, k) in enumerate(self.parameter_keys)}
        return theta_dict
    
    def entropy(self):
        mu = torch.cat(
            [self.A_mu, self.B_mu, self.C_mu, self.D_mu, self.F_mu, self.tau_mu], dim=-1
        )
        lv = torch.cat(
            [self.A_lv, self.B_lv, self.C_lv, self.D_lv, self.F_lv, self.tau_lv], dim=-1
        )
        cov = torch.diag_embed(lv.exp())
        ent = torch.distributions.MultivariateNormal(mu, cov).entropy()
        return ent
    
    def loss(self, agent_out, mask):
        # entropy loss
        loss_ent = -self.entropy()
        
        # likelihood loss
        [logp_pi, logp_obs] = agent_out
        
        loss_pi = -torch.sum(mask * logp_pi)
        loss_obs = -torch.sum(mask * logp_obs)
        loss = loss_pi + self.obs_penalty * loss_obs + loss_ent
        
        # compute stats
        nan_mask = mask.clone()
        nan_mask[nan_mask == 0] = float("nan")
        logp_pi_np = (nan_mask * logp_pi).data.numpy()
        logp_obs_np = (nan_mask * logp_obs).data.numpy()
        stats_dict = {
            "loss": loss.data.numpy(),
            "loss_pi": loss_pi.data.numpy(),
            "loss_obs": loss_obs.data.numpy(),
            "loss_ent": loss_ent.data.numpy(),
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
    