import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as torch_dist
from src.distributions.nn_models import Model
from src.distributions.nn_models import GRUMLP
from src.distributions.utils import rectify
from src.data.train_utils import count_parameters 

class BehaviorCloning(Model):
    """ Supervised behavior cloning algorithm 
    with truncated backpropagation through time
    """
    def __init__(self, agent, bptt_steps=30, obs_penalty=0, lr=1e-3, decay=0, grad_clip=None):
        super().__init__()
        self.bptt_steps = bptt_steps
        self.obs_penalty = obs_penalty
        self.lr = lr
        self.decay = decay
        self.grad_clip = grad_clip
        self.agent = agent
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=lr, weight_decay=decay
        )
        self.loss_keys = ["loss_u", "loss_o"]
    
    def __repr__(self):
        s_agent = self.agent.__repr__()
        s = "{}(bptt_steps={}, obs_penalty={}, lr={}, decay={}, grad_clip={},\nagent={})".format(
            self.__class__.__name__, self.bptt_steps, self.obs_penalty, self.lr, 
            self.decay, self.grad_clip, s_agent
        )
        return s
    
    def stdout(self, train_stats, test_stats):
        s = "loss_u: {:.4f}/{:.4f}, loss_o: {:.4f}/{:.4f}".format(
            train_stats["loss_u"], test_stats["loss_u"], 
            train_stats["loss_o"], test_stats["loss_o"]
        )
        return s

    def run_epoch(self, loader, train=True):
        if train:
            self.agent.train()
        else:
            self.agent.eval()
        
        epoch_stats = []
        for i, batch in enumerate(loader):
            pad_batch, mask = batch
            o = pad_batch["ego"]
            u = pad_batch["act"]

            o = o.to(self.device)
            u = u.to(self.device)
            mask = mask.to(self.device)
            
            hidden = None
            for t, batch_t in enumerate(zip(
                o.split(self.bptt_steps, dim=0),
                u.split(self.bptt_steps, dim=0),
                mask.split(self.bptt_steps, dim=0)
                )):
                o_t, u_t, mask_t = batch_t
                
                if hidden is None:
                    out, hidden = self.agent(o_t, u_t, hidden)
                else:
                    # concat previous ctl
                    u_t_cat = torch.cat([u_t_prev[-1:], u_t[1:]], dim=0)

                    hidden = [h[-1].detach() for h in hidden]
                    out, hidden = self.agent(o_t, u_t_cat, hidden)
                u_t_prev = u_t.clone()
                
                loss_u, stats_u = self.agent.act_loss(o_t, u_t, mask_t, out)
                loss_o, stats_o = self.agent.obs_loss(o_t, u_t, mask_t, out)

                loss = torch.mean(loss_u + self.obs_penalty * loss_o)
                    
                if train:
                    loss.backward()
                    if self.grad_clip is not None:
                        nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

                    self.optimizer.step()
                self.optimizer.zero_grad()
                
                epoch_stats.append({
                    "train": 1 if train else 0,
                    "loss": loss.cpu().data.item(),
                    **stats_u, **stats_o,
                })
        
        stats = pd.DataFrame(epoch_stats).mean().to_dict()
        return stats

        
class LFBehaviorCloning(Model):
    """ Latent factor behavior cloning """
    def __init__(self, agent, bptt_steps=30, obs_penalty=0, lr=1e-3, decay=0, grad_clip=None):
        super().__init__()
        self.bptt_steps = bptt_steps
        self.obs_penalty = obs_penalty
        self.lr = lr
        self.decay = decay
        self.grad_clip = grad_clip

        self.agent = agent
        
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=lr, weight_decay=decay
        )
        self.loss_keys = ["loss_u", "loss_o"]

        for n, p in self.agent.named_parameters():
            print(n, p.shape)
    
    def __repr__(self):
        s_agent = self.agent.__repr__()
        s = "{}(bptt_steps={}, obs_penalty={}, lr={}, decay={}, grad_clip={},\nagent={})".format(
            self.__class__.__name__, self.bptt_steps, self.obs_penalty, self.lr, 
            self.decay, self.grad_clip, s_agent
        )
        return s
    
    def stdout(self, train_stats, test_stats):
        s = "loss_u: {:.4f}/{:.4f}, loss_o: {:.4f}/{:.4f}".format(
            train_stats["loss_u"], test_stats["loss_u"], 
            train_stats["loss_o"], test_stats["loss_o"]
        )
        return s

    def encode(self, o, u):
        """ Compute latent factors """
        z_params = self.agent.encoder(torch.cat([o, u], dim=-1))
        mu, lv = torch.chunk(z_params, 2, dim=-1)
        z_dist = torch_dist.Normal(mu, rectify(lv))
        z = z_dist.rsample()
        ent = z_dist.entropy().sum(-1, keepdim=True)

        theta = self.agent.decoder(z)
        return theta, ent

    def run_epoch(self, loader, train=True):
        if train:
            self.agent.train()
        else:
            self.agent.eval()
        
        epoch_stats = []
        for i, batch in enumerate(loader):
            pad_batch, mask = batch
            o = pad_batch["ego"]
            u = pad_batch["act"]

            o = o.to(self.device)
            u = u.to(self.device)
            mask = mask.to(self.device)

            hidden = None
            for t, batch_t in enumerate(zip(
                o.split(self.bptt_steps, dim=0),
                u.split(self.bptt_steps, dim=0),
                mask.split(self.bptt_steps, dim=0)
                )):
                o_t, u_t, mask_t = batch_t
                
                # compute latent factors
                theta, ent = self.encode(o, u)

                if hidden is None:
                    out, hidden = self.agent(o_t, u_t, hidden, theta)
                else:
                    # concat previous ctl
                    u_t_cat = torch.cat([u_t_prev[-1:], u_t[1:]], dim=0)

                    hidden = [h[-1].detach() for h in hidden]
                    out, hidden = self.agent(o_t, u_t_cat, hidden, theta)
                u_t_prev = u_t.clone()
                
                loss_u, stats_u = self.agent.act_loss(o_t, u_t, mask_t, out)
                loss_o, stats_o = self.agent.obs_loss(o_t, u_t, mask_t, out)

                loss = len(o) * torch.mean(loss_u + self.obs_penalty * loss_o) - torch.mean(ent)
                    
                if train:
                    loss.backward()
                    if self.grad_clip is not None:
                        nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

                    self.optimizer.step()
                self.optimizer.zero_grad()
                
                epoch_stats.append({
                    "train": 1 if train else 0,
                    "loss": loss.cpu().data.item(),
                    **stats_u, **stats_o,
                })
        
        stats = pd.DataFrame(epoch_stats).mean().to_dict()
        return stats