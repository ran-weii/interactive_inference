import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.distributions.nn_models import Model

class BehaviorCloning(Model):
    """ Supervised behavior cloning algorithm 
    with truncated backpropagation through time
    """
    def __init__(self, agent, bptt_steps=30, bc_penalty=1., obs_penalty=0., prior_penalty=0., lr=1e-3, decay=0, grad_clip=None):
        super().__init__()
        assert agent.__class__.__name__ == "VINAgent"
        self.bptt_steps = bptt_steps
        self.bc_penalty = bc_penalty
        self.obs_penalty = obs_penalty
        self.prior_penalty = prior_penalty
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
        s = "{}(bptt_steps={}, bc_penalty={}, obs_penalty={}, prior_penalty={}, lr={}, decay={}, grad_clip={},\nagent={})".format(
            self.__class__.__name__, self.bptt_steps, self.bc_penalty, self.obs_penalty, self.prior_penalty, 
            self.lr, self.decay, self.grad_clip, s_agent
        )
        return s
    
    def stdout(self, train_stats, test_stats):
        s = "loss_u: {:.4f}/{:.4f}, loss_o: {:.4f}/{:.4f}".format(
            train_stats["loss_u"], test_stats["loss_u"], 
            train_stats["loss_o"], test_stats["loss_o"]
        )
        return s
    
    def compute_prior_loss(self):
        # obs variance
        obs_bn_vars = self.agent.obs_model.bn.moving_variance
        obs_vars = self.agent.obs_model.variance().squeeze(0)
        obs_vars = obs_vars / obs_bn_vars
        obs_loss = torch.sum(obs_vars ** 2)

        # ctl variance
        ctl_bn_vars = self.agent.ctl_model.bn.moving_variance
        ctl_vars = self.agent.ctl_model.variance().squeeze(0)
        ctl_vars = ctl_vars / ctl_bn_vars
        ctl_loss = torch.sum(ctl_vars ** 2)
        
        loss = obs_loss + ctl_loss 
        return loss

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
                
                loss_u, stats_u = self.agent.act_loss(o_t, u_t, mask_t, hidden)
                loss_o, stats_o = self.agent.obs_loss(o_t, u_t, mask_t, hidden)
                
                loss_prior = self.compute_prior_loss()
                loss = torch.mean(self.bc_penalty * loss_u + self.obs_penalty * loss_o) + self.prior_penalty * loss_prior
                    
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

        
class HyperBehaviorCloning(Model):
    def __init__(self, agent, bptt_steps=30, bc_penalty=1., obs_penalty=0., prior_penalty=0., sample_z=False, lr=1e-3, decay=0, grad_clip=None):
        super().__init__()
        assert agent.__class__.__name__ == "HyperVINAgent"
        self.bptt_steps = bptt_steps
        self.bc_penalty = bc_penalty
        self.obs_penalty = obs_penalty
        self.prior_penalty = prior_penalty
        self.sample_z = sample_z # whether to compute obs likelihood with sampled z

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
        s = "{}(bptt_steps={}, bc_penalty={}, obs_penalty={}, prior_penalty={}, sample_z={}, lr={}, decay={}, grad_clip={},\nagent={})".format(
            self.__class__.__name__, self.bptt_steps, self.bc_penalty, self.obs_penalty, self.prior_penalty, 
            self.sample_z, self.lr, self.decay, self.grad_clip, s_agent
        )
        return s
    
    def stdout(self, train_stats, test_stats):
        s = "loss_u: {:.4f}/{:.4f}, loss_o: {:.4f}/{:.4f}".format(
            train_stats["loss_u"], test_stats["loss_u"], 
            train_stats["loss_o"], test_stats["loss_o"]
        )
        return s
    
    def compute_prior_loss(self, z):
        # obs variance
        obs_bn_vars = self.agent.obs_model.bn.moving_variance
        obs_vars = self.agent.obs_model.variance(z).squeeze(0)
        # obs_vars = self.agent.obs_model.variance().squeeze(0)
        obs_vars = obs_vars / obs_bn_vars
        obs_loss = torch.sum(obs_vars ** 2)

        # ctl variance
        ctl_bn_vars = self.agent.ctl_model.bn.moving_variance
        ctl_vars = self.agent.ctl_model.variance().squeeze(0)
        ctl_vars = ctl_vars / ctl_bn_vars
        ctl_loss = torch.sum(ctl_vars ** 2)
        
        loss = obs_loss + ctl_loss 
        return loss
    
    def bptt(self, o, u, mask, train):
        """ Backprop through time """
        z, kl = self.agent.encode(o, u)
        z_sample = self.agent.sample_z((o.shape[1],)).squeeze(-2)
        
        hidden = None
        for t, batch_t in enumerate(zip(
            o.split(self.bptt_steps, dim=0),
            u.split(self.bptt_steps, dim=0),
            mask.split(self.bptt_steps, dim=0)
            )):
            o_t, u_t, mask_t = batch_t

            if hidden is None:
                out, hidden = self.agent(o_t, u_t, z)

            else:
                # concat previous ctl
                u_t_cat = torch.cat([u_t_prev[-1:], u_t[1:]], dim=0)

                hidden = [h[-1].detach() for h in hidden]
                
                out, hidden = self.agent(o_t, u_t_cat, z, hidden)
            u_t_prev = u_t.clone()

            loss_u, stats_u = self.agent.act_loss(o_t, u_t, z, mask_t, hidden)
            if self.sample_z:
                loss_o, stats_o = self.agent.obs_loss(o_t, u_t, z_sample, mask_t, hidden)
            else:
                loss_o, stats_o = self.agent.obs_loss(o_t, u_t, z, mask_t, hidden)
            loss_kl = torch.mean(kl) / o.shape[0]
            loss_prior = self.compute_prior_loss(z)
            loss = torch.mean(self.bc_penalty * loss_u + self.obs_penalty * loss_o) + loss_kl + self.prior_penalty * loss_prior
                
            if train:
                loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

                self.optimizer.step()
            self.optimizer.zero_grad()

            stats = {
                "train": 1 if train else 0,
                "loss": loss.cpu().data.item(),
                **stats_u, **stats_o,
            }
        return stats
    
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
            
            stats = self.bptt(o, u, mask, train)
            epoch_stats.append(stats)
        
        stats = pd.DataFrame(epoch_stats).mean().to_dict()
        return stats

    # def run_epoch(self, loader, train=True):
    #     if train:
    #         self.agent.train()
    #     else:
    #         self.agent.eval()
        
    #     epoch_stats = []
    #     for i, batch in enumerate(loader):
    #         pad_batch, mask = batch
    #         o = pad_batch["ego"]
    #         u = pad_batch["act"]

    #         o = o.to(self.device)
    #         u = u.to(self.device)
    #         mask = mask.to(self.device)

    #         hidden = None
    #         for t, batch_t in enumerate(zip(
    #             o.split(self.bptt_steps, dim=0),
    #             u.split(self.bptt_steps, dim=0),
    #             mask.split(self.bptt_steps, dim=0)
    #             )):
    #             o_t, u_t, mask_t = batch_t
                
    #             # compute latent factors
    #             theta, ent = self.agent.encode(o, u)

    #             if hidden is None:
    #                 out, hidden = self.agent(o_t, u_t)
    #             else:
    #                 # concat previous ctl
    #                 u_t_cat = torch.cat([u_t_prev[-1:], u_t[1:]], dim=0)

    #                 hidden = [
    #                     hidden[0][-1].detach(), hidden[1][-1].detach(),
    #                     theta, ent
    #                 ]
    #                 out, hidden = self.agent(o_t, u_t_cat, hidden, theta)
    #             u_t_prev = u_t.clone()
                
    #             # add uniform noise to observation mask
    #             # eps = 1 + torch.rand(1, o_t.shape[1]).uniform_(-0.2, 0.2)

    #             loss_u, stats_u = self.agent.act_loss(o_t, u_t, mask_t, hidden)
    #             loss_o, stats_o = self.agent.obs_loss(o_t, u_t, mask_t, hidden, perm=True)
                
    #             loss_prior = self.compute_prior_loss(hidden)
    #             loss = torch.mean(self.bc_penalty * loss_u + self.obs_penalty * loss_o) + self.prior_penalty * loss_prior
                    
    #             if train:
    #                 loss.backward()
    #                 if self.grad_clip is not None:
    #                     nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

    #                 self.optimizer.step()
    #             self.optimizer.zero_grad()
                
    #             epoch_stats.append({
    #                 "train": 1 if train else 0,
    #                 "loss": loss.cpu().data.item(),
    #                 **stats_u, **stats_o,
    #             })
        
    #     stats = pd.DataFrame(epoch_stats).mean().to_dict()
    #     return stats
