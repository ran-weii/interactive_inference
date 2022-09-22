import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as torch_dist
from src.distributions.nn_models import Model

class HyperBehaviorCloning(Model):
    def __init__(self, agent, train_mode, bptt_steps=30, bc_penalty=1., obs_penalty=0., reg_penalty=0., sample_z=False, lr=1e-3, decay=0, grad_clip=None):
        super().__init__()
        assert agent.__class__.__name__ == "HyperVINAgent"
        assert train_mode in ["prior", "post", "marginal"]
        self.train_mode = train_mode
        self.bptt_steps = bptt_steps
        self.bc_penalty = bc_penalty
        self.obs_penalty = obs_penalty
        self.reg_penalty = reg_penalty
        self.sample_z = sample_z # whether to compute obs likelihood with sampled z

        self.lr = lr
        self.decay = decay
        self.grad_clip = grad_clip

        self.agent = agent

        self._set_params_grad()
        
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=lr, weight_decay=decay
        )
        self.loss_keys = ["total_loss", "loss_u", "loss_o"]
    
    def _set_params_grad(self):
        if self.train_mode == "prior":
            for n, p in self.named_parameters():
                if "encoder" in n:
                    p.requires_grad = False

        elif self.train_mode == "post":
            for n, p in self.named_parameters():
                if "encoder" not in n:
                    p.requires_grad = False

    def __repr__(self):
        s_agent = self.agent.__repr__()
        s = "{}(train_mode={}, bptt_steps={}, bc_penalty={}, obs_penalty={}, reg_penalty={}, sample_z={}, lr={}, decay={}, grad_clip={},\nagent={})".format(
            self.__class__.__name__, self.train_mode, self.bptt_steps, self.bc_penalty, self.obs_penalty, self.reg_penalty, 
            self.sample_z, self.lr, self.decay, self.grad_clip, s_agent
        )
        return s
    
    def stdout(self, train_stats, test_stats):
        s = "total_loss: {:.4f}/{:.4f}, loss_u: {:.4f}/{:.4f}, loss_o: {:.4f}/{:.4f}".format(
            train_stats["total_loss"], test_stats["total_loss"], 
            train_stats["loss_u"], test_stats["loss_u"], 
            train_stats["loss_o"], test_stats["loss_o"]
        )
        return s
    
    def compute_reg_loss(self, z):
        """ Regularization loss """
        # obs variance
        obs_bn_vars = self.agent.obs_model.bn.moving_variance
        obs_vars = self.agent.obs_model.variance(z).squeeze(0)
        obs_vars = obs_vars / obs_bn_vars
        obs_loss = torch.sum(obs_vars ** 2, dim=[-1, -2]).mean()

        # ctl variance
        ctl_bn_vars = self.agent.ctl_model.bn.moving_variance
        ctl_vars = self.agent.ctl_model.variance().squeeze(0)
        ctl_vars = ctl_vars / ctl_bn_vars
        ctl_loss = torch.sum(ctl_vars ** 2)
        
        loss = obs_loss + ctl_loss 
        return loss

    def compute_prior_loss(self, o, u, mask, num_total_eps):
        """ KL loss for fitting maximum entropy hypernet prior """
        prior_dist = self.agent.get_prior_dist()
        z = prior_dist.rsample()
        z = torch.repeat_interleave(z, o.shape[1], dim=0)
        
        _, hidden = self.agent.forward(o, u, z, detach=True)
        act_loss, act_stats = self.agent.act_loss(o, u, z, mask, hidden)
        obs_loss, obs_stats = self.agent.obs_loss(o, u, z, mask, hidden)
        
        ent = self.agent.parameter_entropy(z).mean()
        reg_loss = self.compute_reg_loss(z)
        
        avg_eps_len = torch.mean(mask.sum(0))
        loss = avg_eps_len * num_total_eps * (
            self.bc_penalty * act_loss.mean() + \
            self.obs_penalty * obs_loss.mean() + \
            self.reg_penalty * reg_loss
        ) - ent

        stats = {
            "total_loss": loss.data.item(),
            **act_stats, **obs_stats,
            "ent": ent.data.item()
        }
        return loss, stats
    
    def compute_posterior_loss(self, o, u, mask):
        """ Elbo loss with fixed prior and hypernet """
        prior_dist = self.agent.get_prior_dist()
        z_dist = self.agent.get_posterior_dist(o, u, mask)
        z = z_dist.rsample()
        
        _, hidden = self.agent.forward(o, u, z, detach=False)
        act_loss, act_stats = self.agent.act_loss(o, u, z, mask, hidden)
        obs_loss, obs_stats = self.agent.obs_loss(o, u, z, mask, hidden)

        kl = torch_dist.kl.kl_divergence(z_dist, prior_dist).sum(-1)
        reg_loss = self.compute_reg_loss(z)

        loss = torch.mean(act_loss * mask.sum(0) + self.reg_penalty * reg_loss + kl) 
        
        stats = {
            "total_loss": loss.data.item(),
            **act_stats, **obs_stats,
            "kl": kl.data.mean().item()
        }
        return loss, stats
    
    def compute_loss(self, o, u, mask, num_total_eps):
        if self.train_mode == "prior":
            loss, stats = self.compute_prior_loss(o, u, mask, num_total_eps)
        elif self.train_mode == "post":
            loss, stats = self.compute_posterior_loss(o, u, mask)
        return loss, stats

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
            
            # stats = self.bptt(o, u, mask, train)
            # epoch_stats.append(stats)
            loss, stats = self.compute_loss(o, u, mask, len(loader.dataset))
            
            if train:
                loss.backward()
                for n, p in self.named_parameters():
                    if p.grad is None:
                        print(n, p.requires_grad, None)
                    else:
                        print(n, p.requires_grad, p.grad.data.norm())
                exit()
                
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

                self.optimizer.step()
            self.optimizer.zero_grad()
            
            stats["train"] = 1 if train else 0
            epoch_stats.append(stats)

        stats = pd.DataFrame(epoch_stats).mean().to_dict()
        return stats

    # def compute_prior_loss(self, z):
    #     # obs variance
    #     obs_bn_vars = self.agent.obs_model.bn.moving_variance
    #     obs_vars = self.agent.obs_model.variance(z).squeeze(0)
    #     # obs_vars = self.agent.obs_model.variance().squeeze(0)
    #     obs_vars = obs_vars / obs_bn_vars
    #     obs_loss = torch.sum(obs_vars ** 2)

    #     # ctl variance
    #     ctl_bn_vars = self.agent.ctl_model.bn.moving_variance
    #     ctl_vars = self.agent.ctl_model.variance().squeeze(0)
    #     ctl_vars = ctl_vars / ctl_bn_vars
    #     ctl_loss = torch.sum(ctl_vars ** 2)
        
    #     loss = obs_loss + ctl_loss 
    #     return loss
    
    # def bptt(self, o, u, mask, train):
    #     """ Backprop through time """
    #     z, kl = self.agent.encode(o, u)
    #     z_sample = self.agent.sample_z((o.shape[1],)).squeeze(-2)
        
    #     hidden = None
    #     for t, batch_t in enumerate(zip(
    #         o.split(self.bptt_steps, dim=0),
    #         u.split(self.bptt_steps, dim=0),
    #         mask.split(self.bptt_steps, dim=0)
    #         )):
    #         o_t, u_t, mask_t = batch_t

    #         if hidden is None:
    #             out, hidden = self.agent(o_t, u_t, z)

    #         else:
    #             # concat previous ctl
    #             u_t_cat = torch.cat([u_t_prev[-1:], u_t[1:]], dim=0)

    #             hidden = [h[-1].detach() for h in hidden]
                
    #             out, hidden = self.agent(o_t, u_t_cat, z, hidden)
    #         u_t_prev = u_t.clone()

    #         loss_u, stats_u = self.agent.act_loss(o_t, u_t, z, mask_t, hidden)
    #         if self.sample_z:
    #             loss_o, stats_o = self.agent.obs_loss(o_t, u_t, z_sample, mask_t, hidden)
    #         else:
    #             loss_o, stats_o = self.agent.obs_loss(o_t, u_t, z, mask_t, hidden)
    #         loss_kl = torch.mean(kl) / o.shape[0]
    #         loss_prior = self.compute_prior_loss(z)
    #         loss = torch.mean(self.bc_penalty * loss_u + self.obs_penalty * loss_o) + loss_kl + self.reg_penalty * loss_prior
                
    #         if train:
    #             loss.backward()
    #             if self.grad_clip is not None:
    #                 nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

    #             self.optimizer.step()
    #         self.optimizer.zero_grad()

    #         stats = {
    #             "train": 1 if train else 0,
    #             "loss": loss.cpu().data.item(),
    #             **stats_u, **stats_o,
    #         }
    #     return stats