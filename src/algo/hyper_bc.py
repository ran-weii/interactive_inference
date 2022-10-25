import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as torch_dist
from src.distributions.nn_models import Model

class HyperBehaviorCloning(Model):
    def __init__(
        self, agent, train_mode, detach=True, bptt_steps=30, 
        bc_penalty=1., obs_penalty=0., reg_penalty=0., 
        post_obs_penalty=0., kl_penalty=1.,
        lr=1e-2, lr_post=1e-3, decay=0, grad_clip=None
        ):
        super().__init__()
        assert agent.__class__.__name__ == "HyperVINAgent"
        assert train_mode in ["prior", "post", "marginal"]
        self.train_mode = train_mode
        self.detach = detach
        self.bptt_steps = bptt_steps
        self.bc_penalty = bc_penalty
        self.obs_penalty = obs_penalty
        self.reg_penalty = reg_penalty
        self.post_obs_penalty = post_obs_penalty
        self.kl_penalty = kl_penalty

        self.lr = lr
        self.lr_post = lr_post
        self.decay = decay
        self.grad_clip = grad_clip

        self.agent = agent

        self._set_params_grad()
        
        prior_params = [p for (n, p) in self.agent.named_parameters() if "encoder" not in n]
        post_params = [p for (n, p) in self.agent.named_parameters() if "encoder" in n]
        self.optimizers = [
            torch.optim.AdamW(prior_params, lr=lr, weight_decay=decay),
            torch.optim.AdamW(post_params, lr=lr_post, weight_decay=decay),
        ]
        self.loss_keys = ["total_loss", "loss_u", "loss_o"]
    
    def _set_params_grad(self):
        if self.train_mode == "prior":
            for n, p in self.named_parameters():
                if "encoder" in n:
                    p.requires_grad = False
                if "weight" in n:
                    p.requires_grad = False

        elif self.train_mode == "post":
            for n, p in self.named_parameters():
                if "encoder" not in n:
                    p.requires_grad = False

    def __repr__(self):
        s_agent = self.agent.__repr__()
        s = "{}(train_mode={}, detach={}, bptt_steps={}, bc_penalty={}, obs_penalty={}, reg_penalty={}, "\
        "post_obs_penalty={}, kl_penalty={}, lr={}, lr_post={}, decay={}, grad_clip={},\nagent={})".format(
            self.__class__.__name__, self.train_mode, self.detach, self.bptt_steps, 
            self.bc_penalty, self.obs_penalty, self.reg_penalty, 
            self.post_obs_penalty, self.kl_penalty,
            self.lr, self.lr_post, self.decay, self.grad_clip, s_agent
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
        obs_vars = self.agent.obs_model.lv(z).exp()
        obs_loss = torch.sum(obs_vars ** 2, dim=[-1, -2]).mean()

        # ctl variance
        ctl_vars = self.agent.ctl_model.lv.exp()
        ctl_loss = torch.sum(ctl_vars ** 2)
        
        loss = obs_loss + ctl_loss 
        return loss

    def compute_prior_loss(self, o, u, mask, num_total_eps):
        """ KL loss for fitting maximum entropy hypernet prior """
        prior_dist = self.agent.get_prior_dist()
        z = prior_dist.rsample()
        z = torch.repeat_interleave(z, o.shape[1], dim=0)
        
        _, hidden = self.agent.forward(o, u, z, detach=self.detach)
        act_loss, act_stats = self.agent.act_loss(o, u, z, mask, hidden)
        obs_loss, obs_stats = self.agent.obs_loss(o, u, z, mask, hidden)
        
        # ent = self.agent.parameter_entropy(z).mean()
        reg_loss = self.compute_reg_loss(z)
        
        avg_eps_len = torch.mean(mask.sum(0))
        loss = avg_eps_len * num_total_eps * (
            self.bc_penalty * act_loss.mean() + \
            self.obs_penalty * obs_loss.mean() + \
            self.reg_penalty * reg_loss
        )# - ent

        stats = {
            "total_loss": loss.data.item(),
            **act_stats, **obs_stats,
            # "ent": ent.data.item()
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

        loss = torch.mean(
            act_loss * mask.sum(0) + \
            self.obs_penalty * obs_loss * mask.sum(0) + \
            self.reg_penalty * reg_loss + \
            self.kl_penalty * kl
        ) 
        
        stats = {
            "total_loss": loss.data.item(),
            **act_stats, **obs_stats,
            "kl": kl.data.mean().item()
        }
        return loss, stats
    
    def compute_marginal_loss(self, o, u, mask, num_total_eps):
        """ Elbo loss for marginal likelihood """
        prior_dist = self.agent.get_prior_dist()
        z_prior = prior_dist.rsample()
        z_prior = torch.repeat_interleave(z_prior, o.shape[1], dim=0)
        
        post_dist = self.agent.get_posterior_dist(o, u, mask)
        z_post = post_dist.rsample()

        _, hidden_prior = self.agent.forward(o, u, z_prior, detach=True)
        act_loss_prior, act_stats_prior = self.agent.act_loss(o, u, z_prior, mask, hidden_prior)
        obs_loss_prior, obs_stats_prior = self.agent.obs_loss(o, u, z_prior, mask, hidden_prior)

        _, hidden_post = self.agent.forward(o, u, z_post, detach=False)
        act_loss_post, act_stats_post = self.agent.act_loss(o, u, z_post, mask, hidden_post)
        obs_loss_post, obs_stats_post = self.agent.obs_loss(o, u, z_post, mask, hidden_post)
        kl = torch_dist.kl.kl_divergence(post_dist, prior_dist).sum(-1)
        pred_loss_post, pred_stats_post = self.agent.compute_prediction_loss(o, u, z_post, mask, hidden_post)
        reg_loss_post = self.compute_reg_loss(z_post)
        
        avg_eps_len = torch.mean(mask.sum(0))
        prior_loss = avg_eps_len * num_total_eps * (
            self.bc_penalty * act_loss_prior.mean() + \
            self.obs_penalty * obs_loss_prior.mean()
        )
        post_loss = num_total_eps * (
            torch.mean(act_loss_post * mask.sum(0)) + \
            self.post_obs_penalty * torch.mean(obs_loss_post * mask.sum(0)) + \
            self.post_obs_penalty * self.reg_penalty * torch.mean(pred_loss_post * mask.sum(0)) + \
            self.kl_penalty * torch.mean(kl) + self.reg_penalty * reg_loss_post
        )
        loss = prior_loss + post_loss
        
        prior_stats = {
            "total_loss": prior_loss.data.item(),
            **act_stats_prior, **obs_stats_prior
        }
        prior_stats = {f"prior_{k}": v for (k, v) in prior_stats.items()}
        post_stats = {
            "total_loss": post_loss.data.item(),
            **act_stats_post, **obs_stats_post,
            "kl": kl.data.mean().item(),
            "reg_loss": reg_loss_post.data.item(),
            **pred_stats_post
        }
        post_stats = {f"post_{k}": v for (k, v) in post_stats.items()}
        stats = {
            "total_loss": loss.data.item(),
            **act_stats_post, ** obs_stats_post,
            **prior_stats, **post_stats
        }
        return loss, stats

    def compute_loss(self, o, u, mask, num_total_eps):
        if self.train_mode == "prior":
            loss, stats = self.compute_prior_loss(o, u, mask, num_total_eps)
        elif self.train_mode == "post":
            loss, stats = self.compute_posterior_loss(o, u, mask)
        elif self.train_mode == "marginal":
            loss, stats = self.compute_marginal_loss(o, u, mask, num_total_eps)
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
            
            loss, stats = self.compute_loss(o, u, mask, len(loader.dataset))
            
            if train:
                loss.backward()
                # for n, p in self.named_parameters():
                #     if p.grad is None:
                #         print(n, p.requires_grad, None)
                #     else:
                #         print(n, p.requires_grad, p.grad.data.norm())
                # exit()
                
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                
                for optimizer in self.optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
            
            stats["train"] = 1 if train else 0
            epoch_stats.append(stats)

        stats = pd.DataFrame(epoch_stats).mean().to_dict()
        return stats
