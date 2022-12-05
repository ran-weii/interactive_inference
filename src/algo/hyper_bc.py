import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as torch_dist
from src.distributions.nn_models import Model

class HyperBehaviorCloning(Model):
    def __init__(
        self, agent, train_mode, detach=True, bptt_steps=30, pred_steps=5,
        bc_penalty=1., obs_penalty=0., pred_penalty=0., reg_penalty=0., 
        ortho_penalty=1., post_obs_penalty=0., kl_penalty=1., 
        lr=1e-2, lr_flow=1e-3, lr_post=1e-3, 
        decay=0, grad_clip=None, decay_steps=100, decay_rate=0.8
        ):
        super().__init__()
        assert agent.__class__.__name__ == "HyperVINAgent"
        assert train_mode in ["prior", "post", "marginal"]
        self.train_mode = train_mode
        self.detach = detach
        self.bptt_steps = bptt_steps
        self.pred_steps = pred_steps
        self.bc_penalty = bc_penalty
        self.obs_penalty = obs_penalty
        self.pred_penalty = pred_penalty
        self.reg_penalty = reg_penalty
        self.ortho_penalty = ortho_penalty
        self.post_obs_penalty = post_obs_penalty
        self.kl_penalty = kl_penalty

        self.lr = lr
        self.lr_flow = lr_flow
        self.lr_post = lr_post
        self.decay = decay
        self.grad_clip = grad_clip
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        self.agent = agent

        self._set_params_grad()

        # group parameters by flow parameters
        prior_params = [p for n, p in self.agent.named_parameters() if "encoder" not in n and "arn" not in n]
        post_params = [p for n, p in self.agent.named_parameters() if "encoder" in n]
        flow_params = [p for n, p in self.agent.named_parameters() if "arn" in n]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": prior_params, "lr": lr},
                {"params": flow_params, "lr": lr_flow},
                {"params": post_params, "lr": lr_post},
            ], 
            lr=lr, weight_decay=decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=decay_steps, gamma=decay_rate, last_epoch=-1, verbose=False
        )
        self.loss_keys = ["total_loss", "loss_u", "loss_o"]
        if self.train_mode != "prior":
            self.loss_keys.append("post_kl")
    
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
        s = "{}(train_mode={}, detach={}, bptt_steps={}, pred_steps={}, "\
        "bc_penalty={}, obs_penalty={}, pred_penalty={}, reg_penalty={}, "\
        "ortho_peanalty={}, post_obs_penalty={}, kl_penalty={}, lr={}, lr_flow={}, lr_post={}, "\
        "decay={}, grad_clip={}, decay_steps={}, decay_rate={},\nagent={})".format(
            self.__class__.__name__, self.train_mode, self.detach, self.bptt_steps, self.pred_steps,
            self.bc_penalty, self.obs_penalty, self.pred_penalty, self.reg_penalty, 
            self.ortho_penalty, self.post_obs_penalty, self.kl_penalty, self.lr, self.lr_flow, self.lr_post, 
            self.decay, self.grad_clip, self.decay_steps, self.decay_rate, s_agent
        )
        return s
    
    def stdout(self, train_stats, test_stats):
        s = "total_loss: {:.4f}/{:.4f}, loss_u: {:.4f}/{:.4f}, loss_o: {:.4f}/{:.4f}".format(
            train_stats["total_loss"], test_stats["total_loss"], 
            train_stats["loss_u"], test_stats["loss_u"], 
            train_stats["loss_o"], test_stats["loss_o"]
        )
        if self.train_mode != "prior":
            s = " ".join([s, "post_kl: {:.4f}/{:.4f}".format(train_stats["post_kl"], test_stats["post_kl"])])
        return s
    
    def compute_ortho_loss(self):
        """ Compute factor orthogonality loss """
        weights = []
        for n, p in self.named_parameters():
            if "weight" in n and "arn" not in n and "encoder" not in n:
                weights.append(p)
        weights = torch.cat(weights, dim=0)
        ortho_loss = torch.pow(
            weights.T.matmul(weights) - torch.eye(weights.shape[1], device=self.device), 2
        ).sum()
        
        stats = {
            "ortho_loss": ortho_loss.data.item()
        }
        return ortho_loss, stats

    def compute_reg_loss(self, z):
        """ Regularization loss """
        # obs variance
        obs_vars = self.agent.obs_model.lv(z).exp()
        obs_loss = torch.sum(obs_vars ** 2, dim=[-1, -2]).mean()

        # ctl variance
        ctl_vars = self.agent.ctl_model.lv.exp()
        ctl_loss = torch.sum(ctl_vars ** 2)
        
        loss = obs_loss + ctl_loss 

        stats = {
            "obs_var": obs_vars.mean().data.item(),
            "ctl_var": ctl_vars.mean().data.item(),
        }
        return loss, stats
    
    def compute_prior_loss(self, o, u, mask):
        """ KL loss for fitting maximum entropy hyper prior """
        prior_dist = self.agent.get_prior_dist()
        z = prior_dist.rsample()
        z = torch.repeat_interleave(z, o.shape[1], dim=0)
        
        _, hidden = self.agent.forward(o, u, z, detach=self.detach)
        act_loss, act_stats = self.agent.act_loss(o, u, z, mask, hidden)
        obs_loss, obs_stats = self.agent.obs_loss(o, u, z, mask, hidden)
        
        reg_loss, reg_stats = self.compute_reg_loss(z)
        
        loss = (
            self.bc_penalty * act_loss.mean() + \
            self.obs_penalty * obs_loss.mean() + \
            self.reg_penalty * reg_loss
        )

        stats = {
            "total_loss": loss.data.item(),
            **act_stats, **obs_stats, **reg_stats
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
        reg_loss, reg_stats = self.compute_reg_loss(z)

        loss = torch.mean(
            act_loss * mask.sum(0) + \
            self.obs_penalty * obs_loss * mask.sum(0) + \
            self.reg_penalty * reg_loss + \
            self.kl_penalty * kl
        ) 
        
        stats = {
            "total_loss": loss.data.item(),
            **act_stats, **obs_stats, **reg_stats,
            "post_kl": kl.data.mean().item()
        }
        return loss, stats
    
    def compute_marginal_loss(self, o, u, mask):
        """ Elbo loss for marginal likelihood """
        prior_dist = self.agent.get_prior_dist()
        z_prior = prior_dist.rsample()
        z_prior = torch.repeat_interleave(z_prior, o.shape[1], dim=0)
        
        post_dist = self.agent.get_posterior_dist(o, u, mask)
        z_post = post_dist.rsample()
        
        # compute prior loss
        _, hidden_prior = self.agent.forward(o, u, z_prior, detach=self.detach)
        act_loss_prior, act_stats_prior = self.agent.act_loss(o, u, z_prior, mask, hidden_prior)
        obs_loss_prior, obs_stats_prior = self.agent.obs_loss(o, u, z_prior, mask, hidden_prior)
        
        # compute posterior loss
        _, hidden_post = self.agent.forward(o, u, z_post, detach=False)
        act_loss_post, act_stats_post = self.agent.act_loss(o, u, z_post, mask, hidden_post)
        obs_loss_post, obs_stats_post = self.agent.obs_loss(o, u, z_post, mask, hidden_post, pred_steps=1)
        pred_loss_post, pred_stats_post = self.agent.obs_loss(o, u, z_post, mask, hidden_post, pred_steps=self.pred_steps)
        kl = torch_dist.kl.kl_divergence(post_dist, prior_dist).sum(-1)
        reg_loss_post, reg_stats = self.compute_reg_loss(z_post)
        ortho_loss, ortho_stats = self.compute_ortho_loss()
        
        prior_loss = (
            self.bc_penalty * torch.mean(act_loss_prior) + \
            self.obs_penalty * torch.mean(obs_loss_prior)
        )
        post_loss = (
            torch.mean(act_loss_post) + \
            self.post_obs_penalty * torch.mean(obs_loss_post) + \
            self.pred_penalty * torch.mean(pred_loss_post) + \
            self.kl_penalty * torch.mean(kl) + \
            self.reg_penalty * reg_loss_post + \
            self.ortho_penalty * ortho_loss
        )
        loss = prior_loss + post_loss
        
        pred_stats_post["loss_pred"] = pred_stats_post.pop("loss_o")
        prior_stats = {
            "total_loss": prior_loss.data.item(),
            **act_stats_prior, **obs_stats_prior
        }
        prior_stats = {f"prior_{k}": v for (k, v) in prior_stats.items()}
        post_stats = {
            "total_loss": post_loss.data.item(),
            **act_stats_post, **obs_stats_post,
            "kl": kl.data.mean().item(),
            **pred_stats_post, **reg_stats, **ortho_stats
        }
        post_stats = {f"post_{k}": v for (k, v) in post_stats.items()}
        stats = {
            "total_loss": loss.data.item(),
            **act_stats_post, ** obs_stats_post,
            **prior_stats, **post_stats
        }
        return loss, stats

    def compute_loss(self, o, u, mask):
        if self.train_mode == "prior":
            loss, stats = self.compute_prior_loss(o, u, mask)
        elif self.train_mode == "post":
            loss, stats = self.compute_posterior_loss(o, u, mask)
        elif self.train_mode == "marginal":
            loss, stats = self.compute_marginal_loss(o, u, mask)
        return loss, stats

    def run_epoch(self, loader, train=True):
        if train:
            self.agent.train()
        else:
            self.agent.eval()
        
        epoch_stats = []
        for i, batch in enumerate(loader):
            pad_batch, mask = batch
            o = pad_batch["obs"]
            u = pad_batch["act"]

            o = o.to(self.device)
            u = u.to(self.device)
            mask = mask.to(self.device)
            
            loss, stats = self.compute_loss(o, u, mask)
            
            if train:
                loss.backward()
                
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                
                self.optimizer.step()
            self.optimizer.zero_grad()
            
            stats["train"] = 1 if train else 0
            epoch_stats.append(stats)
        
        if train:
            self.scheduler.step()
        stats = pd.DataFrame(epoch_stats).mean().to_dict()
        return stats
