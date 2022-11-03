import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.distributions.nn_models import Model
from src.agents.vin_agent import VINAgent

class BehaviorCloning(Model):
    """ Behavior cloning algorithm for fully observable policies """
    def __init__(
        self, agent, lr=1e-3, decay=0, grad_clip=None
        ):
        super().__init__()
        self.lr = lr
        self.decay = decay
        self.grad_clip = grad_clip
        self.agent = agent
        
        self.optimizer = torch.optim.AdamW(
            self.agent.parameters(), 
            lr=lr, weight_decay=decay
        )
        self.loss_keys = ["loss"]
    
    def __repr__(self):
        s_agent = self.agent.__repr__()
        s = "{}(lr={}, decay={}, grad_clip={},\nagent={})".format(
            self.__class__.__name__, self.lr, self.decay, self.grad_clip, s_agent
        )
        return s
    
    def stdout(self, train_stats, test_stats):
        s = "loss: {:.4f}/{:.4f}".format(train_stats["loss"], test_stats["loss"])
        return s
    
    def compute_prior_loss(self, o, u, mask, hidden):
        if isinstance(self.agent, VINAgent):
            # obs variance
            obs_vars = self.agent.obs_model.lv.exp()
            obs_loss = torch.sum(obs_vars ** 2)

            # ctl variance
            ctl_vars = self.agent.ctl_model.lv.exp()
            ctl_loss = torch.sum(ctl_vars ** 2)
            
            loss = obs_loss + ctl_loss

            stats = {
                "obs_var": obs_vars.mean().data.item(),
                "ctl_var": ctl_vars.mean().data.item(),
            }
        else:
            loss = torch.zeros(1)
            stats = {}
        return loss, stats

    def run_epoch(self, loader, train=True):
        if train:
            self.agent.train()
        else:
            self.agent.eval()
        
        epoch_stats = []
        for i, batch in enumerate(loader):
            o = batch["obs"]
            u = batch["act"]
            mask = torch.ones(o.shape[0], device=o.device)

            o = o.to(self.device)
            u = u.to(self.device)
            mask = mask.to(self.device)
            
            # ensure o is not recurrent
            assert len(o.shape) == 2
            
            _, out = self.agent.forward(o)
            loss, stats = self.agent.act_loss(o, u, mask, out)
            
            if train:
                loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

                self.optimizer.step()
            self.optimizer.zero_grad()
            
            epoch_stats.append({
                "train": 1 if train else 0,
                "loss": loss.cpu().data.item(),
            }) 
        
        stats = pd.DataFrame(epoch_stats).mean().to_dict()
        return stats


class RecurrentBehaviorCloning(Model):
    """ Behavior cloning algorithm for recurrent policies
    with truncated backpropagation through time
    """
    def __init__(
        self, agent, bptt_steps=30, pred_steps=5, 
        bc_penalty=1., obs_penalty=0., pred_penalty=0., reg_penalty=0., 
        lr=1e-3, lr_flow=1e-3, decay=0, grad_clip=None
        ):
        super().__init__()
        self.bptt_steps = bptt_steps
        self.pred_steps = pred_steps
        self.bc_penalty = bc_penalty
        self.obs_penalty = obs_penalty
        self.pred_penalty = pred_penalty
        self.reg_penalty = reg_penalty
        self.lr = lr
        self.lr_flow = lr_flow
        self.decay = decay
        self.grad_clip = grad_clip
        self.agent = agent
        
        # group parameters by flow parameters
        param_group1 = [p for n, p in self.agent.named_parameters() if "arn" not in n]
        param_group2 = [p for n, p in self.agent.named_parameters() if "arn" in n]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": param_group1},
                {"params": param_group2, "lr": lr_flow}
            ], 
            lr=lr, weight_decay=decay
        )
        self.loss_keys = ["loss_u", "loss_o"]
    
    def __repr__(self):
        s_agent = self.agent.__repr__()
        s = "{}(bptt_steps={}, pred_steps={}, bc_penalty={}, obs_penalty={}, pred_penalty={}, "\
        "reg_penalty={}, lr={}, lr_flow={}, decay={}, grad_clip={},\nagent={})".format(
            self.__class__.__name__, self.bptt_steps, self.pred_steps, 
            self.bc_penalty, self.obs_penalty, self.pred_penalty, self.reg_penalty, 
            self.lr, self.lr_flow, self.decay, self.grad_clip, s_agent
        )
        return s
    
    def stdout(self, train_stats, test_stats):
        s = "loss_u: {:.4f}/{:.4f}, loss_o: {:.4f}/{:.4f}".format(
            train_stats["loss_u"], test_stats["loss_u"], 
            train_stats["loss_o"], test_stats["loss_o"]
        )
        return s
    
    def compute_prior_loss(self, o, u, mask, hidden):
        if isinstance(self.agent, VINAgent):
            # obs variance
            obs_vars = self.agent.obs_model.lv.exp()
            obs_loss = torch.sum(obs_vars ** 2)

            # ctl variance
            ctl_vars = self.agent.ctl_model.lv.exp()
            ctl_loss = torch.sum(ctl_vars ** 2)
            
            loss = obs_loss + ctl_loss

            stats = {
                "obs_var": obs_vars.mean().data.item(),
                "ctl_var": ctl_vars.mean().data.item(),
            }
        else:
            loss = torch.zeros(1)
            stats = {}
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
            
            hidden = [None]
            for t, batch_t in enumerate(zip(
                o.split(self.bptt_steps, dim=0),
                u.split(self.bptt_steps, dim=0),
                mask.split(self.bptt_steps, dim=0)
                )):
                o_t, u_t, mask_t = batch_t
                
                out, hidden = self.agent.forward(o_t, u_t, hidden=hidden)
                
                pred_steps = self.pred_steps if len(o_t) > self.pred_steps else 1
                loss_u, stats_u = self.agent.act_loss(o_t, u_t, mask_t, hidden)
                loss_o, stats_o = self.agent.obs_loss(o_t, u_t, mask_t, hidden, pred_steps=1)
                loss_pred, stats_pred = self.agent.obs_loss(o_t, u_t, mask_t, hidden, pred_steps=pred_steps)
                loss_prior, stats_prior = self.compute_prior_loss(o_t, u_t, mask_t, hidden)
                
                loss_u = torch.mean(loss_u)
                loss_o = torch.mean(loss_o)
                loss_pred = torch.mean(loss_pred)
                
                loss = (
                    self.bc_penalty * loss_u + \
                    self.obs_penalty * loss_o + \
                    self.pred_penalty * loss_pred + \
                    self.reg_penalty * loss_prior
                )
                
                # update running hidden
                hidden = [h[-1].detach() for h in hidden]

                if train:
                    loss.backward()
                    if self.grad_clip is not None:
                        nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

                    self.optimizer.step()
                self.optimizer.zero_grad()
                
                stats_pred["loss_pred"] = stats_pred.pop("loss_o")
                epoch_stats.append({
                    "train": 1 if train else 0,
                    "loss": loss.cpu().data.item(),
                    **stats_u, **stats_o, **stats_pred, **stats_prior
                }) 
        
        stats = pd.DataFrame(epoch_stats).mean().to_dict()
        return stats