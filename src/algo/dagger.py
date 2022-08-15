import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.distributions.nn_models import Model 

class Dagger(Model):
    """ Offline dagger algorithm with counterfactual teaching
    """
    def __init__(
        self, agent, expert, buffer, bptt_steps=30, obs_penalty=0, 
        cf_penalty=1., lr=1e-3, decay=0, grad_clip=None):
        super().__init__()
        self.bptt_steps = bptt_steps
        self.obs_penalty = obs_penalty
        self.cf_penalty = cf_penalty
        self.lr = lr
        self.decay = decay
        self.grad_clip = grad_clip
        self.agent = agent
        self.expert = expert
        self.buffer = buffer
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=lr, weight_decay=decay
        )
        self.loss_keys = ["loss_u", "loss_o", "loss_u_cf", "loss_o_cf"]
    
    def __repr__(self):
        s_agent = self.agent.__repr__()
        s = "{}(bptt_steps={}, obs_penalty={}, cf_penalty={}, lr={}, decay={}, grad_clip={},\nagent={})".format(
            self.__class__.__name__, self.bptt_steps, self.obs_penalty, self.cf_penalty, self.lr, 
            self.decay, self.grad_clip, s_agent
        )
        return s
    
    def stdout(self, train_stats, test_stats):
        s = "loss_u: {:.4f}/{:.4f}, loss_u_cf: {:.4f}/{:.4f}, loss_o: {:.4f}/{:.4f}, loss_o_cf: {:.4f}/{:.4f}".format(
            train_stats["loss_u"], test_stats["loss_u"], 
            train_stats["loss_u_cf"], test_stats["loss_u_cf"], 
            train_stats["loss_o"], test_stats["loss_o"],
            train_stats["loss_o_cf"], test_stats["loss_o_cf"],
        )
        return s
    
    def compute_counterfactual_loss(self, batch_size, num_steps):
        pad_batch, mask = self.buffer.sample_episodes(batch_size, num_steps)
        o = pad_batch["obs"].to(self.device)
        u = pad_batch["ctl"].to(self.device)
        
        with torch.no_grad():
            u_expert, logp_expert = self.expert.choose_action(o)
            u_expert = u_expert.squeeze(0)
        
        out, hidden = self.agent(o, u)
        loss_o, stats_o = self.agent.obs_loss(o, u, mask, hidden)
        loss_u, stats_u = self.agent.act_loss(o, u_expert, mask, hidden)

        stats_o["loss_o_cf"] = stats_o.pop("loss_o")
        stats_u["loss_u_cf"] = stats_u.pop("loss_u")
        return loss_u, loss_o, stats_u, stats_o

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
                loss_u_cf, loss_o_cf, stats_u_cf, stats_o_cf = self.compute_counterfactual_loss(o_t.shape[1], o_t.shape[0])

                loss = loss_u + self.obs_penalty/2 * (loss_o + loss_o_cf) + self.cf_penalty * loss_u_cf
                    
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
                    **stats_u_cf, **stats_o_cf
                })
        
        stats = pd.DataFrame(epoch_stats).mean().to_dict()
        return stats
