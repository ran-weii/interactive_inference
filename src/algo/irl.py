import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class BehaviorCloning(nn.Module):
    """ Supervised behavior cloning algorithm """
    def __init__(self, agent, obs_penalty=0, lr=1e-3, decay=0, grad_clip=None):
        super().__init__()
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
        s = "{}(obs_penalty={}, lr={}, decay={}, grad_clip={},\nagent={})".format(
            self.__class__.__name__, self.obs_penalty, self.lr, 
            self.decay, self.grad_clip, s_agent
        )
        return s

    def run_epoch(self, loader, train=True):
        if train:
            self.agent.train()
        else:
            self.agent.eval()
        
        epoch_stats = []
        for i, batch in enumerate(loader):
            o, u, mask = batch
            out = self.agent(o, u)
            loss_u = self.agent.act_loss(o, u, mask, out)
            loss_o = self.agent.obs_loss(o, u, mask, out)

            loss = torch.mean(loss_u + self.obs_penalty * loss_o)

            if train:
                loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

                self.optimizer.step()
            self.optimizer.zero_grad()

            epoch_stats.append({
                "loss": loss.data.item(),
                "loss_u": loss_u.data.mean().item(),
                "loss_o": loss_o.data.mean().item(),
                "train": 1 if train else 0
            })
        
        df_stats = pd.DataFrame(epoch_stats).mean()
        return df_stats