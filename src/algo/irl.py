import pandas as pd
import torch
import torch.nn as nn
from src.distributions.nn_models import Model

class BehaviorCloning(Model):
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
            o, u, mask = batch
            o = o.to(self.device)
            u = u.to(self.device)
            mask = mask.to(self.device)
            
            out = self.agent(o, u)
            loss_u, stats_u = self.agent.act_loss(o, u, mask, out)
            loss_o, stats_o = self.agent.obs_loss(o, u, mask, out)

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