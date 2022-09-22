import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from CGDs import ACGD

from src.agents.models import MLP

class ReplayBuffer():
    def __init__(self, state_dim, act_dim, obs_dim, max_size):
        self.state = np.array([]).reshape(0, state_dim)
        self.act = np.array([]).reshape(0, act_dim)
        self.obs = np.array([]).reshape(0, obs_dim)

        self.pointer = 0
        self.size = 0
        self.max_size = max_size

    def push(self, state, act, obs):
        self.state = np.concatenate([state, self.state], axis=0)
        self.act = np.concatenate([act, self.act], axis=0)
        self.obs = np.concatenate([obs, self.obs], axis=0)

        if len(self.state) > self.max_size:
            self.state = self.state[:self.max_size]
            self.act = self.act[:self.max_size]
            self.obs = self.obs[:self.max_size]
        self.size = len(self.state)

    def sample_batch(self, batch_size=32):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            state=self.state[idx],
            act=self.act[idx],
            obs=self.obs[idx],
        )
        return {k: torch.from_numpy(v).to(torch.float32) for k, v in batch.items()}


class ImplicitIRL(nn.Module):
    def __init__(
        self, agent, obs_penalty=0, ld=1, lr_primal=0.001, lr_dual=0.001, 
        decay=0, grad_clip=None, plan_batch_size=100, buffer_size=100000
        ):
        super().__init__()
        self.obs_penalty = obs_penalty
        self.lr_dual = lr_dual
        self.lr_primal = lr_primal
        self.decay = decay
        self.grad_clip = grad_clip
        self.plan_batch_size = plan_batch_size

        self.agent = agent
        self.buffer = ReplayBuffer(
            agent.state_dim, agent.ctl_dim, agent.obs_dim, buffer_size
        )
        self.ld = nn.Parameter(ld * torch.ones(1))
        
        self.optimizer = ACGD(
            max_params=[self.ld],
            min_params=self.agent.parameters(),
            lr_max=lr_dual,
            lr_min=lr_primal
        )
        
        self.loss_keys = ["loss_pi", "loss_obs", "loss_td", "ld"]
    
    def fill_buffer(self, o, u, b, mask):
        # flatten 
        mask_flat = mask.flatten(0, 1)
        b_flat = b.data.flatten(0, 1)[mask_flat == 1].numpy()
        o_flat = o.data.flatten(0, 1)[mask_flat == 1].numpy()
        u_flat = u.data.flatten(0, 1)[mask_flat == 1].numpy()
        
        self.buffer.push(b_flat, u_flat, o_flat)

    def train_epoch(self, loader):
        self.train()
        
        epoch_stats = []
        for i, batch in enumerate(loader):
            o, u, mask = batch
            logp_pi, logp_obs, b = self.agent(o, u)
            self.fill_buffer(o, u, b, mask)
            
            loss, stats = self.loss(logp_pi, logp_obs, mask)
            self.optimizer.step(loss)
            epoch_stats.append(stats)
        
        df_stats = pd.DataFrame(epoch_stats).mean()
        return df_stats

    def test_epoch(self, loader):
        epoch_stats = []
        for i, batch in enumerate(loader):
            o, u, mask = batch
            with torch.no_grad():
                logp_pi, logp_obs, b = self.agent(o, u)
                loss, stats = self.loss(logp_pi, logp_obs, mask)
            epoch_stats.append(stats)
            
        df_stats = pd.DataFrame(epoch_stats).mean()
        return df_stats 

    def td_loss(self, sample_batch):
        b = sample_batch["state"]
        a = sample_batch["act"]
        
        # compute target with stratified sampling
        transition = self.agent.hmm.get_transition_matrix(a)
        s_next = torch.sum(transition * b.unsqueeze(-1), dim=-2)
        o_next = self.agent.hmm.obs_model.sample((self.plan_batch_size,)).squeeze(-3)
        
        # compute expected posterior 
        logp_o = self.agent.hmm.obs_model.log_prob(o_next)
        logp_s = torch.log(s_next + 1e-6).unsqueeze(-2)
        b_next = torch.softmax(logp_s + logp_o, dim=-1)
        
        # compute value target
        r_next = self.agent.rwd_model(o_next, self.agent.hmm.obs_model.bn)
        v_next = self.agent.planner.v(b_next).squeeze(-1)
        q_target = torch.sum(s_next * (r_next + self.agent.tau * v_next), dim=-1)

        q = self.agent.planner.q(b, a).squeeze(-1)
        td_error = torch.pow(q - q_target, 2)
        return td_error

    def loss(self, logp_pi, logp_obs, mask):
        # data loss
        loss_pi = torch.sum(logp_pi * mask, dim=0) / mask.sum(0)
        loss_obs = torch.sum(logp_obs * mask, dim=0) / mask.sum(0)
        
        loss_pi = -torch.mean(loss_pi)
        loss_obs = -torch.mean(loss_obs)
        
        # irl loss
        batch = self.buffer.sample_batch(self.plan_batch_size)
        td_error = self.td_loss(batch)
        loss_td = torch.mean(td_error)

        loss = loss_pi + self.obs_penalty * loss_obs + self.ld * loss_td

        stats_dict = {
            "loss": loss.data.numpy(),
            "loss_pi": loss_pi.data.numpy(),
            "loss_obs": loss_obs.data.numpy(),
            "loss_td": loss_td.data.numpy(),
            "ld": self.ld.data.numpy(),
        }
        return loss, stats_dict
    
    def print_message(self, stats):
        s = "loss_pi: {:.4f}, loss_obs: {:.4f}, loss_td: {:.4f}, ld: {:.4f}".format(
            stats["loss_pi"],
            stats["loss_obs"],
            stats["loss_td"],
            stats["ld"],
        )
        return s

class ReverseKL(ImplicitIRL):
    def __init__(
        self, agent, obs_penalty=0, ld=1, lr_primal=0.001, lr_dual=0.001, lr_d=0.001,
        e_steps=1, decay=0, grad_clip=None, plan_batch_size=100, buffer_size=100000
        ):
        super().__init__(
            agent, obs_penalty, ld, lr_primal, lr_dual, 
            decay, grad_clip, plan_batch_size, buffer_size
        )
        self.e_steps = e_steps
        
        self.discriminator = MLP(
            input_dim=self.agent.state_dim + self.agent.obs_dim + self.agent.ctl_dim,
            output_dim=1,
            hidden_dim=64,
            num_hidden=2,
            activation="silu"
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr_d, weight_decay=decay
        )

        self.loss_keys = ["loss_r", "loss_d", "loss_pi", "loss_obs", "loss_td", "ld"]
    
    def train_epoch(self, loader):
        self.train()
        
        epoch_stats = []
        for i, batch in enumerate(loader):
            o, u, mask = batch
            b = self.agent.hmm._forward(o, u)
            self.fill_buffer(o, u, b, mask)
            
            # train discriminator
            d_stats = []
            for e in range(self.e_steps):
                loss_d = self.discriminator_loss()

                loss_d.backward()
                self.d_optimizer.step()
                self.d_optimizer.zero_grad()
                d_stats.append(loss_d.data.item())
            d_stats = np.mean(d_stats)
            
            # train actor critic
            loss, stats = self.loss(o, u, b, mask)
            self.optimizer.step(loss)
            self.d_optimizer.zero_grad()

            stats["loss_d"] = d_stats
            epoch_stats.append(stats)
        
        df_stats = pd.DataFrame(epoch_stats).mean()
        return df_stats

    def test_epoch(self, loader):
        epoch_stats = []
        for i, batch in enumerate(loader):
            o, u, mask = batch
            with torch.no_grad():
                b = self.agent.hmm._forward(o, u)
                loss_d = self.discriminator_loss()
                loss, stats = self.loss(o, u, b, mask)
                stats["loss_d"] = loss_d.data.item()
            epoch_stats.append(stats)
            
        df_stats = pd.DataFrame(epoch_stats).mean()
        return df_stats 
    
    def discriminator_loss(self):
        batch = self.buffer.sample_batch(self.plan_batch_size)
        b = batch["state"]
        a_real = batch["act"]
        o = batch["obs"]
        
        # sample fake actions
        with torch.no_grad():
            a_fake = self.agent.planner.sample(b)
        
        real_inputs = torch.cat([b, o, a_real], dim=-1)
        fake_inputs = torch.cat([b, o, a_fake], dim=-1)
        inputs = torch.cat([real_inputs, fake_inputs], dim=0)
        labels = torch.cat([
            torch.zeros(self.plan_batch_size, 1), 
            torch.ones(self.plan_batch_size, 1)
        ], dim=0)
        
        pred = torch.sigmoid(self.discriminator(inputs))
        loss = F.binary_cross_entropy(pred, labels)
        return loss

    def loss(self, o, u, b, mask):
        # policy density ratio loss
        a_fake = self.agent.planner.sample(b)
        fake_inputs = torch.cat([b, o, a_fake], dim=-1)
        log_r = self.discriminator(fake_inputs).squeeze(-1)
        
        loss_r = torch.sum(log_r * mask, dim=0) / mask.sum(0)
        loss_r = torch.mean(loss_r)
        
        # policy likelihood loss
        logp_pi = self.agent.planner.log_prob(b, u).squeeze(-1)
        loss_pi = torch.sum(logp_pi * mask, dim=0) / mask.sum(0)
        loss_pi = -torch.mean(loss_pi)
        
        # observation loss
        logp_obs = self.agent.hmm.obs_model.mixture_log_prob(b, o)
        loss_obs = torch.sum(logp_obs * mask, dim=0) / mask.sum(0)
        loss_obs = -torch.mean(loss_obs)
        
        # irl loss
        batch = self.buffer.sample_batch(self.plan_batch_size)
        td_error = self.td_loss(batch)
        loss_td = torch.mean(td_error)

        loss = 0.1 * loss_r + loss_pi + self.obs_penalty * loss_obs + self.ld * loss_td

        stats_dict = {
            "loss": loss.data.numpy(),
            "loss_r": loss_r.data.numpy(),
            "loss_pi": loss_pi.data.numpy(),
            "loss_obs": loss_obs.data.numpy(),
            "loss_td": loss_td.data.numpy(),
            "ld": self.ld.data.numpy(),
        }
        return loss, stats_dict

    def print_message(self, stats):
        s = "loss_r: {:.4f}, loss_d: {:.4f}, loss_pi: {:.4f}, loss_obs: {:.4f}, loss_td: {:.4f}, ld: {:.4f}".format(
            stats["loss_r"],
            stats["loss_d"],
            stats["loss_pi"],
            stats["loss_obs"],
            stats["loss_td"],
            stats["ld"],
        )
        return s