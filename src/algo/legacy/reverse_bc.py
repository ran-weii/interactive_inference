import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from src.distributions.nn_models import MLP, Model
from src.algo.replay_buffers import ReplayBuffer

class ReverseBehaviorCloning(Model):
    """ Reverse KL behavior cloning algorithm 
    with truncated backpropagation through time
    """
    def __init__(
        self, agent, hidden_dim, num_hidden, activation, norm_obs=False, use_state=False,
        d_batch_size=200, bptt_steps=30, d_steps=50, grad_target=0., 
        grad_penalty=1., bc_penalty=0., obs_penalty=0, 
        lr_d=1e-3, lr_a=1e-3, decay=0, grad_clip=None
        ):
        super().__init__()
        self.norm_obs = norm_obs
        self.use_state = use_state
        
        self.d_batch_size = d_batch_size
        self.bptt_steps = bptt_steps
        self.d_steps = d_steps
        self.grad_target = grad_target
        self.grad_penalty = grad_penalty
        self.bc_penalty = bc_penalty
        self.obs_penalty = obs_penalty
        self.lr_d = lr_d
        self.lr_a = lr_a
        self.decay = decay
        self.grad_clip = grad_clip
        
        self.agent = agent

        # discriminator imput dim
        disc_input_dim = agent.obs_dim + agent.ctl_dim
        if use_state:
            disc_input_dim += agent.state_dim
        self.discriminator = MLP(
            input_dim=disc_input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation
        )
        
        self.real_buffer = ReplayBuffer(
            agent.obs_dim, agent.ctl_dim, agent.state_dim, int(1e6)
        )
        self.fake_buffer = ReplayBuffer(
            agent.obs_dim, agent.ctl_dim, agent.state_dim, int(1e6)
        )

        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d, weight_decay=decay
        )
        self.a_optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=lr_a, weight_decay=decay
        )

        self.obs_mean = nn.Parameter(torch.zeros(agent.obs_dim), requires_grad=False)
        self.obs_variance = nn.Parameter(torch.ones(agent.obs_dim), requires_grad=False)

        self.loss_keys = ["d_loss", "a_loss", "bc_loss", "obs_loss"]
    
    def __repr__(self):
        s_agent = self.agent.__repr__()
        s_disc = self.discriminator.__repr__()
        s = "{}(bptt_steps={}, grad_penalty={}, bc_penalty={}, obs_penalty={}, "\
            "lr_d={}, lr_a={}, decay={}, grad_clip={},\nagent={}, \ndisciminator={})".format(
            self.__class__.__name__, self.bptt_steps, 
            self.grad_penalty, self.bc_penalty, self.obs_penalty, 
            self.lr_d, self.lr_a, self.decay, self.grad_clip, s_agent, s_disc
        )
        return s
    
    def stdout(self, train_stats, test_stats):
        s = "d_loss: {:.4f}/{:.4f}, a_loss: {:.4f}/{:.4f}, bc_loss: {:.4f}/{:.4f}, obs_loss: {:.4f}/{:.4f}".format(
            train_stats["d_loss"], test_stats["d_loss"], 
            train_stats["a_loss"], test_stats["a_loss"],
            train_stats["bc_loss"], test_stats["bc_loss"],
            train_stats["obs_loss"], test_stats["obs_loss"]
        )
        return s
    
    def fill_buffer(self, dataset):
        for i in range(len(dataset)):
            batch = dataset[i]
            obs = batch["ego"].to(self.device)
            ctl = batch["act"].to(self.device)
            rwd = np.zeros((len(obs), 1))
            done = np.zeros((len(obs), 1))
            with torch.no_grad():
                ctl_sample, _, hidden = self.agent.choose_action_batch(
                    obs.unsqueeze(1), ctl.unsqueeze(1), return_hidden=True
                )
                state = hidden[0].squeeze(1).cpu()
                ctl_sample = ctl_sample[0].squeeze(-2).cpu()
            self.real_buffer.push(obs.numpy(), ctl.numpy(), state.numpy(), rwd, done)
            self.fake_buffer.push(obs.numpy(), ctl_sample.numpy(), state.numpy(), rwd, done)

    def on_epoch_end(self):
        """ Update real buffer hidden states on epoch end """
        num_samples = min(int(self.d_batch_size/4), self.real_buffer.num_eps)
        pad_batch, mask = self.real_buffer.sample_episodes(num_samples)
        obs = pad_batch["obs"].to(self.device)
        ctl = pad_batch["ctl"].to(self.device)

        with torch.no_grad():
            ctl_sample, _, hidden = self.agent.choose_action_batch(obs, ctl, return_hidden=True)
            state = hidden[0].cpu()
            ctl_sample = ctl_sample[0].cpu()
        
        for i in range(num_samples):
            eps_len = int(mask[:, i].sum())
            rwd = np.zeros((eps_len, 1))
            done = np.zeros((eps_len, 1))
            self.real_buffer.push(
                obs[:eps_len, i].cpu().numpy(), 
                ctl[:eps_len, i].cpu().numpy(), 
                state[:eps_len, i].cpu().numpy(), 
                rwd, done
            )
            self.fake_buffer.push(
                obs[:eps_len, i].cpu().numpy(), 
                ctl_sample[:eps_len, i].cpu().numpy(), 
                state[:eps_len, i].cpu().numpy(), 
                rwd, done
            )

    def normalize_obs(self, obs):
        obs_norm = (obs - self.obs_mean) / self.obs_variance**0.5
        return obs_norm
        
    def concat_inputs(self, state, obs, ctl):
        out = torch.cat([obs, ctl], dim=-1)
        if self.use_state:
            out = torch.cat([state, out], dim=-1)
        return out

    def gradient_penalty(self, real_inputs, fake_inputs):
        # interpolate data
        alpha = torch.randn(len(real_inputs), 1).to(self.device)
        interpolated = alpha * real_inputs + (1 - alpha) * fake_inputs
        interpolated = Variable(interpolated, requires_grad=True)
        
        prob = torch.sigmoid(self.discriminator(interpolated))
        
        grad = torch_grad(
            outputs=prob, inputs=interpolated, 
            grad_outputs=torch.ones_like(prob),
            create_graph=True, retain_graph=True
        )[0]
        
        grad_norm = torch.linalg.norm(grad, dim=-1)
        grad_pen = torch.pow(grad_norm - self.grad_target, 2).mean()
        return grad_pen

    def compute_discrimiantor_loss(self):
        real_batch = self.real_buffer.sample_random(self.d_batch_size, prioritize=False)
        fake_batch = self.fake_buffer.sample_random(self.d_batch_size, prioritize=True)
        
        real_state = real_batch["state"].to(self.device)
        real_obs = real_batch["obs"].to(self.device)
        real_ctl = real_batch["ctl"].to(self.device)
        fake_state = fake_batch["state"].to(self.device)
        fake_obs = fake_batch["obs"].to(self.device)
        fake_ctl = fake_batch["ctl"].to(self.device)
        
        # normalize obs
        real_obs_norm = self.normalize_obs(real_obs)
        fake_obs_norm = self.normalize_obs(fake_obs)
        
        real_inputs = self.concat_inputs(real_state, real_obs_norm, real_ctl)
        fake_inputs = self.concat_inputs(fake_state, fake_obs_norm, fake_ctl)
        inputs = torch.cat([real_inputs, fake_inputs], dim=0)
        
        real_labels = torch.zeros(self.d_batch_size, 1)
        fake_labels = torch.ones(self.d_batch_size, 1)
        labels = torch.cat([real_labels, fake_labels], dim=0).to(self.device)
        
        out = torch.sigmoid(self.discriminator(inputs))
        d_loss = F.binary_cross_entropy(out, labels)
        
        gp = self.gradient_penalty(real_inputs, fake_inputs)
        return d_loss, gp
    
    def compute_actor_loss(self, o, u, mask, hidden):
        u_sample, _, hidden = self.agent.choose_action_batch(o, u, hidden, return_hidden=True)
        u_sample = u_sample.squeeze(0)
        state = hidden[0]
        
        # compute density ratio loss
        obs_norm = self.normalize_obs(o)
        disc_inputs = self.concat_inputs(state, obs_norm, u_sample)
        log_r = self.discriminator(disc_inputs).squeeze(-1)

        a_loss = torch.sum(log_r * mask) / (mask.sum() + 1e-6)

        # compute bc loss
        bc_loss, _ = self.agent.act_loss(o, u, mask, hidden)
        bc_loss = bc_loss.mean()

        # compute obs_loss
        obs_loss, _ = self.agent.obs_loss(o, u, mask, hidden)
        obs_loss = obs_loss.mean()
        return a_loss, bc_loss, obs_loss

    def run_epoch(self, loader, train=True):
        if train:
            self.agent.train()
            self.discriminator.train()
        else:
            self.agent.eval()
            self.discriminator.eval()
        
        # train discriminator
        d_loss_epoch = []
        for i in range(self.d_steps):
            d_loss, gp = self.compute_discrimiantor_loss()
            d_total_loss = d_loss + self.grad_penalty * gp
            d_loss_epoch.append(d_loss.data.item())
            if train:
                d_total_loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                self.d_optimizer.step()
                self.d_optimizer.zero_grad()
            else:
                break
        
        # train actor
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
                
                a_loss, bc_loss, obs_loss = self.compute_actor_loss(
                    o_t, u_t, mask_t, hidden
                )
                
                a_total_loss = a_loss + self.bc_penalty * bc_loss + self.obs_penalty * obs_loss
                    
                if train:
                    a_total_loss.backward()
                    if self.grad_clip is not None:
                        nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                    self.a_optimizer.step()
                self.a_optimizer.zero_grad()
                self.d_optimizer.zero_grad()
                
                epoch_stats.append({
                    "train": 1 if train else 0,
                    "a_loss": a_loss.data.item(),
                    "bc_loss": bc_loss.data.item(),
                    "obs_loss": obs_loss.data.item(),
                })
        
        if train:
            self.on_epoch_end()

        stats = pd.DataFrame(epoch_stats).mean().to_dict()
        stats["d_loss"] = np.mean(d_loss_epoch)
        return stats