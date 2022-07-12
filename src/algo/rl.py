from copy import deepcopy
import torch
import torch.nn as nn
from src.distributions.nn_models import Model
from src.distributions.nn_models import MLP
from src.algo.replay_buffers import ReplayBuffer

class DoubleQNetwork(Model):
    """ Double Q network for fully observable use """
    def __init__(self, obs_dim, ctl_dim, hidden_dim, num_hidden, activation="silu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.ctl_dim = ctl_dim

        self.q1 = MLP(
            input_dim=obs_dim + ctl_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation,
            batch_norm=False
        )
        self.q2 = MLP(
            input_dim=obs_dim + ctl_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation,
            batch_norm=False
        )
    
    def forward(self, o, u):
        """ Compute q1 and q2 values
        
        Args:
            o (torch.tensor): observation. size=[batch_size, obs_dim]
            u (torch.tensor): action. size=[batch_size, ctl_dim]

        Returns:
            q1 (torch.tensor): q1 value. size=[batch_size, 1]
            q2 (torch.tensor): q2 value. size=[batch_size, 1]
        """
        x = torch.cat([o, u], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2


class SAC(nn.Module):
    """ Soft actor critic """
    def __init__(
        self, agent, hidden_dim, num_hidden, gamma, beta, 
        buffer_size, batch_size, lr, decay, polyak, grad_clip
        ):
        super().__init__()
        self.gamma = gamma
        self.beta = beta

        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.polyak = polyak

        self.agent = agent

        self.critic = DoubleQNetwork(
            agent.obs_dim, agent.ctl_dim, hidden_dim, num_hidden, "relu"
        )
        self.critic_target = deepcopy(self.critic)

        # freeze target parameters
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr, weight_decay=decay
        )
        self.actor_optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=lr, weight_decay=decay
        )
        self.replay_buffer = ReplayBuffer(agent.obs_dim, agent.ctl_dim, buffer_size)
    
    def normalize_obs(self, obs):
        mu = torch.from_numpy(self.replay_buffer.moving_mean).to(torch.float32)
        std = torch.from_numpy(self.replay_buffer.moving_variance**0.5).to(torch.float32)
        obs_norm = (obs - mu) / std
        return obs_norm
    
    def choose_action(self, obs):
        obs = self.normalize_obs(obs)
        prev_ctl = self.agent._prev_ctl

        with torch.no_grad():
            ctl = self.agent.choose_action(obs, prev_ctl).squeeze(0)
        return ctl

    def compute_critic_loss(self):
        fake_batch = self.replay_buffer.sample_random(self.batch_size)
        obs = fake_batch["obs"]
        ctl = fake_batch["ctl"]
        r = fake_batch["rwd"]
        next_obs = fake_batch["next_obs"]
        done = fake_batch["done"]      
        
        # normalize observation
        obs = self.normalize_obs(obs)
        next_obs = self.normalize_obs(next_obs)

        # sample next action
        with torch.no_grad():
            next_ctl = self.agent.choose_action(next_obs, ctl).squeeze(0)
            logp = self.agent.ctl_log_prob(next_obs, next_ctl).unsqueeze(-1)
        
        with torch.no_grad():    
            # compute value target
            q1_next, q2_next = self.critic_target(next_obs, next_ctl)
            q_next = torch.min(q1_next, q2_next)
            q_target = r + (1 - done) * self.gamma * (q_next - self.beta * logp)

        q1, q2 = self.critic(obs, ctl)
        q1_loss = torch.pow(q1 - q_target, 2).mean()
        q2_loss = torch.pow(q2 - q_target, 2).mean()
        q_loss = (q1_loss + q2_loss) / 2
        return q_loss

    def compute_actor_loss(self):
        fake_batch = self.replay_buffer.sample_random(self.batch_size)
        obs = fake_batch["obs"]
        ctl = fake_batch["ctl"]

        # normalize observation
        obs = self.normalize_obs(obs)

        ctl_sample = self.agent.choose_action(obs, ctl).squeeze(0)
        logp_u = self.agent.ctl_log_prob(obs, ctl_sample).unsqueeze(-1)
        
        q1, q2 = self.critic(obs, ctl_sample)
        q = torch.min(q1, q2)
        a_loss = torch.mean(self.beta * logp_u - q)
        return a_loss

    def take_gradient_step(self):
        self.critic.train()
        self.agent.train()

        # train critic
        critic_loss = self.compute_critic_loss()
        critic_loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        # train actor
        actor_loss = self.compute_actor_loss()
        actor_loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip)
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        # update target networks
        with torch.no_grad():
            for p, p_target in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                p_target.data.mul_(self.polyak)
                p_target.data.add_((1 - self.polyak) * p.data)
 
        stats = {
            "critic_loss": critic_loss.data.item(),
            "actor_loss": actor_loss.data.item(),
        }
        return stats