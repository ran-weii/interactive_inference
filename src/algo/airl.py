import time
import numpy as np
import pandas as pd
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

# rollout imports
from src.evaluation.online import eval_episode
from src.simulation.controllers import AgentWrapper

# model imports
from src.distributions.nn_models import Model, MLP
from src.algo.rl import DoubleQNetwork
from src.algo.replay_buffers import ReplayBuffer

""" TODO: modify this to work with recurrent agent """
class AIRL(Model):
    """ Fully observable Adversarial IRL algorithm """
    def __init__(
        self, 
        agent, 
        obs_mean, 
        obs_var, 
        act_mean, 
        act_var, 
        hidden_dim=32,
        gru_layers=2,
        mlp_layers=2,
        gamma=0.9,
        beta=0.2,
        buffer_size=1e5, 
        rnn_steps=10, 
        batch_size=128, 
        max_eps_len=200,
        d_steps=10,
        ac_steps=10,
        lr_agent=1e-3, 
        lr_d=1e-3, 
        lr_q=1e-3,
        decay=0, 
        grad_clip=None, 
        polyak=0.995
        ):
        """
        Args:
            agent (Agent): agent object
            gamma (float): discount factor. Default=0.9
            buffer_size (int): replay buffer max size. Default=1e5
            rnn_steps (int): number of recurrent steps to store in buffer. Default=10
            batch_size (int): training batch size. Default=128
            lr_agent (float): agent learning rate. Default=1e-3
            lr_algo (float): value and discriminator learning rate. Default=1e-3
            decay (float): weight decay. Default==0
            grad_clip (float, None): gradient clipping norm. Default=None
        """
        super().__init__()
        self.obs_mean = obs_mean
        self.obs_var = obs_var
        self.act_mean = act_mean
        self.act_var = act_var
        
        self.gamma = gamma # discount factor
        self.beta = beta # softmax temperature
        self.rnn_steps = rnn_steps
        self.batch_size = batch_size
        self.max_eps_len = max_eps_len
        self.d_steps = d_steps
        self.ac_steps = ac_steps
        self.lr_agent = lr_agent
        self.lr_d = lr_d
        self.lr_q=lr_q
        self.decay = decay
        self.grad_clip = grad_clip
        self.polyak = polyak
        
        self.agent = agent
        self.ref_agent = deepcopy(agent)
        self.real_buffer = ReplayBuffer(
            agent.obs_dim, agent.ctl_dim, 1e10
        )
        self.fake_buffer = ReplayBuffer(
            agent.obs_dim, agent.ctl_dim, buffer_size
        )

        input_dim = self.agent.obs_dim + self.agent.ctl_dim
        self.discriminator = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_hidden=mlp_layers,
            activation="silu",
            batch_norm=True
        )
        self.critic = DoubleQNetwork(
            agent.obs_dim, agent.ctl_dim, hidden_dim, mlp_layers, activation="silu"
        )
        self.critic_target = deepcopy(self.critic)
        
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d, weight_decay=decay
        )
        self.q_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_q, weight_decay=decay
        )
        self.agent_optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=lr_agent, weight_decay=decay
        )
        self.loss_keys = ["loss_d", "loss_q", "loss_a"]
    
    def __repr__(self):
        s_agent = self.agent.__repr__()
        s = "{}(buffer_size={}, rnn_steps={}, batch_size={}, "\
            "lr_agent={}, lr_d={}, lr_q={}, decay={}, grad_clip={}, "\
            "d_steps={}, ac_steps={},\nagent={})".format(
            self.__class__.__name__, self.fake_buffer.max_size, self.rnn_steps, 
            self.batch_size, self.lr_agent, self.lr_d, self.lr_q, self.decay, self.grad_clip,
            self.d_steps, self.ac_steps, s_agent
        )
        return s
    
    def stdout(self, stats):
        s = "loss_d: {:.4f}, loss_q: {:.4f}, loss_a: {:.4f}".format(
            stats["loss_d"], stats["loss_q"], stats["loss_a"]
        )
        return s
    
    def normalize(self, x, mean, var):
        """ Normalize a batch of sequences """
        std = torch.sqrt(var)
        x_ = (x - mean.view(1, -1)) / std.view(1, -1)
        return x_

    def discriminator_loss(self):
        real_batch = self.real_buffer.sample_random(self.batch_size)
        fake_batch = self.fake_buffer.sample_random(self.batch_size)
        
        real_obs = real_batch["obs"]
        real_ctl = real_batch["ctl"]
        fake_obs = fake_batch["obs"]
        fake_ctl = fake_batch["ctl"]
        
        real_inputs = torch.cat([real_obs, real_ctl], dim=-1)
        fake_inputs = torch.cat([fake_obs, fake_ctl], dim=-1)
        inputs = torch.cat([real_inputs, fake_inputs], dim=0)

        real_labels = torch.zeros(self.batch_size, 1)
        fake_labels = torch.ones(self.batch_size, 1)
        labels = torch.cat([real_labels, fake_labels], dim=0)

        out = torch.sigmoid(self.discriminator(inputs))
        loss = F.binary_cross_entropy(out, labels)
        return loss
    
    def compute_reward(self, obs, ctl):
        inputs = torch.cat([obs, ctl], dim=-1)
        log_r = self.discriminator(inputs)
        logp_ref = self.ref_agent.ctl_log_prob(obs, ctl).unsqueeze(-1)
        r = -log_r + logp_ref
        return r

    def critic_loss(self):
        fake_batch = self.fake_buffer.sample_random(self.batch_size)
        obs = fake_batch["obs"]
        ctl = fake_batch["ctl"]
        next_obs = fake_batch["next_obs"]
        done = fake_batch["done"]
        
        # sample next action
        with torch.no_grad():
            next_ctl = self.agent.choose_action(next_obs, ctl).squeeze(0)
            logp = self.agent.ctl_log_prob(next_obs, next_ctl).unsqueeze(-1)
        
        with torch.no_grad():
            # compute reward
            r = self.compute_reward(obs, ctl)
            
            # compute value target
            q1_next, q2_next = self.critic(next_obs, next_ctl)
            q_next = torch.min(q1_next, q2_next)
            q_target = r + (1 - done) * self.gamma * (q_next - self.beta * logp)

        q1, q2 = self.critic(obs, ctl)
        q1_loss = torch.pow(q1 - q_target, 2).mean()
        q2_loss = torch.pow(q2 - q_target, 2).mean()
        # q1_loss = nn.HuberLoss()(q1, q_target)
        # q2_loss = nn.HuberLoss()(q2, q_target)
        q_loss = (q1_loss + q2_loss) / 2
        return q_loss
    
    def actor_loss(self):
        fake_batch = self.fake_buffer.sample_random(self.batch_size)
        obs = fake_batch["obs"]
        ctl = fake_batch["ctl"]

        ctl_sample = self.agent.choose_action(obs, ctl).squeeze(0)
        logp_u = self.agent.ctl_log_prob(obs, ctl_sample).unsqueeze(-1)
        
        q1, q2 = self.critic(obs, ctl_sample)
        q = torch.min(q1, q2)
        a_loss = torch.mean(self.beta * logp_u - q)
        return a_loss

    def take_gradient_step(self):
        # train discriminator 
        loss_d = []
        for i in range(self.d_steps):
            d_loss = self.discriminator_loss()
            d_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
            self.d_optimizer.step()
            self.d_optimizer.zero_grad()
            loss_d.append(d_loss.data.item())
            # print("d_loss", d_loss.data.item())
        # exit()
        
        # train critic
        loss_q = []
        loss_a = []
        for i in range(self.ac_steps):
            # train critic
            q_loss = self.critic_loss()
            q_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.q_optimizer.step()
            self.agent_optimizer.step()
            self.q_optimizer.zero_grad()
            self.agent_optimizer.zero_grad()

            # train actor
            a_loss = self.actor_loss()
            a_loss.backward()
            
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip)
            self.agent_optimizer.step()
            self.agent_optimizer.zero_grad()
            self.q_optimizer.zero_grad()

            loss_q.append(q_loss.data.item())
            loss_a.append(a_loss.data.item())
        #     print("q_loss", q_loss.data.item(), "a_loss", a_loss.data.item())
        # exit()

        # update target networks
        with torch.no_grad():
            for p, p_target in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                p_target.data.mul(self.polyak)
                p_target.data.add_((1 - self.polyak) * p.data)
        
        # update reference agent
        # with torch.no_grad():
        #     for p, p_ref in zip(
        #         self.agent.parameters(), self.ref_agent.parameters()
        #     ):
        #         p_ref.data.mul(self.polyak)
        #         p_ref.data.add_((1 - self.polyak) * p.data)
        self.ref_agent = deepcopy(self.agent)

        stats = {
            "loss_d": np.mean(loss_d),
            "loss_q": np.mean(loss_q),
            "loss_a": np.mean(loss_a),
        }
        return stats


def train(env, observer, model, real_dataset, action_set, num_eps, epochs, max_steps, verbose=1):
    """ Training loop for adversarial inverse reinforcment learning 
    
    Args:
        env (Env): rl environment
        observer (Observer): observer object
        model (class): trainer model class
        real_dataset (torch.dataset): real dataset object
        action_set (list): agent action set
        num_episodes (int): number of episodes per epoch
        epochs (int): number of training epochs
        max_steps (int): maximum number of steps per episode
        verbose (int, optional): verbose interval. Default=1
    """
    # fill real buffer
    for i in range(len(real_dataset)):
        data = real_dataset[i]
        obs = data["ego"].numpy()
        ctl = data["act"].numpy()
        done = np.zeros((len(obs), 1))
        model.real_buffer.push(obs, ctl, done)
    
    start = time.time()
    history = []
    for e in range(epochs):
        eps_ids = np.random.choice(np.arange(len(env.dataset)), num_eps)
        for i, eps_id in enumerate(eps_ids):
            fake_controller = AgentWrapper(observer, model.ref_agent, action_set, "ace")

            # rollout fake episode
            sim_states, sim_acts, track_data = eval_episode(
                env, fake_controller, eps_id, max_steps, model.fake_buffer
            )
            model.fake_buffer.push()
        
        stats = model.take_gradient_step()
        tnow = time.time() - start
        stats.update({"epoch": e, "time": tnow, "train": 1})
        history.append(stats)

        if (e + 1) % verbose == 0:
            s = model.stdout(stats)
            print("e: {}/{}, {}, t: {:.2f}".format(e + 1, epochs, s, tnow))

    df_history = pd.DataFrame(history)
    return model, df_history

# def train(env, observer, model, real_dataset, action_set, num_eps, epochs, max_steps, verbose=1):
#     """ Training loop for adversarial inverse reinforcment learning 
    
#     Args:
#         env (Env): rl environment
#         observer (Observer): observer object
#         model (class): trainer model class
#         real_dataset (torch.dataset): real dataset object
#         action_set (list): agent action set
#         num_episodes (int): number of episodes per epoch
#         epochs (int): number of training epochs
#         max_steps (int): maximum number of steps per episode
#         verbose (int, optional): verbose interval. Default=1
#     """
#     # fill real buffer
#     for i in range(len(real_dataset)):
#         data = real_dataset[i]
#         obs = data["ego"].numpy()
#         ctl = data["act"].numpy()
#         model.real_buffer.push(obs, ctl)
    
#     start = time.time()
#     history = []
#     for e in range(epochs):
#         eps_ids = np.random.choice(np.arange(len(env.dataset)), num_eps)
#         for i, eps_id in enumerate(eps_ids):
#             fake_controller = AgentWrapper(observer, model.ref_agent, action_set, "ace")

#             # rollout fake episode
#             sim_states, sim_acts, track_data = eval_episode(
#                 env, fake_controller, eps_id, max_steps, model.fake_buffer
#             )
#             model.fake_buffer.push()
        
#         stats = model.take_gradient_step()
#         tnow = time.time() - start
#         stats.update({"epoch": e, "time": tnow, "train": 1})
#         history.append(stats)

#         if (e + 1) % verbose == 0:
#             s = model.stdout(stats)
#             print("e: {}/{}, {}, t: {:.2f}".format(e + 1, epochs, s, tnow))

#     df_history = pd.DataFrame(history)
#     return model, df_history