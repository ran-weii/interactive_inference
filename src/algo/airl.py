import time
import numpy as np
import pandas as pd
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.distributions.nn_models import Model, GRUMLP
from src.evaluation.online import eval_episode
from src.simulation.controllers import AgentWrapper

class AIRL(Model):
    """ Adversarial IRL algorithm """
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
        buffer_size=1e5, 
        rnn_steps=10, 
        batch_size=128, 
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
        
        self.gamma = gamma
        self.rnn_steps = rnn_steps
        self.batch_size = batch_size
        self.d_steps = d_steps
        self.ac_steps = ac_steps
        self.lr_agent = lr_agent
        self.lr_d = lr_d
        self.lr_q=lr_q
        self.decay = decay
        self.grad_clip = grad_clip
        self.polyak = polyak
        
        self.agent = agent
        self.real_buffer = ReplayBuffer(
            agent.obs_dim, agent.ctl_dim, buffer_size, rnn_steps
        )
        self.fake_buffer = ReplayBuffer(
            agent.obs_dim, agent.ctl_dim, buffer_size, rnn_steps
        )
        self.discriminator = GRUMLP(
            input_dim=agent.obs_dim + agent.ctl_dim, 
            output_dim=1, 
            hidden_dim=hidden_dim, 
            gru_layers=gru_layers, 
            mlp_layers=mlp_layers, 
            activation="silu"
        )
        self.q1 = GRUMLP(
            input_dim=agent.obs_dim + agent.ctl_dim, 
            output_dim=1, 
            hidden_dim=hidden_dim, 
            gru_layers=gru_layers, 
            mlp_layers=mlp_layers, 
            activation="silu"
        )
        self.q2 = GRUMLP(
            input_dim=agent.obs_dim + agent.ctl_dim, 
            output_dim=1, 
            hidden_dim=hidden_dim, 
            gru_layers=gru_layers, 
            mlp_layers=mlp_layers, 
            activation="silu"
        )
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d, weight_decay=decay
        )
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), 
            lr=lr_q, weight_decay=decay
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
            self.__class__.__name__, self.fake_buffer.max_size, self.fake_buffer.rnn_steps, 
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
        x_ = (x - mean.view(1, 1, -1)) / std.view(1, 1, -1)
        return x_

    def discriminator_loss(self, real_batch, fake_batch):
        real_obs = self.normalize(real_batch["obs"], self.obs_mean, self.obs_var)
        real_act = self.normalize(real_batch["act"], self.act_mean, self.act_var)

        fake_obs = self.normalize(fake_batch["obs"], self.obs_mean, self.obs_var)
        fake_act = self.normalize(fake_batch["act"], self.act_mean, self.act_var)
        
        real_inputs = torch.cat([real_obs, real_act], dim=-1)
        fake_inputs = torch.cat([fake_obs, fake_act], dim=-1)
        inputs = torch.cat([real_inputs, fake_inputs], dim=1)

        real_labels = torch.zeros(self.batch_size, 1)
        fake_labels = torch.ones(self.batch_size, 1)
        labels = torch.cat([real_labels, fake_labels], dim=0)

        out = torch.sigmoid(self.discriminator(inputs))
        loss = F.binary_cross_entropy(out, labels)
        return loss
    
    def critic_loss(self, fake_batch):
        obs = self.normalize(fake_batch["obs"], self.obs_mean, self.obs_var)
        act = self.normalize(fake_batch["act"], self.act_mean, self.act_var)
        next_obs = self.normalize(fake_batch["next_obs"], self.obs_mean, self.obs_var)
        next_act = torch.cat([act[1:], torch.zeros(1, self.batch_size, self.agent.ctl_dim)], dim=0)
        inputs = torch.cat([obs, act], dim=-1)
        
        with torch.no_grad():
            # sample next action
            self.agent.reset()
            out = self.agent.choose_action_batch(next_obs, next_act).squeeze(0)
            next_act = torch.cat([act[1:], out[-1].unsqueeze(0)], dim=0)
            next_inputs = torch.cat([next_obs, next_act], dim=-1)

            # compute reward
            log_r = self.discriminator(inputs)
            
            # compute value target
            q1_next = self.q1_target(next_inputs)
            q2_next = self.q2_target(next_inputs)
            q_next = torch.max(q1_next, q2_next)
            q_target = log_r + self.gamma * q_next

        q1 = self.q1(inputs)
        q2 = self.q2(inputs)
        q1_loss = torch.pow(q1 - q_target, 2).mean()
        q2_loss = torch.pow(q2 - q_target, 2).mean()
        q_loss = (q1_loss + q2_loss) / 2
        return q_loss
    
    def actor_loss(self, fake_batch):
        obs = self.normalize(fake_batch["obs"], self.obs_mean, self.obs_var)
        act = self.normalize(fake_batch["act"], self.act_mean, self.act_var)
        
        # sample actions
        act_sample = self.agent.hmm.ctl_model.sample((self.batch_size,)).squeeze(1)
        
        # compute value for each strata
        q = [torch.empty(0)] * self.agent.act_dim
        for i in range(self.agent.act_dim):
            act_i = torch.cat([act[:-1], act_sample[:, i].unsqueeze(0)], dim=0)
            inputs_i = torch.cat([obs, act_i], dim=-1)
            q1 = self.q1(inputs_i)
            q2 = self.q2(inputs_i)
            q[i] = torch.max(q1, q2)
        q = torch.stack(q).transpose(0, 1).squeeze(-1)

        # compute stratified loss
        self.agent.reset()
        alpha_b, alpha_a = self.agent(obs, act)
        a_loss = torch.sum(alpha_a[-1] * q, dim=-1).mean()
        return a_loss

    def take_gradient_step(self):
        # train discriminator 
        for i in range(self.d_steps):
            real_batch = self.real_buffer.sample_batch(self.batch_size)
            fake_batch = self.fake_buffer.sample_batch(self.batch_size)
            
            d_loss = self.discriminator_loss(real_batch, fake_batch)
            d_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
            self.d_optimizer.step()
            self.d_optimizer.zero_grad()
            # print("d_loss", d_loss.data.item())
        
        # train critic
        for i in range(self.ac_steps):
            fake_batch = self.fake_buffer.sample_batch(self.batch_size)

            # train critic
            q_loss = self.critic_loss(fake_batch)
            q_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(
                    list(self.q1.parameters()) + list(self.q2.parameters()), self.grad_clip
                )
            self.q_optimizer.step()
            self.q_optimizer.zero_grad()

            # train actor
            a_loss = self.actor_loss(fake_batch)
            a_loss.backward()
            # for n, p in self.agent.named_parameters():
            #     if p.grad is not None:
            #         print(n, p.grad.data.norm())
            #     else:
            #         print(n, None)
            # exit()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip)
            self.agent_optimizer.step()
            self.agent_optimizer.zero_grad()
            self.q_optimizer.zero_grad()
            # print("q_loss", q_loss.data.item(), "a_loss", a_loss.data.item())

        # update target networks
        with torch.no_grad():
            for p, p_target in zip(
                list(self.q1.parameters()) + list(self.q2.parameters()),
                list(self.q1_target.parameters()) + list(self.q2_target.parameters()),
            ):
                p_target.data.mul(self.polyak)
                p_target.data.add_((1 - self.polyak) * p.data)
        stats = {
            "loss_d": d_loss.data.item(),
            "loss_q": q_loss.data.item(),
            "loss_a": a_loss.data.item(),
        }
        return stats


class ReplayBuffer:
    """ Replay buffer for recurrent policy """
    def __init__(self, obs_dim, act_dim, max_size, rnn_steps):
        """
        Args:
            obs_dim (int): observation dimension
            act_dim (int): action dimension
            max_size (int): maximum buffer size
            rnn_steps (int): number of recurrent steps to store
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.size = 0
        self.max_size = max_size
        self.rnn_steps = rnn_steps
        
        self.obs = np.array([]).reshape(0, rnn_steps, obs_dim)
        self.act = np.array([]).reshape(0, rnn_steps, act_dim)
        self.next_obs = np.array([]).reshape(0, rnn_steps, obs_dim)
        
        self.obs_eps = [] # store a single episode
        self.act_eps = [] # store a single episode
    
    def __call__(self, controller):
        """ Append episode history """
        self.obs_eps.append(controller._obs.data.numpy())
        self.act_eps.append(controller._prev_act.data.numpy())
    
    def push(self, obs_eps=None, act_eps=None):
        """ Pack episode into chunk 
        
        Args:
            obs_eps (np.array): array of episode observations. size=[T, obs_dim]
            act_eps (np.array): array of episode actions. size=[T, act_dim]
        """
        if obs_eps is None and act_eps is None:
            obs_eps = self.obs_eps
            act_eps = self.act_eps

        T = len(obs_eps) - self.rnn_steps - 1
        for t in range(T):
            obs = np.vstack(obs_eps[t:t+self.rnn_steps]).reshape(1, self.rnn_steps, self.obs_dim)
            act = np.vstack(act_eps[t:t+self.rnn_steps]).reshape(1, self.rnn_steps, self.act_dim)
            next_obs = np.vstack(obs_eps[t+1:t+self.rnn_steps+1]).reshape(1, self.rnn_steps, self.obs_dim)
            
            self.obs = np.concatenate([obs, self.obs], axis=0)
            self.act = np.concatenate([act, self.act], axis=0)
            self.next_obs = np.concatenate([next_obs, self.next_obs], axis=0)

        if len(self.obs) > self.max_size:
            self.obs = self.obs[:self.max_size]
            self.act = self.act[:self.max_size]
            self.next_obs = self.next_obs[:self.max_size]
        self.size = len(self.obs)
        self.obs_eps = []
        self.act_eps = []
    
    def sample_batch(self, batch_size=32):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs[idx], act=self.act[idx], next_obs=self.next_obs[idx])
        return {k: torch.from_numpy(v).transpose(0, 1).to(torch.float32) for k, v in batch.items()}


""" TODO: finish replay buffer, define discriminator on state or observation? """
def train(env, observer, model, dataset, action_set, num_eps, epochs, max_steps, verbose=1):
    """ Training loop for adversarial inverse reinforcment learning 
    
    Args:
        env (Env): rl environment
        observer (Observer): observer object
        model (class): trainer model class
        dataset (torch.dataset): real dataset object
        action_set (list): agent action set
        num_episodes (int): number of episodes per epoch
        epochs (int): number of training epochs
        max_steps (int): maximum number of steps per episode
        verbose (int, optional): verbose interval. Default=1
    """

    start = time.time()
    history = []
    for e in range(epochs):
        controller = AgentWrapper(observer, model.agent, action_set, "ace")
        
        eps_ids = np.random.choice(np.arange(len(env.dataset)), num_eps)
        for i, eps_id in enumerate(eps_ids):
            # rollout fake episode
            sim_states, sim_acts, track_data = eval_episode(
                env, controller, eps_id, max_steps, model.fake_buffer
            )
            model.fake_buffer.push()
            
            # get real episode data
            real_obs = dataset[eps_id]["ego"].data.numpy()
            real_act = dataset[eps_id]["act"].data.numpy()
            model.real_buffer.push(real_obs, real_act)
        
        stats = model.take_gradient_step()
        tnow = time.time() - start
        stats.update({"epoch": e, "time": tnow})
        history.append(stats)

        if (e + 1) % verbose == 0:
            s = model.stdout(stats)
            print("e: {}/{}, {}, t: {:.2f}".format(e + 1, epochs, s, tnow))

    df_history = pd.DataFrame(history)
    return model, df_history