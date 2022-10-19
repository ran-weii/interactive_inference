import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

# model imports
from src.distributions.nn_models import Model, MLP
from src.algo.rl import DoubleQNetwork
from src.algo.replay_buffers import ReplayBuffer

class RecurrentDAC(Model):
    """ Discriminator actor critic for recurrent networks """
    def __init__(
        self, agent, hidden_dim, num_hidden, activation, 
        gamma=0.9, beta=0.2, polyak=0.995, norm_obs=False, use_state=False,
        buffer_size=int(1e6), d_batch_size=100, a_batch_size=32, rnn_len=50, 
        d_steps=50, a_steps=50, lr_d=1e-3, lr_a=1e-3, lr_c=1e-3, decay=0, grad_clip=None, 
        grad_penalty=1., grad_target=1., bc_penalty=1., obs_penalty=1., reg_penalty=1.
        ):
        """
        Args:
            agent (Agent): actor agent
            hidden_dim (int): value network hidden dim
            num_hidden (int): value network hidden layers
            activation (str): network activation
            gamma (float, optional): discount factor. Default=0.9
            beta (float, optional): softmax temperature. Default=0.2
            polyak (float, optional): target network polyak averaging factor. Default=0.995
            norm_obs (bool, optional): whether to normalize observations for critic. Default=False
            use_state (bool, optional): whether to use state in discriminator and critic. Default=False
            buffer_size (int, optional): replay buffer size. Default=1e6
            d_batch_size (int, optional): discriminator batch size. Default=100
            a_batch_size (int, optional): actor critic batch size. Default=32
            rnn_len (int, optional): number of recurrent steps to sample. Default=50
            d_steps (int, optional): discriminator update steps per training step. Default=50
            a_steps (int, optional): actor critic update steps per training step. Default=50
            lr_d (float, optional): discriminator learning rate. Default=1e-3
            lr_a (float, optional): actor learning rate. Default=1e-3
            lr_c (float, optional): critic learning rate. Default=1e-3
            decay (float, optional): weight decay. Default=0
            grad_clip (float, optional): gradient clipping. Default=None
            grad_penalty (float, optional): discriminator gradient norm penalty. Default=1.
            grad_target (float, optional): discriminator gradient norm target. Default=1.
            bc_penalty (float, optional): behavior cloning penalty. Default=1.
            obs_penalty (float, optional): observation penalty. Default=1.
            reg_penalty (float, optional): regularization penalty. Default=1.
        """
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.polyak = polyak
        self.norm_obs = norm_obs
        self.use_state = use_state
    
        self.buffer_size = buffer_size
        self.d_batch_size = d_batch_size
        self.a_batch_size = a_batch_size
        self.rnn_len = rnn_len
        self.d_steps = d_steps
        self.a_steps = a_steps
        self.lr_d = lr_d
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.decay = decay
        self.grad_clip = grad_clip
        self.grad_penalty = grad_penalty
        self.grad_target = grad_target
        self.bc_penalty = bc_penalty
        self.obs_penalty = obs_penalty
        self.reg_penalty = reg_penalty

        self.agent = agent
        self.ref_agent = deepcopy(agent)

        # discriminator and critic input dims
        # extra dimension for absorbing state flag
        disc_input_dim = agent.obs_dim + agent.ctl_dim + 1
        critic_input_dim = agent.obs_dim + 1
        if use_state:
            disc_input_dim += agent.state_dim
            critic_input_dim += agent.state_dim

        self.discriminator = MLP(
            input_dim=disc_input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation,
            batch_norm=False
        )
        self.critic = DoubleQNetwork(
            critic_input_dim, agent.ctl_dim, hidden_dim, num_hidden, activation
        )
        self.critic_target = deepcopy(self.critic)

        # freeze target parameters
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        self.d_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(), lr=lr_d, weight_decay=decay
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=lr_c, weight_decay=decay
        )
        self.actor_optimizer = torch.optim.AdamW(
            self.agent.parameters(), lr=lr_a, weight_decay=decay
        )

        self.real_buffer = ReplayBuffer(agent.obs_dim, agent.ctl_dim, agent.state_dim, buffer_size)
        self.replay_buffer = ReplayBuffer(agent.obs_dim, agent.ctl_dim, agent.state_dim, buffer_size)

        self.obs_mean = nn.Parameter(torch.zeros(agent.obs_dim), requires_grad=False)
        self.obs_variance = nn.Parameter(torch.ones(agent.obs_dim), requires_grad=False)
        
        self.plot_keys = [
            "eps_return_avg", "d_loss_avg", "critic_loss_avg", 
            "actor_loss_avg", "bc_loss_avg", "obs_loss_avg"
        ]
    
    def __repr__(self):
        s_agent = self.agent.__repr__()
        s_critic = self.critic.__repr__()
        s_discriminator = self.discriminator.__repr__()
        s = "{}(gamma={}, beta={}, polyak={}, norm_obs={}, "\
            "buffer_size={}, d_batch_size={}, a_steps={}, d_steps={}, "\
            "lr_d={}, lr_c={}, lr_a={}, decay={}, grad_clip={}, grad_penalty={}, grad_target={}, "\
            "bc_penalty={}, obs_penalty={}, reg_penalty={}\n    discriminator={}\n    agent={}, \n    critic={}\n)".format(
            self.__class__.__name__, self.gamma, self.beta, self.polyak, self.norm_obs,
            self.replay_buffer.max_size, self.d_batch_size, self.a_steps, self.d_steps,
            self.lr_d, self.lr_c, self.lr_a, self.decay, self.grad_clip, self.grad_penalty, self.grad_target,
            self.bc_penalty, self.obs_penalty, self.reg_penalty, s_discriminator, s_agent, s_critic
        )
        return s
    
    def fill_real_buffer(self, dataset):
        for i in range(len(dataset)):
            batch = dataset[i]
            obs = batch["ego"].to(self.device)
            ctl = batch["act"].to(self.device)
            rwd = np.zeros((len(obs), 1))
            done = np.zeros((len(obs), 1))

            if self.use_state:
                with torch.no_grad():
                    _, [state, _] = self.agent(obs.unsqueeze(1), ctl.unsqueeze(1))
                    state = state.squeeze(1).cpu()
            else:
                state = torch.zeros(len(obs), self.agent.state_dim)

            self.real_buffer.push(obs.numpy(), ctl.numpy(), state.numpy(), rwd, done)
    
    def on_epoch_end(self):
        # udpate ref agent
        with torch.no_grad():
            for p, p_target in zip(
                self.agent.parameters(), self.ref_agent.parameters()
            ):
                p_target.data = p.data
        self.ref_agent.eval()

        # Update real buffer hidden states on epoch end
        if self.use_state:
            num_samples = min(self.a_batch_size, self.real_buffer.num_eps)
            eps_ids = np.random.choice(np.arange(self.real_buffer.num_eps), num_samples, replace=False)
            for i in eps_ids:
                # buffer size trim handle
                if (i + 1) >= self.real_buffer.num_eps:
                    break
                
                obs = self.real_buffer.episodes[i]["obs"]
                ctl = self.real_buffer.episodes[i]["ctl"]
                next_obs = self.real_buffer.episodes[i]["next_obs"]
                next_ctl = self.real_buffer.episodes[i]["next_ctl"]
                done = self.real_buffer.episodes[i]["done"]
                rwd = self.real_buffer.episodes[i]["rwd"]
                absorb = self.real_buffer.episodes[i]["absorb"]
                
                # remove absorbing states
                obs = obs[absorb.flatten() == 0]
                ctl = ctl[absorb.flatten() == 0]
                next_obs = next_obs[absorb.flatten() == 0]
                next_ctl = next_ctl[absorb.flatten() == 0]
                done = done[absorb.flatten() == 0]
                rwd = rwd[absorb.flatten() == 0]
                absorb = absorb[absorb.flatten() == 0]

                # offset done by one time step
                done = np.vstack([np.zeros((1, 1)), done])[:-1]
                
                # compute state from agent
                obs_torch = torch.from_numpy(obs).to(torch.float32).to(self.device)
                ctl_torch = torch.from_numpy(ctl).to(torch.float32).to(self.device)
                with torch.no_grad():
                    _, [state, _] = self.ref_agent(obs_torch.unsqueeze(1), ctl_torch.unsqueeze(1))
                    state = state.squeeze(1).cpu().numpy()

                self.real_buffer.push(obs, ctl, state, rwd, done)
    
    def update_normalization_stats(self):
        if self.norm_obs:
            mean = torch.from_numpy(self.replay_buffer.moving_mean).to(torch.float32).to(self.device)
            variance = torch.from_numpy(self.replay_buffer.moving_variance).to(torch.float32).to(self.device)

            self.obs_mean.data = mean
            self.obs_variance.data = variance
            
            if hasattr(self.agent, "obs_mean"):
                self.agent.obs_mean.data = mean
                self.agent.obs_variance.data = variance
    
    def reset(self):
        self.ref_agent.eval()
        self.ref_agent.reset()

    def choose_action(self, obs):
        with torch.no_grad():
            ctl, _ = self.ref_agent.choose_action(obs.to(self.device))
        return ctl.squeeze(0).cpu()

    def normalize_obs(self, obs):
        obs_norm = (obs - self.obs_mean) / self.obs_variance**0.5
        return obs_norm
    
    def concat_inputs(self, state, obs, absorb, ctl=None):
        out = torch.cat([obs, absorb], dim=-1)
        if self.use_state:
            out = torch.cat([state, out], dim=-1)
        if ctl is not None:
            out = torch.cat([out, ctl], dim=-1)
        return out
    
    def gradient_penalty(self, real_inputs, fake_inputs):
        # interpolate data
        alpha = torch.rand(len(real_inputs), 1).to(self.device)
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

    def compute_discriminator_loss(self): 
        real_batch = self.real_buffer.sample_random(self.d_batch_size, prioritize=False)
        fake_batch = self.replay_buffer.sample_random(self.d_batch_size, prioritize=True)
        
        real_state = real_batch["state"].to(self.device)
        real_obs = real_batch["obs"].to(self.device)
        real_ctl = real_batch["ctl"].to(self.device)
        real_absorb = real_batch["absorb"].to(self.device)

        fake_state = fake_batch["state"].to(self.device)
        fake_obs = fake_batch["obs"].to(self.device)
        fake_ctl = fake_batch["ctl"].to(self.device)
        fake_absorb = fake_batch["absorb"].to(self.device)
        
        # normalize obs
        real_obs_norm = self.normalize_obs(real_obs)
        fake_obs_norm = self.normalize_obs(fake_obs)
        
        # mask absorbing state observation with zeros
        real_obs_norm *= 1 - real_absorb
        fake_obs_norm *= 1 - fake_absorb

        real_inputs = self.concat_inputs(real_state, real_obs_norm, real_absorb, real_ctl)
        fake_inputs = self.concat_inputs(fake_state, fake_obs_norm, fake_absorb, fake_ctl)
        inputs = torch.cat([real_inputs, fake_inputs], dim=0)

        real_labels = torch.zeros(self.d_batch_size, 1)
        fake_labels = torch.ones(self.d_batch_size, 1)
        labels = torch.cat([real_labels, fake_labels], dim=0).to(self.device)

        out = torch.sigmoid(self.discriminator(inputs))
        d_loss = F.binary_cross_entropy(out, labels)

        gp = self.gradient_penalty(real_inputs, fake_inputs)
        return d_loss, gp
    
    def compute_reward(self, state, obs, absorb, ctl):
        inputs = self.concat_inputs(state, obs, absorb, ctl)
        log_r = self.discriminator(inputs)
        r = -log_r
        return r

    def compute_critic_loss(self):
        real_batch = self.real_buffer.sample_episodes(int(self.a_batch_size/2), self.rnn_len, prioritize=False)
        fake_batch = self.replay_buffer.sample_episodes(int(self.a_batch_size/2), self.rnn_len, prioritize=True)
        real_pad_batch, real_mask = real_batch
        fake_pad_batch, fake_mask = fake_batch
        
        real_state = real_pad_batch["state"].to(self.device)
        real_obs = real_pad_batch["obs"].to(self.device)
        real_absorb = real_pad_batch["absorb"].to(self.device)
        real_ctl = real_pad_batch["ctl"].to(self.device)
        real_next_state = real_pad_batch["next_state"].to(self.device)
        real_next_obs = real_pad_batch["next_obs"].to(self.device)
        real_next_absorb = real_pad_batch["next_absorb"].to(self.device)
        real_next_ctl = real_pad_batch["next_ctl"].to(self.device)
        real_done = real_pad_batch["done"].to(self.device)
        real_mask = real_mask.to(self.device)
        
        fake_state = fake_pad_batch["state"].to(self.device)
        fake_obs = fake_pad_batch["obs"].to(self.device)
        fake_absorb = fake_pad_batch["absorb"].to(self.device)
        fake_ctl = fake_pad_batch["ctl"].to(self.device)
        fake_next_state = fake_pad_batch["next_state"].to(self.device)
        fake_next_obs = fake_pad_batch["next_obs"].to(self.device)
        fake_next_absorb = fake_pad_batch["next_absorb"].to(self.device)
        fake_next_ctl = fake_pad_batch["next_ctl"].to(self.device)
        fake_done = fake_pad_batch["done"].to(self.device)
        fake_mask = fake_mask.to(self.device)
        
        state = torch.cat([real_state, fake_state], dim=1)
        obs = torch.cat([real_obs, fake_obs], dim=1)
        absorb = torch.cat([real_absorb, fake_absorb], dim=1)
        ctl = torch.cat([real_ctl, fake_ctl], dim=1)
        next_state = torch.cat([real_next_state, fake_next_state], dim=1)
        next_obs = torch.cat([real_next_obs, fake_next_obs], dim=1)
        next_absorb = torch.cat([real_next_absorb, fake_next_absorb], dim=1)
        next_ctl = torch.cat([real_next_ctl, fake_next_ctl], dim=1)
        done = torch.cat([real_done, fake_done], dim=1)
        mask = torch.cat([real_mask, fake_mask], dim=1).unsqueeze(-1)
        
        # normalize observation
        obs_norm = self.normalize_obs(obs)
        next_obs_norm = self.normalize_obs(next_obs)

        # mask absorbing state observation with zeros
        obs_norm *= 1 - absorb
        next_obs_norm *= 1 - absorb
        
        # sample next action
        with torch.no_grad():
            next_ctl, logp = self.agent.choose_action_batch(next_obs, next_ctl, sample_z=True)
            next_ctl = next_ctl.squeeze(0)
            logp = logp.squeeze(0).unsqueeze(-1)
        
        # mask absorbing state ctl sample and logp with zeros
        next_ctl *= 1 - next_absorb
        logp *= 1 - next_absorb
        
        critic_inputs = self.concat_inputs(state, obs_norm, absorb)
        critic_next_inputs = self.concat_inputs(next_state, next_obs_norm, next_absorb)
        
        with torch.no_grad():
            # compute reward
            r = self.compute_reward(state, obs_norm, absorb, ctl)

            # compute absorbing reward
            state_a = torch.zeros(1, self.agent.state_dim)
            obs_a = torch.zeros(1, self.agent.obs_dim)
            ctl_a = torch.zeros(1, self.agent.ctl_dim)
            absorb_a = torch.ones(1, 1)
            inputs_a = self.concat_inputs(state_a, obs_a, absorb_a, ctl_a)
            r_a = -self.discriminator(inputs_a)

            # compute value target
            q1_next, q2_next = self.critic_target(critic_next_inputs, next_ctl)
            q_next = torch.min(q1_next, q2_next)
            v_next = q_next - self.beta * logp
            v_absorb = self.gamma / (1 - self.gamma) * r_a
            q_target = r + (1 - next_absorb) * self.gamma * v_next + next_absorb * v_absorb
        
        q1, q2 = self.critic(critic_inputs, ctl)
        q1_loss = torch.pow(q1 - q_target, 2) * mask
        q2_loss = torch.pow(q2 - q_target, 2) * mask
        q1_loss = q1_loss.sum() / mask.sum()
        q2_loss = q2_loss.sum() / mask.sum()
        q_loss = (q1_loss + q2_loss) / 2
        return q_loss

    def compute_actor_loss(self):
        real_batch = self.real_buffer.sample_episodes(self.a_batch_size, self.rnn_len, prioritize=False, sample_terminal=False)
        fake_batch = self.replay_buffer.sample_episodes(self.a_batch_size, self.rnn_len, prioritize=False, sample_terminal=False)
        real_pad_batch, real_mask = real_batch
        fake_pad_batch, fake_mask = fake_batch

        real_obs = real_pad_batch["obs"].to(self.device)
        real_absorb = real_pad_batch["absorb"].to(self.device)
        real_ctl = real_pad_batch["ctl"].to(self.device)
        real_mask = real_mask.to(self.device)

        fake_obs = fake_pad_batch["obs"].to(self.device)
        fake_absorb = fake_pad_batch["absorb"].to(self.device)
        fake_ctl = fake_pad_batch["ctl"].to(self.device)
        fake_mask = fake_mask.to(self.device)
        
        # sample from actor
        real_ctl_sample, _, real_hidden = self.agent.choose_action_batch(
            real_obs, real_ctl, tau=0.1, hard=True, sample_z=True, return_hidden=True
        )
        real_ctl_sample = real_ctl_sample.squeeze(0)
        real_state = real_hidden[0]

        fake_ctl_sample, _, fake_hidden = self.agent.choose_action_batch(
            fake_obs, fake_ctl, tau=0.1, hard=True, sample_z=True, return_hidden=True
        )
        fake_ctl_sample = fake_ctl_sample.squeeze(0)
        fake_state = fake_hidden[0]

        # concat real and fake
        state = torch.cat([real_state, fake_state], dim=1)
        obs = torch.cat([real_obs, fake_obs], dim=1)
        absorb = torch.cat([real_absorb, fake_absorb], dim=1)
        ctl_sample = torch.cat([real_ctl_sample, fake_ctl_sample], dim=1)
        mask = torch.cat([real_mask, fake_mask], dim=1)

        # normalize observation
        obs_norm = self.normalize_obs(obs)
        
        # compute actor loss on both data
        critic_inputs = self.concat_inputs(state, obs_norm, absorb)
        q1, q2 = self.critic(critic_inputs, ctl_sample)
        q = torch.min(q1, q2).squeeze(-1)
        ent = self.agent.ctl_model.entropy().mean()
        
        a_loss = (-self.beta * ent - q) * mask
        a_loss = a_loss.sum() / (mask.sum() + 1e-6)
        
        # compute obs loss on both data
        real_obs_loss, _ = self.agent.obs_loss(real_obs, real_ctl, real_mask, real_hidden)
        fake_obs_loss, _ = self.agent.obs_loss(fake_obs, fake_ctl, fake_mask, fake_hidden)
        obs_loss = (real_obs_loss.mean() + fake_obs_loss.mean()) / 2

        # compute bc loss on real data only
        # _, real_hidden = self.agent(real_obs, real_ctl, sample_z=False)
        bc_loss, _ = self.agent.act_loss(real_obs, real_ctl, real_mask, real_hidden)
        bc_loss = bc_loss.mean()
        return a_loss, bc_loss, obs_loss
    
    def compute_reg_loss(self):
        # obs variance
        obs_vars = self.agent.obs_model.lv.exp()
        obs_loss = torch.sum(obs_vars ** 2)

        # ctl variance
        ctl_vars = self.agent.ctl_model.lv.exp()
        ctl_loss = torch.sum(ctl_vars ** 2)
        
        reg_loss = obs_loss + ctl_loss
        return reg_loss

    def take_gradient_step(self, logger=None):
        self.discriminator.train()
        self.critic.train()
        self.agent.train()
        self.update_normalization_stats()
        
        d_loss_epoch = []
        for i in range(self.d_steps):
            # train discriminator
            d_loss, gp = self.compute_discriminator_loss()
            d_total_loss = d_loss + self.grad_penalty * gp
            d_total_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
            self.d_optimizer.step()
            self.d_optimizer.zero_grad()

            d_loss_epoch.append(d_loss.data.item())
            
            if logger is not None:
                logger.push({"d_loss": d_loss.data.item()})

        critic_loss_epoch = []
        actor_loss_epoch = []
        bc_loss_epoch = []
        obs_loss_epoch = []
        reg_loss_epoch = []
        for i in range(self.a_steps):
            # train critic
            critic_loss = self.compute_critic_loss()
            critic_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()

            critic_loss_epoch.append(critic_loss.data.item())

            # train actor
            actor_loss, bc_loss, obs_loss = self.compute_actor_loss()
            reg_loss = self.compute_reg_loss()
            actor_total_loss = (
                actor_loss + \
                self.bc_penalty * bc_loss + \
                self.obs_penalty * obs_loss + \
                self.reg_penalty * reg_loss
            )
            actor_total_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip)
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss_epoch.append(actor_loss.data.item())
            bc_loss_epoch.append(bc_loss.data.item())
            obs_loss_epoch.append(obs_loss.data.item())
            reg_loss_epoch.append(reg_loss.data.item())
            
            # update target networks
            with torch.no_grad():
                for p, p_target in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    p_target.data.mul_(self.polyak)
                    p_target.data.add_((1 - self.polyak) * p.data)

            if logger is not None:
                logger.push({
                    "critic_loss": critic_loss.cpu().data.item(),
                    "actor_loss": actor_loss.cpu().data.item(),
                    "bc_loss": bc_loss.cpu().data.item(),
                    "obs_loss": obs_loss.cpu().data.item(),
                    "reg_loss": reg_loss.cpu().data.item(),
                })

        stats = {
            "d_loss": np.mean(d_loss_epoch),
            "critic_loss": np.mean(critic_loss_epoch),
            "actor_loss": np.mean(actor_loss_epoch),
            "bc_loss": np.mean(bc_loss_epoch),
            "obs_loss": np.mean(obs_loss_epoch),
            "reg_loss": np.mean(reg_loss_epoch),
        }
        
        self.discriminator.eval()
        self.critic.eval()
        self.agent.eval()
        return stats