import torch
import torch.nn as nn
import torch.distributions as torch_dist
from src.agents.core import AbstractAgent
from src.distributions.utils import rectify

class RelativeAcceleration:
    """ Constant relative acceleration dynamics model with state: [s_rel, ds_rel, inv_tau] """
    def __init__(self, dt=0.1):
        self.state_dim = 3
        self.act_dim = 1

        self.A = torch.tensor([
            [1, -dt],
            [0, 1]
        ]).to(torch.float32)

        self.B = torch.tensor([
            [0],
            [-dt]
        ])

    def __call__(self, state, action):
        next_state = self.A.matmul(state[:, :2].T).T + self.B.matmul(action.T).T
        inv_tau = next_state[:, 1] / next_state[:, 0]
        next_state = torch.cat([next_state, inv_tau.view(-1, 1)], dim=-1)
        return next_state


class VINReward(nn.Module):
    """ VIN observation marginal reward """
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.target_dist = agent.compute_target_dist()

    def forward(self, state):
        logp = self.agent.obs_model.mixture_log_prob(self.target_dist, state)
        return logp


class CEM(AbstractAgent):
    """ Cross entropy method model predictive control """
    def __init__(
        self, model, reward, num_samples=50, topk=5, horizon=6, num_iters=20
        ):
        """
        Args:
            model (class): dynamics model
            reward (nn.module): reward model
            num_samples (int): number of cem samples
            topk (int): top candidates to keep
            horizon (int): planning horizon
            num_iters (int): cem iterations
        """
        super().__init__()        
        self.model = model
        self.reward = reward
        self.act_dim = model.act_dim
        self.num_samples = num_samples
        self.topk = topk
        self.horizon = horizon
        self.num_iters = num_iters
    
    def __repr__(self):
        s = "{}(v0={:.2f}, tau={:.2f}, s0={:.2f}, a={:.2f}, b={:.2f}, lv={:.2f})".format(
            self.__class__.__name__, 
            self.v0.exp().item(),
            self.tau.exp().item(),
            self.s0.exp().item(),
            self.a.exp().item(),
            self.b.exp().item(),
            self.lv.exp().item(),
        )
        return s

    def reset(self):
        self._b = None
        self._state = {
            "b": None, # dummy belief distribution
            "pi": None, # previous policy/action prior
        }
    
    def forward(self, state, verbose=False):
        """ Optimize action distribution parameters 
        
        Args:
            state (torch.tensor): state variable. size=[batch_size, state_dim]

        Returns:
            action_mean (torch.tensor): action mean. size=[batch_size, act_dim]
            action_std (torch.tensor): action std. size=[batch_size, act_dim]
        """
        action_mean, action_std = 0, 1
        history = {"actions":[], "states":[], "rewards":[]}
        for i in range(self.num_iters):
            actions = action_mean + action_std * torch.randn(
                self.horizon, self.num_samples, self.act_dim
            )
            states, rewards = self.rollout(state, actions)
            action_mean, action_std = self.fit_gaussian(actions, rewards)
            if verbose:
                print("{0} rewards mean {1:.2f} std {2:.2f}".format(
                    i+1, rewards.mean(), rewards.std()
                ))
            history["actions"].append(actions)
            history["states"].append(states)
            history["rewards"].append(rewards)
        return action_mean[0][0], action_std[0][0]

    def rollout(self, state, actions):
        """ Rollout environment dynamics model with input actions
        
        Args:
            state (np.array): [state_dim]
            actions (np.array): [T, num_samples, act_dim]

        Returns:
            states (np.array): [T, num_samples, state_dim]
            rewards (np.array): [num_samples]
        """
        states = [torch.empty(0)] * (self.horizon + 1)
        rewards = [torch.empty(0)] * self.horizon
        states[0] = state.view(1, -1).repeat_interleave(self.num_samples, 0)
        for t in range(self.horizon):
            states[t+1] = self.model(states[t], actions[t])
            rewards[t] = self.reward(states[t+1])
            
        states = torch.stack(states)[1:]
        rewards = torch.stack(rewards).sum(0)
        return states, rewards
    
    def fit_gaussian(self, actions, rewards):
        rewards = torch.nan_to_num(rewards, nan=0, posinf=1e8, neginf=-1e8)
        topk = torch.argsort(rewards, descending=True)[:self.topk]
        top_actions = actions[:, topk]
        action_mean, action_std = (
            top_actions.mean(1, keepdims=True),
            top_actions.std(1, keepdims=True))
        return action_mean, action_std

    def choose_action(self, o, sample_method="ace", num_samples=1):
        """ Choose action online for a single time step
        
        Args:
            o (torch.tensor): observation sequence. size[batch_size, obs_dim]
            sample_method (str, optional): sampling method. 
                choices=["bma", "ace", "ace"]. Default="ace"
            num_samples (int, optional): number of samples to draw. Default=1
        
        Returns:
            u_sample (torch.tensor): sampled controls. size=[num_samples, batch_size, ctl_dim]
        """
        mu, lv = self.forward(o)
        if sample_method == "ace":
            a = torch_dist.Normal(mu, rectify(lv)).sample((num_samples,))
        else:
            a = mu.unsqueeze(0).repeat_interleave(num_samples, dim=0)
        logp = torch_dist.Normal(mu, rectify(lv)).log_prob(a).sum(-1)

        self._state["b"] = None
        self._state["pi"] = mu.clone()
        return a, logp