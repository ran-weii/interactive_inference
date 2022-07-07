import numpy as np
import torch
from src.data.ego_dataset import collate_fn, sample_sequence

class ReplayBuffer:  
    """ Replay buffer with hidden state """  
    def __init__(self, state_dim, act_dim, obs_dim, max_size):
        """
        Args:
            state_dim (int): hidden state dimension
            act_dim (int): action dimension
            obs_dim (int): observation dimension
            max_size (int): maximum buffer size
        """
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        
        self.episodes = []
        self.eps_len = []
        
        self.num_eps = 0
        self.size = 0
        self.max_size = max_size
        
        self.state_eps = [] # store a single episode
        self.act_eps = [] # store a single episode
        self.obs_eps = [] # store a single episode

    def __call__(self, controller):
        """ Append episode history """
        self.state_eps.append(controller._b.data.numpy())
        self.act_eps.append(controller._prev_act.data.numpy())
        self.obs_eps.append(controller._obs.data.numpy())

    def push(self):
        """ Store episode """
        state = np.vstack(self.state_eps)
        act = np.vstack(self.act_eps)
        obs = np.vstack(self.obs_eps)

        self.episodes.append({
            "state": state[:-1],
            "obs": obs[:-1],
            "act": act[:-1],
            "next_state": state[1:],
            "next_obs": obs[1:]
        })
        self.eps_len.append(len(state))
        
        self.num_eps += 1
        self.size += len(self.episodes[-1]["state"])
        if self.size > self.max_size:
            while self.size > self.max_size:
                self.size -= len(self.episodes[0]["state"])
                self.episodes = self.episodes[1:]
                self.eps_len = self.eps_len[1:]
                self.num_eps = len(self.eps_len)
        self.state_eps = []
        self.act_eps = [] 
        self.obs_eps = []

    def sample_random(self, batch_size):
        """ sample random steps """
        state = np.vstack([e["state"] for e in self.episodes])
        obs = np.vstack([e["obs"] for e in self.episodes])
        act = np.vstack([e["act"] for e in self.episodes])
        next_state = np.vstack([e["next_state"] for e in self.episodes])
        next_obs = np.vstack([e["next_obs"] for e in self.episodes])
        # print("sample random", state.shape, obs.shape, act.shape, next_state.shape, next_obs.shape)

        idx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            state=state[idx], obs=obs[idx], act=act[idx],
            next_state=next_state[idx], next_obs=next_obs[idx]
        )
        return {k: torch.from_numpy(v).to(torch.float32) for k, v in batch.items()}

    def sample_episodes(self, batch_size, max_len=200):
        """ sample random episodes """
        idx = np.random.randint(0, self.num_eps, size=batch_size)
        
        batch = []
        for i in idx:
            obs = torch.from_numpy(self.episodes[i]["obs"]).to(torch.float32)
            act = torch.from_numpy(self.episodes[i]["act"]).to(torch.float32)

            # truncate sequence
            sample_ids = sample_sequence(len(obs), max_len, gamma=1.)
            obs = obs[sample_ids]
            act = act[sample_ids]

            batch.append({"ego": obs, "act": act})
        
        out = collate_fn(batch)
        return out
        
        
class RecurrentBuffer:
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
