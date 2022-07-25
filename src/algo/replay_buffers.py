import numpy as np
import torch
from src.data.ego_dataset import collate_fn, sample_sequence

class ReplayBuffer:
    def __init__(self, obs_dim, ctl_dim, state_dim, max_size, momentum=0.1):
        """
        Args:
            obs_dim (int): observation dimension
            ctl_dim (int): action dimension
            state_dim (int): hidden state dimension
            max_size (int): maximum buffer size
            momentum (float): moving stats momentum
        """
        self.obs_dim = obs_dim
        self.ctl_dim = ctl_dim
        self.state_dim = state_dim

        self.episodes = []
        self.eps_len = []

        self.num_eps = 0
        self.size = 0
        self.max_size = max_size
        
        self.obs_eps = [] # store a single episode
        self.ctl_eps = [] # store a single episode
        self.state_eps = [] # store a single episode
        self.rwd_eps = [] # store a single episode
        self.done_eps = [] # store a single episode
        
        self.momentum = momentum
        self.moving_mean = np.zeros((obs_dim,))
        self.moving_mean_square = np.zeros((obs_dim,))
        self.moving_variance = np.ones((obs_dim, ))

    def __call__(self, obs, ctl, state, rwd, done=False):
        """ Append episode history """ 
        self.obs_eps.append(obs.data.numpy())
        self.ctl_eps.append(ctl.data.numpy())
        self.state_eps.append(state.data.numpy())
        self.rwd_eps.append(np.array(rwd).reshape(1, 1))
        self.done_eps.append(np.array([int(done)]).reshape(1, 1))
    
    def clear(self):
        self.episodes = []
        self.eps_len = []
        self.num_eps = 0
        self.size = 0
        
    def push(self, obs=None, ctl=None, state=None, rwd=None, done=None):
        """ Store episode """
        if obs is None and ctl is None:
            obs = np.vstack(self.obs_eps)
            ctl = np.vstack(self.ctl_eps)
            state = np.vstack(self.state_eps)
            rwd = np.vstack(self.rwd_eps)
            done = np.vstack(self.done_eps)
        
        self.episodes.append({ 
            "obs": obs[:-1],
            "ctl": ctl[:-1],
            "state": state[:-1],
            "rwd": rwd[:-1],
            "next_obs": obs[1:],
            "next_ctl": ctl[1:],
            "next_state": state[1:],
            "done": done[1:]
        })
        self.update_obs_stats(obs)
        
        self.eps_len.append(len(self.episodes[-1]["obs"]))
        
        self.num_eps += 1
        self.size += len(self.episodes[-1]["obs"])
        
        if self.size > self.max_size:
            while self.size > self.max_size:
                self.size -= len(self.episodes[0]["obs"])
                self.episodes = self.episodes[1:]
                self.eps_len = self.eps_len[1:]
                self.num_eps = len(self.eps_len)
        self.obs_eps = []
        self.ctl_eps = [] 
        self.state_eps = []
        self.rwd_eps = []
        self.done_eps = []

    def sample_random(self, batch_size, prioritize=False):
        """ sample random steps """ 
        obs = np.vstack([e["obs"] for e in self.episodes])
        ctl = np.vstack([e["ctl"] for e in self.episodes])
        state = np.vstack([e["state"] for e in self.episodes])
        rwd = np.vstack([e["rwd"] for e in self.episodes])
        next_obs = np.vstack([e["next_obs"] for e in self.episodes])
        next_ctl = np.vstack([e["next_ctl"] for e in self.episodes])
        next_state = np.vstack([e["next_state"] for e in self.episodes])
        done = np.vstack([e["done"] for e in self.episodes])
        
        # prioritize new data for sampling
        if prioritize:
            idx = np.random.randint(max(0, self.size - batch_size * 100), self.size, size=batch_size)
        else:
            idx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=obs[idx], 
            ctl=ctl[idx], 
            state=state[idx],
            rwd=rwd[idx], 
            next_obs=next_obs[idx], 
            next_ctl=next_ctl[idx],
            next_state=next_state[idx],
            done=done[idx]
        )
        return {k: torch.from_numpy(v).to(torch.float32) for k, v in batch.items()}
    
    def sample_episodes(self, batch_size, max_len=200, prioritize=False):
        """ sample random episodes """
        if prioritize:
            idx = np.random.randint(max(0, self.num_eps - batch_size * 5), self.num_eps, size=batch_size)
        else:
            idx = np.random.randint(0, self.num_eps, size=batch_size)
        
        batch = []
        for i in idx:
            obs = torch.from_numpy(self.episodes[i]["obs"]).to(torch.float32)
            ctl = torch.from_numpy(self.episodes[i]["ctl"]).to(torch.float32)
            state = torch.from_numpy(self.episodes[i]["state"]).to(torch.float32)
            rwd = torch.from_numpy(self.episodes[i]["rwd"]).to(torch.float32)
            next_obs = torch.from_numpy(self.episodes[i]["next_obs"]).to(torch.float32)
            next_ctl = torch.from_numpy(self.episodes[i]["next_ctl"]).to(torch.float32)
            next_state = torch.from_numpy(self.episodes[i]["next_state"]).to(torch.float32)
            done = torch.from_numpy(self.episodes[i]["done"]).to(torch.float32)

            # truncate sequence
            sample_ids = sample_sequence(len(obs), max_len, gamma=1.)
            obs = obs[sample_ids]
            ctl = ctl[sample_ids]
            state = state[sample_ids]
            rwd = rwd[sample_ids]
            next_obs = next_obs[sample_ids]
            next_ctl = next_ctl[sample_ids]
            next_state = next_state[sample_ids]
            done = done[sample_ids]

            batch.append({
                "obs": obs, 
                "ctl": ctl, 
                "state": state,
                "rwd": rwd, 
                "next_obs": next_obs, 
                "next_ctl": next_ctl,
                "next_state": next_state,
                "done": done
            })
        
        out = collate_fn(batch)
        return out
        
    def update_obs_stats(self, obs):
        batch_size = len(obs)
        
        moving_mean = (self.moving_mean * self.size + np.sum(obs, axis=0)) / (self.size + batch_size)
        moving_mean_square = (self.moving_mean_square * self.size + np.sum(obs**2, axis=0)) / (self.size + batch_size)
        moving_variance = moving_mean_square - moving_mean**2

        self.moving_mean = self.moving_mean * (1 - self.momentum) + moving_mean * self.momentum
        self.moving_mean_square = self.moving_mean_square * (1 - self.momentum) + moving_mean_square * self.momentum
        self.moving_variance = self.moving_variance * (1 - self.momentum) + moving_variance * self.momentum 


class RecurrentBuffer:  
    """ Replay buffer with hidden state """  
    def __init__(self, state_dim, ctl_dim, obs_dim, max_size):
        """
        Args:
            state_dim (int): hidden state dimension
            ctl_dim (int): action dimension
            obs_dim (int): observation dimension
            max_size (int): maximum buffer size
        """
        self.state_dim = state_dim
        self.ctl_dim = ctl_dim
        self.obs_dim = obs_dim
        
        self.episodes = []
        self.eps_len = []
        
        self.num_eps = 0
        self.size = 0
        self.max_size = max_size
        
        self.state_eps = [] # store a single episode
        self.ctl_eps = [] # store a single episode
        self.obs_eps = [] # store a single episode

    def __call__(self, controller):
        """ Append episode history """
        self.state_eps.append(controller._b.data.numpy())
        self.ctl_eps.append(controller._prev_act.data.numpy())
        self.obs_eps.append(controller._obs.data.numpy())

    def push(self):
        """ Store episode """
        state = np.vstack(self.state_eps)
        ctl = np.vstack(self.ctl_eps)
        obs = np.vstack(self.obs_eps)

        self.episodes.append({
            "state": state[:-1],
            "obs": obs[:-1],
            "ctl": ctl[:-1],
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
        self.ctl_eps = [] 
        self.obs_eps = []

    def sample_random(self, batch_size):
        """ sample random steps """
        state = np.vstack([e["state"] for e in self.episodes])
        obs = np.vstack([e["obs"] for e in self.episodes])
        ctl = np.vstack([e["ctl"] for e in self.episodes])
        next_state = np.vstack([e["next_state"] for e in self.episodes])
        next_obs = np.vstack([e["next_obs"] for e in self.episodes])
        # print("sample random", state.shape, obs.shape, ctl.shape, next_state.shape, next_obs.shape)

        idx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            state=state[idx], obs=obs[idx], ctl=ctl[idx],
            next_state=next_state[idx], next_obs=next_obs[idx]
        )
        return {k: torch.from_numpy(v).to(torch.float32) for k, v in batch.items()}

    def sample_episodes(self, batch_size, max_len=200):
        """ sample random episodes """
        idx = np.random.randint(0, self.num_eps, size=batch_size)
        
        batch = []
        for i in idx:
            obs = torch.from_numpy(self.episodes[i]["obs"]).to(torch.float32)
            ctl = torch.from_numpy(self.episodes[i]["ctl"]).to(torch.float32)

            # truncate sequence
            sample_ids = sample_sequence(len(obs), max_len, gamma=1.)
            obs = obs[sample_ids]
            ctl = ctl[sample_ids]

            batch.append({"ego": obs, "ctl": ctl})
        
        out = collate_fn(batch)
        return out
        