import numpy as np
import torch
from src.data.ego_dataset import collate_fn, sample_sequence

class ReplayBuffer:
    def __init__(self, obs_dim, ctl_dim, state_dim, max_size, momentum=0.1):
        """
        Args:
            obs_dim (int): observation dimension
            ctl_dim (int): action dimension
            state_dim (int): hidden state dimension. To not use state dim enter None. 
                Sampled outputs will be zeros.
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

    def __call__(self, obs, ctl, state, rwd, done):
        """ Append episode history """ 
        self.obs_eps.append(obs)
        self.ctl_eps.append(ctl)
        self.state_eps.append(state)
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
        
        if self.state_dim is None:
            state = np.zeros((len(obs), 1))        

        # add absorbing state flag based on done
        if done[-1] == 1:
            obs_a = np.zeros((1, obs.shape[1]))
            ctl_a = np.zeros((1, ctl.shape[1]))
            state_a = np.zeros((1, state.shape[1]))
            rwd_a = rwd[-1:]
            done_a = np.ones((1, 1))

            obs = np.vstack([obs, obs_a, obs_a])
            ctl = np.vstack([ctl, ctl_a, ctl_a])
            state = np.vstack([state, state_a, state_a])
            rwd = np.vstack([rwd, rwd_a, rwd_a])
            done = np.vstack([done, done_a, done_a])

            absorb = np.zeros((len(obs), 1))
            absorb[-2:] = 1
        else:
            absorb = np.zeros((len(obs), 1))

        self.episodes.append({ 
            "obs": obs[:-1],
            "state": state[:-1],
            "absorb": absorb[:-1],
            "ctl": ctl[:-1],
            "rwd": rwd[:-1],
            "next_obs": obs[1:],
            "next_state": state[1:],
            "next_absorb": absorb[1:],
            "next_ctl": ctl[1:],
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
        state = np.vstack([e["state"] for e in self.episodes])
        absorb = np.vstack([e["absorb"] for e in self.episodes])
        ctl = np.vstack([e["ctl"] for e in self.episodes])
        rwd = np.vstack([e["rwd"] for e in self.episodes])
        next_obs = np.vstack([e["next_obs"] for e in self.episodes])
        next_state = np.vstack([e["next_state"] for e in self.episodes])
        next_absorb = np.vstack([e["next_absorb"] for e in self.episodes])
        next_ctl = np.vstack([e["next_ctl"] for e in self.episodes])
        done = np.vstack([e["done"] for e in self.episodes])
        
        # prioritize new data for sampling
        if prioritize:
            idx = np.random.randint(max(0, self.size - batch_size * 100), self.size, size=batch_size)
        else:
            idx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=obs[idx], 
            state=state[idx],
            absorb=absorb[idx],
            ctl=ctl[idx], 
            rwd=rwd[idx], 
            next_obs=next_obs[idx],
            next_state=next_state[idx],
            next_absorb=next_absorb[idx], 
            next_ctl=next_ctl[idx],
            done=done[idx]
        )
        return {k: torch.from_numpy(v).to(torch.float32) for k, v in batch.items()}
    
    def sample_episodes(self, batch_size, max_len=200, prioritize=False, sample_terminal=True):
        """ sample random episodes """
        if prioritize:
            idx = np.random.randint(max(0, self.num_eps - batch_size * 5), self.num_eps, size=batch_size)
        else:
            idx = np.random.randint(0, self.num_eps, size=batch_size)
        
        batch = []
        for i in idx:
            obs = torch.from_numpy(self.episodes[i]["obs"]).to(torch.float32)
            state = torch.from_numpy(self.episodes[i]["state"]).to(torch.float32)
            absorb = torch.from_numpy(self.episodes[i]["absorb"]).to(torch.float32)
            ctl = torch.from_numpy(self.episodes[i]["ctl"]).to(torch.float32)
            rwd = torch.from_numpy(self.episodes[i]["rwd"]).to(torch.float32)
            next_obs = torch.from_numpy(self.episodes[i]["next_obs"]).to(torch.float32)
            next_state = torch.from_numpy(self.episodes[i]["next_state"]).to(torch.float32)
            next_absorb = torch.from_numpy(self.episodes[i]["next_absorb"]).to(torch.float32)
            next_ctl = torch.from_numpy(self.episodes[i]["next_ctl"]).to(torch.float32)
            done = torch.from_numpy(self.episodes[i]["done"]).to(torch.float32)
            
            if not sample_terminal:
                obs = obs[done.flatten() == 0]
                state = state[done.flatten() == 0]
                absorb = absorb[done.flatten() == 0]
                ctl = ctl[done.flatten() == 0]
                rwd = rwd[done.flatten() == 0]
                next_obs = next_obs[done.flatten() == 0]
                next_state = next_state[done.flatten() == 0]
                next_absorb = next_absorb[done.flatten() == 0]
                next_ctl = next_ctl[done.flatten() == 0]
                done = done[done.flatten() == 0]

            # truncate sequence
            sample_ids = sample_sequence(len(obs), max_len, gamma=0.)
            obs = obs[sample_ids]
            state = state[sample_ids]
            absorb = absorb[sample_ids]
            ctl = ctl[sample_ids]
            rwd = rwd[sample_ids]
            next_obs = next_obs[sample_ids]
            next_state = next_state[sample_ids]
            next_absorb = next_absorb[sample_ids]
            next_ctl = next_ctl[sample_ids]
            done = done[sample_ids]

            batch.append({
                "obs": obs, 
                "state": state,
                "absorb": absorb,
                "ctl": ctl, 
                "rwd": rwd, 
                "next_obs": next_obs,
                "next_state": next_state,
                "next_absorb": next_absorb, 
                "next_ctl": next_ctl,
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
