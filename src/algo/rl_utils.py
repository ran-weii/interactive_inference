import numpy as np
import torch

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

        self.state = np.array([]).reshape(0, state_dim)
        self.act = np.array([]).reshape(0, act_dim)
        self.obs = np.array([]).reshape(0, obs_dim)

        self.size = 0
        self.max_size = max_size
        
        self.state_eps = []
        self.act_eps = [] # store a single episode
        self.obs_eps = [] # store a single episode

    def __call__(self, controller):
        """ Append episode history """
        self.state_eps.append(controller._b.data.numpy())
        self.act_eps.append(controller._prev_act.data.numpy())
        self.obs_eps.append(controller._obs.data.numpy())

    def push(self, obs_eps=None, act_eps=None):
        """ Store episode """
        state = np.vstack(self.state_eps)
        act = np.vstack(self.act_eps)
        obs = np.vstack(self.obs_eps)

        self.state = np.concatenate([state, self.state], axis=0)
        self.act = np.concatenate([act, self.act], axis=0)
        self.obs = np.concatenate([obs, self.obs], axis=0)

        if len(self.obs) > self.max_size:
            self.state = self.state[:self.max_size]
            self.act = self.act[:self.max_size]
            self.obs = self.obs[:self.max_size]
        self.size = len(self.obs)


        
        
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
