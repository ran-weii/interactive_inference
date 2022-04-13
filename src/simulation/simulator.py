import gym
import numpy as np
from src.simulation.dynamics import ConstantAcceleration

class InteractionSimulator(gym.Env):
    def __init__(self, dataset, map_data):
        super().__init__()
        self.dt = 0.1

        self.dataset = dataset
        self.map_data = map_data
        self.dynamics_model = ConstantAcceleration(self.dt)
        
        self.action_limits = np.array([10, 5]).astype(np.float32)
        self.action_space = gym.spaces.Box(
            low=-self.action_limits,
            high=self.action_limits,
            dtype=np.float64
        )
        
        self._track = None
        self.t = None
        self.T = None

    def __len__(self):
        return len(self.dataset)
    
    def reset(self, eps_id):
        self._track = self.dataset[eps_id]
        self.t = 0
        self.T = len(self._track["act"])
        self._states = np.zeros((self.T, 5))
        self._acts = np.zeros((self.T, 2))
        
        self._states[0] = self._track["ego"][0][:5]
        obs_dict = {
            "ego": self._states[0],
            "agents": self._track["agents"][0][:, :5]
        }
        return obs_dict
    
    def get_action(self):
        """ Get observed action from track data """
        assert self._track is not None, "Please reset episode"
        act = self._track["act"][self.t]
        return act
        
    def step(self, action):
        state_action = np.hstack([self._states[self.t][:4], action]).reshape(-1, 1)
        next_state = self.dynamics_model.step(state_action).flatten()[:4]
        heading = np.arctan2(next_state[-1], next_state[-2])
        self._states[self.t+1] = np.hstack([next_state, heading])
        self._acts[self.t] = action

        self.t += 1
        obs_dict = {
            "ego": self._states[self.t],
            "agents": self._track["agents"][self.t][:, :5]
        }
        reward = None
        done = True if self.t == self.T - 1 else False
        info = {}
        return obs_dict, reward, done, info

    def render(self):
        return 

    def close(self):
        return 