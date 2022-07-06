import gym
import numpy as np
from src.simulation.dynamics import ConstantAcceleration

"""
TODO: 
be more specific with property naming so we can easily identify data for animation
"""
class InteractionSimulator(gym.Env):
    """ Simulator with 
    
    observations
        ego: [x, y, vx, vy, psi, kappa] 
        agents: [x, y, vx, vy, psi]
    
    actions:
        [ax, ay]
    """
    def __init__(self, dataset, map_data, dt=0.1):
        super().__init__()
        self.dt = dt

        self.dataset = dataset
        self.map_data = map_data
        self.dynamics_model = ConstantAcceleration(self.dt)
        
        self.action_limits = np.array([10, 5]).astype(np.float32)
        self.action_space = gym.spaces.Box(
            low=-self.action_limits,
            high=self.action_limits,
            dtype=np.float64
        )
        
        self._track_data = None # recorded track data {"ego", "agents"}
        self._sim_states = None # simulated states [x, y, vx, vy, psi, kappa]
        self._sim_acts = None # simulated actions [ax, ay]
        self.t = None # current time
        self.T = None # episode max time

    def __len__(self):
        return len(self.dataset)
    
    def reset(self, eps_id):
        self._track_data = self.dataset[eps_id]
        self.t = 0
        self.T = len(self._track_data["act"])
        self._sim_states = np.zeros((self.T, 6))
        self._sim_acts = np.zeros((self.T, 2))
        
        self._sim_states[0] = self._track_data["ego"][0][:6]
        obs_dict = {
            "ego": self._sim_states[0],
            "agents": self._track_data["agents"][0][:, :5]
        }
        return obs_dict
    
    def get_action(self):
        """ Get observed action from track data """
        assert self._track_data is not None, "Please reset episode"
        act = self._track_data["act"][self.t]
        return act
    
    def compute_psi_kappa(self, state, last_psi):
        """ Compute heading and curvature 
        
        Args:
            state (np.array): current state vector [x, y, vx, vy]
            last_state (np.array): last state vector [x, y, vx, vy]
            last_psi (float): last heading value
        
        Returns:
            psi (float): current heading
            kappa (float): current curvature
        """
        if state[2] != 0 and state[3] != 0:
            psi = np.arctan2(state[3], state[2])
        else:
            psi = last_psi
        
        # computer curvature
        psi_history = np.hstack([self._sim_states[:self.t+1, 4], np.array([psi])])
        x_history = np.hstack([self._sim_states[:self.t+1, 0], np.array([state[0]])])
        y_history = np.hstack([self._sim_states[:self.t+1, 1], np.array([state[1]])])

        d_psi = np.gradient(psi_history)
        d_x = np.gradient(x_history)
        d_y = np.gradient(y_history)
        d_s = np.sqrt(d_x**2 + d_y**2)
        kappa = d_psi[-1] / d_s[-1]
        return psi, kappa

    def step(self, action):
        state = self._sim_states[self.t][:4]
        psi = self._sim_states[self.t][4]
        state_action = np.hstack([state, action]).reshape(-1, 1)
        next_state = self.dynamics_model.step(state_action).flatten()[:4]
        next_psi, next_kappa = self.compute_psi_kappa(next_state, psi)
        
        self._sim_states[self.t+1] = np.hstack([next_state, next_psi, next_kappa])
        self._sim_acts[self.t] = action

        self.t += 1
        obs_dict = {
            "ego": self._sim_states[self.t],
            "agents": self._track_data["agents"][self.t][:, :5]
        }
        reward = None
        done = True if self.t == self.T - 1 else False
        info = {}
        return obs_dict, reward, done, info
    
    def render(self):
        return 

    def close(self):
        return 