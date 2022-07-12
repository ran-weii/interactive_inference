import gym
import numpy as np
from src.simulation.dynamics import ConstantAcceleration
from src.simulation.reward import CarfollowingReward
from src.data.geometry import clip_norm

"""
TODO: 
be more specific with property naming so we can easily identify data for animation
vectorize environment
"""
class InteractionSimulator(gym.Env):
    """ Simulator with 
    
    global states
        ego: [x, y, vx, vy, psi, l, w] 
        agents: [x, y, vx, vy, psi, l, w]
    
    global actions:
        [ax, ay]
    """
    def __init__(self, dataset, map_data, observer, reward=CarfollowingReward, max_eps_steps=1000, dt=0.1):
        super().__init__()
        self.max_eps_steps = max_eps_steps
        self.dt = dt

        self.dataset = dataset
        self.map_data = map_data
        self.observer = observer
        self.dynamics_model = ConstantAcceleration(self.dt)
        self.reward_model = CarfollowingReward(observer.feature_set)
        
        self.x_lim = map_data.x_lim
        self.y_lim = map_data.y_lim
        self.v_lim = 150. # velocity limit
        self.a_lim = 5. # acceleration limit
        self.action_limits = np.array([8, 3]).astype(np.float32)
        self.action_space = gym.spaces.Box(
            low=-self.action_limits,
            high=self.action_limits,
            dtype=np.float64
        )
        self.obs_dim = len(self.observer.feature_set)
        
        self._track_data = None # recorded track data {"ego", "agents"}
        self._sim_states = None # simulated states [x, y, vx, vy, psi, kappa]
        self._sim_acts = None # simulated actions [ax, ay]
        self._sim_obs = None # agent observation 
        self._sim_ctl = None # agent control
        self.t = None # current time
        self.T = None # episode max time

    def __len__(self):
        return len(self.dataset)
    
    def reset(self, eps_id):
        self._track_data = self.dataset[eps_id]
        self.t = 0
        self.T = len(self._track_data["ego"]) - 1
        self._sim_states = np.zeros((self.T + 1, 7))
        self._sim_acts = np.zeros((self.T + 1, 2))
        self._sim_obs = np.zeros((self.T + 1, self.obs_dim))
        self._sim_ctl = np.zeros((self.T, 2))
        
        self._sim_states[0] = self._track_data["ego"][0][:7]
        
        state_dict = {
            "ego": self._sim_states[0],
            "agents": self._track_data["agents"][0][:, :7]
        }

        self.observer.reset()
        obs = self.observer.observe(state_dict)
        self._sim_obs[self.t] = obs.view(-1).numpy()
        return obs
    
    def step(self, ctl):
        """ Step the environment
        
        Args:
            ctl (torch.tensor): control vector in agent's chosen frame of reference. 
                To be transformed by the observer into the global frame. size=[1, 2]

        Returns:
            obs (torch.tensor): observation vector. size=[1, obs_dim]
            reward ():
            done (bool): whether maximum steps have been reached
            info (dict): 
        """
        # unpack current state
        state = self._sim_states[self.t][:4]
        psi = self._sim_states[self.t][4]
        l = self._sim_states[self.t][5]
        w = self._sim_states[self.t][6]
        
        # transfrom agent control
        ctl = ctl.numpy().flatten()
        ctl = np.clip(ctl, -self.action_limits, self.action_limits)
        action = self.observer.agent_control_to_global(ctl[0], ctl[1], psi)
        
        # step the dynamics
        state_action = np.hstack([state, action]).reshape(-1, 1)
        next_state = self.dynamics_model.step(state_action).flatten()[:4]

        # clip position and velocity
        next_state[0] = np.clip(next_state[0], self.map_data.x_lim[0], self.map_data.x_lim[1])
        next_state[1] = np.clip(next_state[1], self.map_data.y_lim[0], self.map_data.y_lim[1])
        next_state[[2, 3]] = clip_norm(next_state[[2, 3]], self.v_lim)
        next_psi = self.compute_psi_kappa(next_state, psi)
        
        self._sim_states[self.t+1] = np.hstack([next_state, next_psi, l, w])
        self._sim_acts[self.t] = action
        
        state_dict = {
            "ego": self._sim_states[self.t+1],
            "agents": self._track_data["agents"][self.t+1][:, :7]
        }
        obs = self.observer.observe(state_dict)
        reward = self.reward_model(self._sim_obs[self.t], self._sim_ctl[self.t])
        done = True if any([self.t + 2 >= self.T, self.t >= self.max_eps_steps]) else False
        info = self.observer.get_info()

        # udpate self
        self.t += 1
        self._sim_obs[self.t] = obs.view(-1).numpy()
        self._sim_ctl[self.t-1] = ctl
        return obs, reward, done, info

    def get_action_from_data(self):
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
        # psi_history = np.hstack([self._sim_states[:self.t+1, 4], np.array([psi])])
        # x_history = np.hstack([self._sim_states[:self.t+1, 0], np.array([state[0]])])
        # y_history = np.hstack([self._sim_states[:self.t+1, 1], np.array([state[1]])])

        # d_psi = np.gradient(psi_history)
        # d_x = np.gradient(x_history)
        # d_y = np.gradient(y_history)
        # d_s = np.sqrt(d_x**2 + d_y**2)
        # kappa = d_psi[-1] / d_s[-1]
        return psi
    
    def render(self):
        return 

    def close(self):
        return 