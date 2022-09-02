import gym
import numpy as np
import torch
from src.simulation.dynamics import ConstantAcceleration
from src.simulation.reward import CarfollowingReward
from src.simulation.observers import Observer
from src.data.geometry import clip_norm

STATE_KEYS = [
    "x", "y", "vx", "vy", "ax", "ay", "psi_rad", 
    "length", "width", "track_id"
]

class InteractionSimulator:
    """ Temporary simulation used to test lidar observation """
    def __init__(self, map_data, sensors, action_set, svt_object):
        """
        Args:
            map_data (MapReader):
            sensor (List): list of sensors
            action_set (List): list of action names
            svt_object (VehicleTrajectories):
        """
        self.state_keys = STATE_KEYS
        self.svt = svt_object.svt # stack vehicle trajectories per frame
        self.ego_track_ids = svt_object.ego_track_ids # ego track id of each episode
        self.track_ids = svt_object.track_ids # track ids per frame
        self.t_range = svt_object.t_range # start and end time for each track
        self.num_eps = len(self.ego_track_ids) # total number of episodes

        self.dynamics = ConstantAcceleration()
        self.sensors = sensors
        self.sensor_names = [s.__class__.__name__ for s in self.sensors]
        self.observer = Observer(map_data, sensors, action_set)
        self.reward_model = CarfollowingReward(sensors)
        
        self.ax_idx = self.state_keys.index("ax")
        self.ay_idx = self.state_keys.index("ay")
        self.x_lim = map_data.x_lim
        self.y_lim = map_data.y_lim
        self.v_lim = 150. # velocity norm limit
        self.a_lim = np.array([8, 3]).astype(np.float32) # acceleration limits

    def reset(self, eps_id):
        self.observer.reset()
        t_start = self.t_range[eps_id, 0]
        t_end = self.t_range[eps_id, 1]
        
        # episode properties
        self.eps_svt = self.svt[t_start:t_end]
        self.eps_ego_track_id = self.ego_track_ids[eps_id]
        self.eps_track_ids = self.track_ids[t_start:t_end]
        self.t = 0
        self.T = len(self.eps_svt) - 1

        # get sim states
        ego_true_state, agent_states = self.get_sim_state(
            self.eps_svt[self.t], self.eps_track_ids[self.t], self.eps_ego_track_id
        )
        self.ego_state = ego_true_state
        sim_state = {
            "ego_state": self.ego_state,
            "ego_true_state": ego_true_state,
            "agent_states": agent_states
        } 
        sim_act = None

        # get sensor measurements
        sensor_obs, sensor_pos = {}, {}
        for sensor_name, sensor in zip(self.sensor_names, self.sensors):
            s_obs, s_pos = sensor.get_obs(self.ego_state, agent_states) 
            sensor_obs[sensor_name] = s_obs
            sensor_pos[sensor_name] = s_pos
        
        obs = self.observer.observe(sensor_obs)
        rwd = None
        
        self._state = {
            "sim_state": sim_state, 
            "sim_act": sim_act,
            "sensor_obs": sensor_obs, 
            "sensor_pos": sensor_pos,
            "rwd": rwd
        }
        return obs
    
    def step(self, action):
        self.t += 1
        
        # convert action to global
        action = action.clone().flatten()
        action_local = action.clone().data.numpy().flatten()
        psi_old = self.ego_state[6]
        action = self.observer.agent_control_to_global(action, psi_old)
        action = np.clip(action, -self.a_lim, self.a_lim)
        
        # step ego state
        state = self.ego_state[:4].copy().reshape(4, 1)
        action = action.reshape(2, 1)
        next_state = self.dynamics.step(state, action).flatten()

        # clip position and velocity
        next_state[0] = np.clip(next_state[0], self.x_lim[0], self.x_lim[1])
        next_state[1] = np.clip(next_state[1], self.y_lim[0], self.y_lim[1])
        next_state[[2, 3]] = clip_norm(next_state[[2, 3]], self.v_lim)

        psi = self.ego_state[6] # set to old heading
        if next_state[3] != 0 and next_state[2] != 0:
            psi = np.arctan2(next_state[3], next_state[2])
        
        next_ego_state = self.ego_state.copy()
        next_ego_state[:4] = next_state.flatten()
        next_ego_state[[4, 5]] = action.flatten()
        next_ego_state[6] = psi
        
        # get sim states
        ego_true_state, agent_states = self.get_sim_state(
            self.eps_svt[self.t], self.eps_track_ids[self.t], self.eps_ego_track_id
        )
        self.ego_state = next_ego_state 
        sim_state = {
            "ego_state": self.ego_state,
            "ego_true_state": ego_true_state,
            "agent_states": agent_states
        }
        sim_act = {
            "act": action.flatten(),
            "true_act": ego_true_state[[self.ax_idx, self.ay_idx]].flatten(),
            "act_local": action_local.flatten() # action in local coordinate
        }
        
        # get sensor measurements
        sensor_obs, sensor_pos = {}, {}
        for sensor_name, sensor in zip(self.sensor_names, self.sensors):
            s_obs, s_pos = sensor.get_obs(self.ego_state, agent_states)
            sensor_obs[sensor_name] = s_obs
            sensor_pos[sensor_name] = s_pos

        obs = self.observer.observe(sensor_obs)
        rwd = self.reward_model(sensor_obs, action) 
        done = True if (self.t + 1) >= self.T else False
        info = self.observer.get_info()
        
        self._state = {
            "sim_state": sim_state, 
            "sim_act": sim_act,
            "sensor_obs": sensor_obs, 
            "sensor_pos": sensor_pos,
            "reward": rwd
        }
        return obs, rwd, done, info

    def get_sim_state(self, state, track_ids, ego_track_id):
        ego_idx = np.where(track_ids == (ego_track_id))[0][0]
        agent_idx = np.where(track_ids != ego_track_id)[0]
        ego_state = state[ego_idx]
        agent_states = state[agent_idx]
        return ego_state, agent_states

    def get_data_action(self):
        ego_state, _ = self.get_sim_state(
            self.eps_svt[self.t], self.eps_track_ids[self.t], self.eps_ego_track_id
        )
        ax_id = self.state_keys.index("ax")
        ay_id = self.state_keys.index("ay")
        psi_id = self.state_keys.index("psi_rad")
        
        ax = ego_state[ax_id]
        ay = ego_state[ay_id]
        psi = ego_state[psi_id]
        action = self.observer.agent_control_to_local(ax, ay, psi)
        action = torch.from_numpy(action).view(-1, 2).to(torch.float32)
        return action


# class InteractionSimulator(gym.Env):
#     """ Simulator with 
    
#     global states
#         ego: [x, y, vx, vy, psi, l, w] 
#         agents: [x, y, vx, vy, psi, l, w]
    
#     global actions:
#         [ax, ay]
#     """
#     def __init__(self, dataset, map_data, observer, reward=CarfollowingReward, max_eps_steps=1000, dt=0.1):
#         super().__init__()
#         self.max_eps_steps = max_eps_steps
#         self.dt = dt

#         self.dataset = dataset
#         self.map_data = map_data
#         self.observer = observer
#         self.dynamics_model = ConstantAcceleration(self.dt)
#         self.reward_model = CarfollowingReward(observer.feature_set)
        
#         self.x_lim = map_data.x_lim
#         self.y_lim = map_data.y_lim
#         self.v_lim = 150. # velocity limit
#         self.a_lim = 5. # acceleration limit
#         self.action_limits = np.array([8, 3]).astype(np.float32)
#         self.action_space = gym.spaces.Box(
#             low=-self.action_limits,
#             high=self.action_limits,
#             dtype=np.float64
#         )
#         self.obs_dim = len(self.observer.feature_set)
        
#         self._track_data = None # recorded track data {"ego", "agents"}
#         self._sim_states = None # simulated states [x, y, vx, vy, psi, kappa]
#         self._sim_acts = None # simulated actions [ax, ay]
#         self._sim_obs = None # agent observation 
#         self._sim_ctl = None # agent control
#         self.t = None # current time
#         self.T = None # episode max time

#     def __len__(self):
#         return len(self.dataset)
    
#     def reset(self, eps_id):
#         self._track_data = self.dataset[eps_id]
#         self.t = 0
#         self.T = len(self._track_data["ego"]) - 1
#         self._sim_states = np.zeros((self.T + 1, 7))
#         self._sim_acts = np.zeros((self.T + 1, 2))
#         self._sim_obs = np.zeros((self.T + 1, self.obs_dim))
#         self._sim_ctl = np.zeros((self.T, 2))
        
#         self._sim_states[0] = self._track_data["ego"][0][:7]
        
#         state_dict = {
#             "ego": self._sim_states[0],
#             "agents": self._track_data["agents"][0][:, :7]
#         }

#         self.observer.reset()
#         obs = self.observer.observe(state_dict)
#         self._sim_obs[self.t] = obs.view(-1).numpy()
#         return obs
    
#     def step(self, ctl):
#         """ Step the environment
        
#         Args:
#             ctl (torch.tensor): control vector in agent's chosen frame of reference. 
#                 To be transformed by the observer into the global frame. size=[1, 2]

#         Returns:
#             obs (torch.tensor): observation vector. size=[1, obs_dim]
#             reward ():
#             done (bool): whether maximum steps have been reached
#             info (dict): 
#         """
#         # unpack current state
#         state = self._sim_states[self.t][:4]
#         psi = self._sim_states[self.t][4]
#         l = self._sim_states[self.t][5]
#         w = self._sim_states[self.t][6]
        
#         # transfrom agent control
#         ctl = ctl.numpy().flatten()
#         ctl = np.clip(ctl, -self.action_limits, self.action_limits)
#         action = self.observer.agent_control_to_global(ctl[0], ctl[1], psi)
        
#         # step the dynamics
#         state_action = np.hstack([state, action]).reshape(-1, 1)
#         next_state = self.dynamics_model.step(state_action).flatten()[:4]

#         # clip position and velocity
#         next_state[0] = np.clip(next_state[0], self.map_data.x_lim[0], self.map_data.x_lim[1])
#         next_state[1] = np.clip(next_state[1], self.map_data.y_lim[0], self.map_data.y_lim[1])
#         next_state[[2, 3]] = clip_norm(next_state[[2, 3]], self.v_lim)
#         next_psi = self.compute_psi_kappa(next_state, psi)
        
#         self._sim_states[self.t+1] = np.hstack([next_state, next_psi, l, w])
#         self._sim_acts[self.t] = action
        
#         state_dict = {
#             "ego": self._sim_states[self.t+1],
#             "agents": self._track_data["agents"][self.t+1][:, :7]
#         }
#         obs = self.observer.observe(state_dict)
#         reward = self.reward_model(self._sim_obs[self.t], self._sim_ctl[self.t])
#         done = True if any([self.t + 2 >= self.T, self.t >= self.max_eps_steps]) else False
#         info = self.observer.get_info()

#         # udpate self
#         self.t += 1
#         self._sim_obs[self.t] = obs.view(-1).numpy()
#         self._sim_ctl[self.t-1] = ctl
#         return obs, reward, done, info

#     def get_action_from_data(self):
#         """ Get observed action from track data """
#         assert self._track_data is not None, "Please reset episode"
#         act = self._track_data["act"][self.t]
#         return act
    
#     def compute_psi_kappa(self, state, last_psi):
#         """ Compute heading and curvature 
        
#         Args:
#             state (np.array): current state vector [x, y, vx, vy]
#             last_state (np.array): last state vector [x, y, vx, vy]
#             last_psi (float): last heading value
        
#         Returns:
#             psi (float): current heading
#             kappa (float): current curvature
#         """
#         if state[2] != 0 and state[3] != 0:
#             psi = np.arctan2(state[3], state[2])
#         else:
#             psi = last_psi
        
#         # computer curvature
#         # psi_history = np.hstack([self._sim_states[:self.t+1, 4], np.array([psi])])
#         # x_history = np.hstack([self._sim_states[:self.t+1, 0], np.array([state[0]])])
#         # y_history = np.hstack([self._sim_states[:self.t+1, 1], np.array([state[1]])])

#         # d_psi = np.gradient(psi_history)
#         # d_x = np.gradient(x_history)
#         # d_y = np.gradient(y_history)
#         # d_s = np.sqrt(d_x**2 + d_y**2)
#         # kappa = d_psi[-1] / d_s[-1]
#         return psi
    
#     def render(self):
#         return 

#     def close(self):
#         return 