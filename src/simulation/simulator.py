import gym
import numpy as np
import torch
from src.simulation.dynamics import ConstantAcceleration
from src.simulation.reward import CarfollowingReward
from src.data.geometry import clip_norm

STATE_KEYS = [
    "x", "y", "vx", "vy", "ax", "ay", "psi_rad", 
    "length", "width", "track_id"
]

class InteractionSimulator:
    """ Single agent simulator for the INTERACTION dataset 
    
    The simulator propagates ego agent dynamics using the constant acceleration model. 
    Other agents are playback of the dataset

    Attributes:
        _state (dict): simulator state data with the following fields:
            sim_state: dict with fields [ego_state, ego_true_state, agent_states]
            sim_act: dict with fields [act, true_act, act_local]. act_local is action in agent's local coordinate. 
                It is the flattened numpy version of the action argument to the step method. 
            sensor_obs: dict with sensor names mapping to sensor observations in origional dimension
            sensor_pos: dict with sensor names mapping to sensor target positions
            rwd: scalar reward from a custom reward model
    """
    def __init__(self, map_data, sensors, observer, svt_object):
        """
        Args:
            map_data (MapReader):
            sensor (List): list of sensors
            observer (Observer): observer object
            svt_object (VehicleTrajectories): vehicle trajectories object
        """
        self.state_keys = STATE_KEYS
        self.svt = svt_object.svt # stack vehicle trajectories per frame
        self.ego_track_ids = svt_object.ego_track_ids # ego track id of each episode
        self.track_ids = svt_object.track_ids # track ids per frame
        self.t_range = svt_object.t_range # start and end time for each track
        self.num_eps = len(self.ego_track_ids) # total number of episodes

        self.dynamics = ConstantAcceleration()
        self.sensors = sensors
        self.observer = observer
        self.reward_model = CarfollowingReward(sensors)
        self.sensor_names = [s.__class__.__name__ for s in self.sensors]
        
        self.ax_idx = self.state_keys.index("ax")
        self.ay_idx = self.state_keys.index("ay")
        self.x_lim = map_data.x_lim
        self.y_lim = map_data.y_lim
        self.v_lim = 150. # velocity norm limit
        self.a_lim = np.array([8, 3]).astype(np.float32) # acceleration limits

    def reset(self, eps_id, playback=False):
        self.playback = playback
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
        """ Propagates simulator forward
        
        Args:
            action (torch.tensor): agent action. size[1, act_dim]

        Returns:
            obs (torch.tensor): observation vector processed by the observer. size=[1, obs_dim]
            rwd (float): scalar reward computed by the reward model
            done (bool): whether the end of a data trajectory has been reached. 
            info (dict): environment termination information. 
        """
        self.t += 1
        
        # convert action to global
        action = action.clone().flatten()
        action_local = action.clone().data.numpy().flatten()
        psi_old = self.ego_state[6]
        action = self.observer.agent_control_to_global(action, psi_old)
        action = np.clip(action, -self.a_lim, self.a_lim)

        if self.playback:
            ax_idx = self.state_keys.index("ax")
            ay_idx = self.state_keys.index("ay")
            action = self._state["sim_state"]["ego_true_state"][[ax_idx, ay_idx]].copy()
        
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
        rwd = self.reward_model(sim_state, sensor_obs, action)
        done = True if (self.t + 1) >= self.T else False
        info = self.observer.get_info(self.ego_state, obs)
        
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