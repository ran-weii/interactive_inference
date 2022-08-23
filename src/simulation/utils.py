import numpy as np
import pandas as pd
from tqdm import tqdm
from src.simulation.dynamics import ConstantAcceleration
from src.simulation.observers import Observer

def create_svt_from_df(df):
    """ Build stacked vehicle trajectories 
    
    Returns:
        svt (list): stack vehicle states for each frame. Each item in the list 
            consists of all vehicles in the frame. size=[num_frames]
        track_ids (list): vehicle track ids in each frame. size=[num_frames]
        t_range (list): start and end frame index of each vehicle. size=[num_tracks, 2]
    """
    state_keys = [
        "x", "y", "vx", "vy", "ax", "ay", "psi_rad", 
        "length", "width", "track_id"
    ]
    svt = [] # stacked vehicle state for each frame
    track_ids = [] # track id for each frame
    for fid in tqdm(df["frame_id"].unique()):
        df_frame = df.loc[df["frame_id"] == fid]
        state = df_frame[state_keys].values
        tid = df_frame["track_id"].values

        svt.append(state)
        track_ids.append(tid)

    t_start = df.groupby("track_id")["frame_id"].head(1).values # first time step of every track
    t_end = df.groupby("track_id")["frame_id"].tail(1).values # last time step of every track
    t_range = np.stack([t_start, t_end]).T
    return svt, track_ids, t_range

class LidarSim:
    """ Temporary simulation used to test lidar observation """
    def __init__(self, map_data, sensors, svt, track_ids, t_range):
        self.svt = svt # stack vehicle trajectories per frame
        self.track_ids = track_ids # track ids per frame
        self.t_range = t_range # start and end time for each track
        self.num_eps = len(t_range)
        
        self.state_keys = [
            "x", "y", "vx", "vy", "ax", "ay", "psi_rad", 
            "length", "width", "track_id"
        ]

        self.dynamics = ConstantAcceleration()
        self.sensors = sensors
        self.sensor_names = [s.__class__.__name__ for s in self.sensors]
        self.observer = Observer(map_data, sensors)

        self.x_lim = map_data.x_lim
        self.y_lim = map_data.y_lim

    def reset(self, ego_id):
        t_start = self.t_range[ego_id, 0]
        t_end = self.t_range[ego_id, 1]
        
        # episode properties
        self.ego_id = ego_id
        self.t = 0
        self.T = t_end - t_start
        self.eps_svt = self.svt[t_start-1:t_end]
        self.eps_track_id = self.track_ids[t_start-1:t_end]
        self._data = []

        # get sim states
        ego_true_state, agent_states = self.get_sim_state(
            self.eps_svt[self.t], self.eps_track_id[self.t], self.ego_id
        )
        self.ego_state = ego_true_state
        sim_state = {
            "ego_state": self.ego_state,
            "ego_true_state": ego_true_state,
            "agent_states": agent_states
        } 

        # get sensor measurements
        sensor_obs, sensor_pos = {}, {}
        for sensor_name, sensor in zip(self.sensor_names, self.sensors):
            s_obs, s_pos = sensor.get_obs(self.ego_state, agent_states) 
            sensor_obs[sensor_name] = s_obs
            sensor_pos[sensor_name] = s_pos

        self._data.append({
            "sim_state": sim_state, 
            "sensor_obs": sensor_obs, 
            "sensor_pos": sensor_pos
        })
        
        obs = self.observer.observe(sensor_obs)
        return obs
    
    def step(self, action):
        self.t += 1
        
        # convert action to global
        action = action.copy().flatten()
        psi_old = self.ego_state[6]
        action = self.observer.agent_control_to_global(action[0], action[1], psi_old)
        
        # step ego state
        state = self.ego_state[:4].copy().reshape(4, 1)
        action = action.reshape(2, 1)
        next_state = self.dynamics.step(state, action).flatten()

        # clip position and velocity
        next_state[0] = np.clip(next_state[0], self.x_lim[0], self.x_lim[1])
        next_state[1] = np.clip(next_state[1], self.y_lim[0], self.y_lim[1])

        psi = self.ego_state[6] # set to old heading
        if next_state[3] != 0 and next_state[2] != 0:
            psi = np.arctan2(next_state[3], next_state[2])
        
        next_ego_state = self.ego_state.copy()
        next_ego_state[:4] = next_state.flatten()
        next_ego_state[[4, 5]] = action.flatten()
        next_ego_state[6] = psi
        
        # get sim states
        ego_true_state, agent_states = self.get_sim_state(
            self.eps_svt[self.t], self.eps_track_id[self.t], self.ego_id
        )
        self.ego_state = next_ego_state 
        sim_state = {
            "ego_state": self.ego_state,
            "ego_true_state": ego_true_state,
            "agent_states": agent_states
        }
        
        # get sensor measurements
        sensor_obs, sensor_pos = {}, {}
        for sensor_name, sensor in zip(self.sensor_names, self.sensors):
            s_obs, s_pos = sensor.get_obs(self.ego_state, agent_states)
            sensor_obs[sensor_name] = s_obs
            sensor_pos[sensor_name] = s_pos

        self._data.append({
            "sim_state": sim_state, 
            "sensor_obs": sensor_obs, 
            "sensor_pos": sensor_pos
        })

        obs = self.observer.observe(sensor_obs)
        rwd = None
        done = True if (self.t + 1) > self.T else False
        info = {}
        return obs, rwd, done, info

    def get_sim_state(self, state, track_ids, ego_id):
        ego_idx = np.where(track_ids == (ego_id + 1))[0][0]
        agent_idx = np.where(track_ids != ego_id + 1)[0]
        ego_state = state[ego_idx]
        agent_states = state[agent_idx]
        return ego_state, agent_states

    def get_data_action(self):
        ego_state, _ = self.get_sim_state(
            self.eps_svt[self.t], self.eps_track_id[self.t], self.ego_id
        )
        ax_id = self.state_keys.index("ax")
        ay_id = self.state_keys.index("ay")
        action = ego_state[[ax_id, ay_id]]
        return action