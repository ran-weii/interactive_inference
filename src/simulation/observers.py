import numpy as np
import pandas as pd
import torch
from src.data.geometry import wrap_angles, coord_transformation
from src.data.geometry import angle_to_vector
from src.map_api.frenet_utils import compute_normal_from_kappa
from src.map_api.frenet_utils import compute_acceleration_vector

""" TODO: 
make lane feature calculation a part of the observer 
make observer a wrapper for simulation
"""

ACTION_SET = {
    "ax_ego", # ego centric x acceleration
    "ay_ego", # ego centric y acceleration
    "dds", # frenet s acceleration
    "ddd" # frenet d acceleration
}

FEATURE_SET = {
    "ego": [
        "s", # frenet s distance
        "d", # frenet lateral distance/centerline distance
        "ds", # frenet s velocty
        "dd", # frenet d velocity
        "lbd", # left bound distance
        "rbd", # right bound distance
        "kappa_r", # current lane curvature
        "psi_error_r", # current heading error
        "psi_error_fp_0", # far point 0 error
        "psi_error_fp_1", # far point 0 error
        "psi_error_fp_2", # far point 0 error
        "psi_error_fp_3", # far point 0 error
    ],
    "relative": [
        "s_rel", 
        "d_rel", 
        "ds_rel", 
        "dd_rel", 
        "psi_rel", # relative heading
        "loom_s", 
        "loom_d"
    ],
}

""" TODO: temporary solution, to be integrated with FEATURE_SET """
AUGMENTATION_PARAMETERS = {
    "s": {"flip_lr": 0}, 
    "d": {"flip_lr": 1}, 
    "ds": {"flip_lr": 0}, 
    "dd": {"flip_lr": 1}, 
    "lbd": {"flip_lr": None}, # to be handeled by the function
    "rbd": {"flip_lr": None}, # to be handeled by the function
    "kappa_r": {"flip_lr": 1}, 
    "psi_error_r": {"flip_lr": 1}, 
    "psi_error_fp_0": {"flip_lr": 1}, 
    "psi_error_fp_1": {"flip_lr": 1}, 
    "psi_error_fp_2": {"flip_lr": 1}, 
    "psi_error_fp_3": {"flip_lr": 1}, 
    "s_rel": {"flip_lr": 0}, 
    "d_rel": {"flip_lr": 1}, 
    "ds_rel": {"flip_lr": 0}, 
    "dd_rel": {"flip_lr": 1}, 
    "psi_rel": {"flip_lr": 1}, 
    "loom_s": {"flip_lr": 0}, 
    "loom_d": {"flip_lr": 0}
}

def calc_looming(s_rel, ds_rel, l, w, l0):
    """ Calculate longitudinal looming 
    
    Args:
        s_rel (float): relative distance
        ds_rel (float): relative velocity
        l (float): lead vehicle length
        w (float): lead vehicle width
        l0 (float): ego vehicle length
    """
    x = s_rel - l/2 - l0/2
    theta = 2 * np.arctan(0.5 * w / x)
    theta_dot = w * ds_rel / (x**2 + w**2/4)
    loom = theta_dot/theta
    return loom

class Observer:
    """ Observer object that computes features in the frenet frame """
    default_feature_set = [
        "ego_d", "ego_ds", "ego_dd", "ego_psi_error_r", "ego_kappa_r",
        "lv_s_rel", "lv_ds_rel", "lv_inv_tau", "lv_d", "lv_dd"
    ]
    default_action_set = ["dds", "ddd"]
    def __init__(self, map_data, sensors, feature_set=None, action_set=default_action_set):
        """
        Args:
            map_data (MapReader): map data object
            sensors (list): list of sensor objects
            feature_set (list): list of feature variable names. Default=ego and lead vehicle sensor feature names
            action_set (list): list of action variable names. Default=["dds", "ddd"]
        """
        self.map_data = map_data
        self.sensors = sensors
        if feature_set is None:
            self.feature_set = sum([s.feature_names for s in sensors], [])
        else:
            self.feature_set = feature_set
        self.action_set = action_set
        
        self.sensor_names = [s.__class__.__name__ for s in self.sensors]
        assert "EgoSensor" in self.sensor_names
        assert "LeadVehicleSensor" in self.sensor_names
        self.ego_sensor_idx = self.sensor_names.index("EgoSensor")
        self.lv_sensor_idx = self.sensor_names.index("LeadVehicleSensor")
        
        sensor_feature_names = np.hstack([s.feature_names for s in sensors]).tolist()
        self.feature_idx = [sensor_feature_names.index(f) for f in self.feature_set]
        
        self.reset()
    
    def reset(self):
        self._ref_path_id = None
        self._ref_path = None
        self._trajectory = None
        self._s_condition_ego = None # [s, ds]
        self._d_condition_ego = None # [d, dd]
        self._lv_track_id = None
        for sensor in self.sensors:
            sensor.reset()
        self._data = []

    def observe(self, obs):
        """ Convert environment observations into a vector """
        obs = np.hstack([o.flatten() for o in obs.values()]).reshape(1, -1)
        obs = obs[:, self.feature_idx]
        obs = torch.from_numpy(obs).to(torch.float32)
        
        # update state
        ego_sensor = self.sensors[self.ego_sensor_idx]
        lv_sensor = self.sensors[self.lv_sensor_idx]
        self._ref_path_id = ego_sensor._ref_path_id
        self._ref_path = ego_sensor._ref_path
        self._s_condition_ego = ego_sensor._s_condition_ego # [s, ds]
        self._d_condition_ego = ego_sensor._d_condition_ego # [d, dd]
        self._lv_track_id = lv_sensor._lv_track_id
        return obs
    
    def agent_control_to_global(self, act, psi):
        """ Convert agent control to global frame """
        ax, ay = act[0], act[1]
        if self.action_set[0] == "ax_ego":
            act_env = self.ego_action_to_global(ax, ay, psi)
        else:
            act_env = self.frenet_action_to_global(ax, ay)
        return act_env
    
    def agent_control_to_local(self, ax, ay, psi):
        if self.action_set[0] == "ax_ego":
            ax_ego, ay_ego = coord_transformation(ax, ay, None, None, theta=psi)
            act = np.array([ax_ego, ay_ego])
        else:
            raise NotImplementedError
        return act

    def ego_action_to_global(self, ax, ay, psi):
        """ Convert actions from ego frame to global frame 
        
        Args:
            ax (float): x acceleration in ego frame
            ay (float): y acceleration in ego frame
            psi (np.array): heading in radians

        Returns:
            act_env (np.array): action in global frame. size=[2]
        """
        ax_env, ay_env = coord_transformation(ax, ay, None, None, theta=psi, inverse=True)
        act_env = np.array([ax_env, ay_env])
        return act_env
    
    def frenet_action_to_global(self, dds, ddd):
        """ Convert actions from frenet frame to global frame 
        
        Args:
            dds (float): s acceleration in frenet frame
            ddd (float): d acceleration in frenet frame

        Returns:
            act_env (np.array): action in global frame. size=[2]
        """
        ref_path = self._ref_path
        s_condition = self._s_condition_ego
        d_condition = self._d_condition_ego
        s_condition = np.hstack([s_condition, np.array([dds])])
        d_condition = np.hstack([d_condition, np.array([ddd])])
        cartesian_state = ref_path.frenet_to_cartesian(
            s_condition, d_condition
        )
        
        [x, y, v, a, theta, kappa] = cartesian_state
        norm = compute_normal_from_kappa(theta, kappa)
        tan_vec = angle_to_vector(theta)
        norm_vec = angle_to_vector(norm)
        act_env = compute_acceleration_vector(
            a, v, kappa, tan_vec, norm_vec
        ).flatten()
        return act_env
    
    def get_info(self, ego_state, obs):
        """ Get simulator info. Return terminated=True if d > max_d (3.8) """
        max_d = 3.8
        info = {
            "terminated": np.abs(self._d_condition_ego[0]) > max_d,
            "s": self._s_condition_ego[0],
            "d": self._d_condition_ego[0]
        }
        return info

    def push(self, sim_state, agent_state):
        self._data.append({"sim_state": sim_state.copy(), "agent_state": agent_state.copy()})


class CarfollowObserver:
    """ Car following observer """
    default_feature_set = ["ego_ds", "lv_s_rel", "lv_ds_rel"]
    default_action_set = ["dds"]
    def __init__(self, map_data, sensors, feature_set=None, action_set=default_action_set, max_s_rel=-1.):
        """
        Args:
            map_data (MapReader): map data object
            sensors (list): list of sensor objects
            feature_set (list): list of feature variable names
            action_set (list): list of action variable names
            max_s_rel (float): maximum relative distance to lead vehicle to set terminate to true in info. Default=-1.
        """
        self.map_data = map_data
        self.sensors = sensors
        if feature_set is None:
            self.feature_set = sum([s.feature_names for s in sensors], [])
        else:
            self.feature_set = feature_set
        self.action_set = action_set
        
        self.sensor_names = [s.__class__.__name__ for s in self.sensors]
        assert "EgoSensor" in self.sensor_names
        assert "LeadVehicleSensor" in self.sensor_names
        self.ego_sensor_idx = self.sensor_names.index("EgoSensor")
        self.lv_sensor_idx = self.sensor_names.index("LeadVehicleSensor")
        
        sensor_feature_names = np.hstack([s.feature_names for s in sensors]).tolist()
        self.feature_idx = [sensor_feature_names.index(f) for f in self.feature_set]
        
        # map limits
        self.x_lim = map_data.x_lim
        self.y_lim = map_data.y_lim

        self.x_idx = self.state_key.index("x")
        self.y_idx = self.state_key.index("y")

        self.max_s_rel = max_s_rel

        self.reset()
    
    def reset(self):
        self._ref_path_id = None
        self._ref_path = None
        self._trajectory = None
        self._s_condition_ego = None # [s, ds]
        self._d_condition_ego = None # [d, dd]
        self._lv_track_id = None
        for sensor in self.sensors:
            sensor.reset()
        self._data = []

    def observe(self, obs):
        """ Convert environment observations into a vector """
        obs = np.hstack([o.flatten() for o in obs.values()]).reshape(1, -1)
        obs = obs[:, self.feature_idx]
        obs = torch.from_numpy(obs).to(torch.float32)        
        
        # update state
        ego_sensor = self.sensors[self.ego_sensor_idx]
        lv_sensor = self.sensors[self.lv_sensor_idx]
        self._ref_path_id = ego_sensor._ref_path_id
        self._ref_path = ego_sensor._ref_path
        self._s_condition_ego = ego_sensor._s_condition_ego # [s, ds]
        self._d_condition_ego = ego_sensor._d_condition_ego # [d, dd]
        self._lv_track_id = lv_sensor._lv_track_id
        return obs
    
    def agent_control_to_global(self, act, psi):
        """ Convert agent control to global frame """  
        # parameters for feedback lateral controller
        k1=-0.01
        k2=-0.2

        ax = act[0]
        ay = k1 * self._d_condition_ego[0] + k2 * self._d_condition_ego[1]
        if self.action_set[0] == "ax_ego":
            act_env = self.ego_action_to_global(ax, ay, psi)
        else:
            act_env = self.frenet_action_to_global(ax, ay)
        return act_env
    
    def agent_control_to_local(self, ax, ay, psi):
        if self.action_set[0] == "ax_ego":
            ax_ego, ay_ego = coord_transformation(ax, ay, None, None, theta=psi)
            act = np.array([ax_ego, ay_ego])
        return act

    def ego_action_to_global(self, ax, ay, psi):
        """ Convert actions from ego frame to global frame 
        
        Args:
            ax (float): x acceleration in ego frame
            ay (float): y acceleration in ego frame
            psi (np.array): heading in radians

        Returns:
            act_env (np.array): action in global frame. size=[2]
        """
        ax_env, ay_env = coord_transformation(ax, ay, None, None, theta=psi, inverse=True)
        act_env = np.array([ax_env, ay_env])
        return act_env
    
    def frenet_action_to_global(self, dds, ddd):
        """ Convert actions from frenet frame to global frame 
        
        Args:
            dds (float): s acceleration in frenet frame
            ddd (float): d acceleration in frenet frame

        Returns:
            act_env (np.array): action in global frame. size=[2]
        """
        ref_path = self._ref_path
        s_condition = self._s_condition_ego
        d_condition = self._d_condition_ego
        s_condition = np.hstack([s_condition, np.array([dds])])
        d_condition = np.hstack([d_condition, np.array([ddd])])
        cartesian_state = ref_path.frenet_to_cartesian(
            s_condition, d_condition
        )
        
        [x, y, v, a, theta, kappa] = cartesian_state
        norm = compute_normal_from_kappa(theta, kappa)
        tan_vec = angle_to_vector(theta)
        norm_vec = angle_to_vector(norm)
        act_env = compute_acceleration_vector(
            a, v, kappa, tan_vec, norm_vec
        ).flatten()
        return act_env
    
    def get_info(self, ego_state, obs):
        """ Get simulator info. Return terminate flag """
        # check if ego is ahead of lv
        lv_s_rel_idx = self.feature_set.index("lv_s_rel")
        lv_s_rel = obs[:, lv_s_rel_idx]
        terminate = lv_s_rel < -self.max_s_rel

        # check ego lane deviation
        max_d = 3.8
        terminate = any([terminate, np.abs(self._d_condition_ego[0]) > max_d])
        
        info = {
            "terminate": terminate,
            "s": self._s_condition_ego[0],
            "d": self._d_condition_ego[0]
        }
        return info

    def push(self, sim_state, agent_state):
        self._data.append({"sim_state": sim_state.copy(), "agent_state": agent_state.copy()})