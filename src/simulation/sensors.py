import numpy as np
from src.simulation.observers import calc_looming
from src.simulation.simulator import STATE_KEYS
from src.data.geometry import compute_bounding_box
from src.data.geometry import coord_transformation
from src.data.geometry import wrap_angles

class EgoSensor:
    def __init__(self, map_data, state_keys=STATE_KEYS):
        self.map_data = map_data

        self.x_idx = state_keys.index("x")
        self.y_idx = state_keys.index("y")
        self.vx_idx = state_keys.index("vx")
        self.vy_idx = state_keys.index("vy")
        self.psi_idx = state_keys.index("psi_rad")
        self.l_idx = state_keys.index("length")
        self.w_idx = state_keys.index("width")
        self.id_idx = state_keys.index("track_id")
        
        self.feature_names = [
            "ego_d", "ego_ds", "ego_dd", "ego_psi_error_r", "ego_kappa_r", "ego_lane_id"
        ]
        self.reset()
    
    def reset(self):
        self._ref_path_id = None
        self._ref_path = None
        self._s_condition_ego = None
        self._d_condition_ego = None

    """ TODO: add far point features """
    def get_obs(self, ego_state, *args):
        """ Compute the following ego features
        
        -----------------
        lane offset
        lane longitudinal speed
        lane lateral speed
        lane heading error
        lane curvature
        lane id
        -----------------
        
        Args:
            ego_state (np.array): size=[state_dim]

        Returns:
            ego_measurements (np.array): ego vehicle measurements. size=[5]
            ego_pos (np.array): ego position. size=[2]
        """
        x_ego, y_ego, vx_ego, vy_ego, psi_ego, l_ego, w_ego = (
            ego_state[[
                self.x_idx, self.y_idx, self.vx_idx, self.vy_idx, 
                self.psi_idx, self.l_idx, self.w_idx
            ]]
        )
        
        # match lanes
        if self._ref_path is None:
            self._ref_path_id = self.map_data.match_lane(x_ego, y_ego, psi_ego, l_ego, w_ego)
            self._ref_path = self.map_data.lanes[self._ref_path_id].centerline.frenet_path
        
        # get ego frenet coordinates
        v_ego = np.sqrt(vx_ego**2 + vy_ego**2)
        s_condition_ego, d_condition_ego = self._ref_path.cartesian_to_frenet(
            x_ego, y_ego, v_ego, None, psi_ego, None, order=2
        )
        
        # compute features
        [s, ds], [d, dd] = s_condition_ego, d_condition_ego 
        psi_tan = self._ref_path.get_tangent(s)
        kappa_r = self._ref_path.get_curvature(s)
        psi_error_r = wrap_angles(psi_ego - psi_tan)
        
        ego_measurements = np.array([d, ds, dd, psi_error_r, kappa_r, self._ref_path_id])
        ego_pos = ego_state[[self.x_idx, self.y_idx]]

        self._s_condition_ego = s_condition_ego
        self._d_condition_ego = d_condition_ego
        return ego_measurements, ego_pos


class LeadVehicleSensor:
    def __init__(self, map_data, max_range=60., state_keys=STATE_KEYS, track_lv=True):
        self.map_data = map_data
        self.max_range = max_range
        self.track_lv = track_lv # whether to keep tracking a single lv

        self.x_idx = state_keys.index("x")
        self.y_idx = state_keys.index("y")
        self.vx_idx = state_keys.index("vx")
        self.vy_idx = state_keys.index("vy")
        self.psi_idx = state_keys.index("psi_rad")
        self.l_idx = state_keys.index("length")
        self.w_idx = state_keys.index("width")
        self.id_idx = state_keys.index("track_id")
        
        self.feature_names = [
            "lv_s_rel", "lv_ds_rel", "lv_inv_tau", "lv_d", "lv_dd", "lv_track_id"
        ]
        self.reset()

    def reset(self):
        self._ref_path_id = None
        self._ref_path = None
        self._lv_track_id = None

    def get_obs(self, ego_state, agent_states):
        """ Compute the following lead vehicle related features
        
        -----------------
        relative distance
        relative speed
        inverse ttc
        lv lane offset
        lv lane lateral speed
        lv track id
        -----------------

        Args:
            ego_state (np.array): size=[state_dim]
            agent_states (np.array): size=[num_agents, state_dim]

        Returns:
            lv_measurements (np.array): lead vehicle measurements. size=[5]
            lv_pos (np.array): lead vehicle position. size=[2]
        """
        # default measurements
        lv_measurements = np.array([self.max_range, 0., 0, 0., 0., 0.]) 
        lv_pos = np.nan * np.ones(2)

        x_ego, y_ego, vx_ego, vy_ego, psi_ego, l_ego, w_ego = (
            ego_state[[
                self.x_idx, self.y_idx, self.vx_idx, self.vy_idx, 
                self.psi_idx, self.l_idx, self.w_idx
            ]]
        )
        relative_states = agent_states - ego_state

        # remove out of range agents
        distances = np.linalg.norm(relative_states[:, :2], axis=-1)
        agent_states = agent_states[distances <= self.max_range].copy()
        
        # match lanes
        if self._ref_path is None:
            self._ref_path_id = self.map_data.match_lane(x_ego, y_ego, psi_ego, l_ego, w_ego)
            self._ref_path = self.map_data.lanes[self._ref_path_id].centerline.frenet_path
        
        # get ego frenet coordinates
        v_ego = np.sqrt(vx_ego**2 + vy_ego**2)
        s_condition_ego, d_condition_ego = self._ref_path.cartesian_to_frenet(
            x_ego, y_ego, v_ego, None, psi_ego, None, order=2
        )
        
        if self._lv_track_id is not None and self.track_lv:
            agent_states = agent_states[np.where(agent_states[:, self.id_idx] == self._lv_track_id)]
            if len(agent_states) > 0:
                x_agent = agent_states[0, self.x_idx]
                y_agent = agent_states[0, self.y_idx]
                v_agent = np.sqrt(agent_states[0, self.vx_idx]**2 + agent_states[0, self.vy_idx]**2)
                psi_agent = agent_states[0, self.psi_idx]
                l_agent = agent_states[0, self.l_idx]
                w_agent = agent_states[0, self.w_idx]
                id_agent = agent_states[0, self.id_idx]
                s_condition_agent, d_condition_agent = self._ref_path.cartesian_to_frenet(
                    x_agent, y_agent, v_agent, None, psi_agent, None, order=2
                )
                lv_measurements = self.get_measurements(
                    s_condition_ego, d_condition_ego, s_condition_agent, d_condition_agent, 
                    l_ego, l_agent, w_agent, id_agent
                )
                lv_pos = agent_states[0, :2]
            return lv_measurements, lv_pos

        # filter agent not in the same lane
        agent_path_ids = np.array([self.map_data.match_lane(
            agent_states[i, self.x_idx], 
            agent_states[i, self.y_idx],
            agent_states[i, self.psi_idx],
            agent_states[i, self.l_idx],
            agent_states[i, self.w_idx],
        ) for i in range(len(agent_states))])
        is_same_lane = np.where(agent_path_ids == self._ref_path_id)
        agent_states = agent_states[is_same_lane].copy() 

        # get agent frenet coordinates
        headway_distance_lv = np.inf
        for i in range(len(agent_states)):
            x_agent = agent_states[i, self.x_idx]
            y_agent = agent_states[i, self.y_idx]
            v_agent = np.sqrt(agent_states[i, self.vx_idx]**2 + agent_states[i, self.vy_idx]**2)
            psi_agent = agent_states[i, self.psi_idx]
            l_agent = agent_states[i, self.l_idx]
            w_agent = agent_states[i, self.w_idx]
            id_agent = agent_states[i, self.id_idx]
            s_condition_agent, d_condition_agent = self._ref_path.cartesian_to_frenet(
                x_agent, y_agent, v_agent, None, psi_agent, None, order=2
            )
            headway_distance = s_condition_agent[0] - s_condition_ego[0]
            if headway_distance > 0 and headway_distance < headway_distance_lv:
                headway_distance_lv = headway_distance
                lv_measurements = self.get_measurements(
                    s_condition_ego, d_condition_ego, s_condition_agent, d_condition_agent, 
                    l_ego, l_agent, w_agent, id_agent
                )
                lv_pos = agent_states[i, :2]
                self._lv_track_id = agent_states[i, self.id_idx]
        
        return lv_measurements, lv_pos 

    def get_measurements(
        self, s_condition_ego, d_condition_ego, s_condition_agent, d_condition_agent, 
        l_ego, l_agent, w_agent, id_agent
        ):
        s_rel = s_condition_agent[0] - s_condition_ego[0]
        ds_rel = s_condition_agent[1] - s_condition_ego[1]
        lv_measurements = np.zeros(6) 
        lv_measurements[0] = s_rel - l_agent / 2 - l_ego / 2
        lv_measurements[1] = s_condition_agent[1] - s_condition_ego[1]
        lv_measurements[2] = calc_looming(s_rel, ds_rel, l_agent, w_agent, l_ego)
        lv_measurements[3] = d_condition_agent[0]
        lv_measurements[4] = d_condition_agent[1]
        lv_measurements[5] = id_agent
        return lv_measurements
    
    @staticmethod
    def calc_looming(s_rel, ds_rel, l, w, l0, max_val=10):
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
        inv_tau = np.clip(theta_dot / (theta + 1e-6), -max_val, max_val)
        return inv_tau


class FollowVehicleSensor:
    def __init__(self, map_data, max_range=60., state_keys=STATE_KEYS, track_fv=True):
        self.map_data = map_data
        self.max_range = max_range
        self.track_fv = track_fv # whether to keep tracking a single fv

        self.x_idx = state_keys.index("x")
        self.y_idx = state_keys.index("y")
        self.vx_idx = state_keys.index("vx")
        self.vy_idx = state_keys.index("vy")
        self.psi_idx = state_keys.index("psi_rad")
        self.l_idx = state_keys.index("length")
        self.w_idx = state_keys.index("width")
        self.id_idx = state_keys.index("track_id")
        
        self.feature_names = [
            "fv_s_rel", "fv_ds_rel", "fv_inv_tau", "fv_d", "fv_dd", "fv_track_id"
        ]
        self.reset()

    def reset(self):
        self._ref_path_id = None
        self._ref_path = None
        self._fv_track_id = None

    def get_obs(self, ego_state, agent_states):
        """ Compute the following follow vehicle related features
        
        -----------------
        relative distance
        relative speed
        inverse ttc
        fv lane offset
        fv lane lateral speed
        fv track id
        -----------------

        Args:
            ego_state (np.array): size=[state_dim]
            agent_states (np.array): size=[num_agents, state_dim]

        Returns:
            fv_measurements (np.array): follow vehicle measurements. size=[5]
            fv_pos (np.array): follow vehicle position. size=[2]
        """
        # default measurements
        fv_measurements = np.array([self.max_range, 0., 0, 0., 0., 0.]) 
        fv_pos = np.nan * np.ones(2)

        x_ego, y_ego, vx_ego, vy_ego, psi_ego, l_ego, w_ego = (
            ego_state[[
                self.x_idx, self.y_idx, self.vx_idx, self.vy_idx, 
                self.psi_idx, self.l_idx, self.w_idx
            ]]
        )
        relative_states = agent_states - ego_state

        # remove out of range agents
        distances = np.linalg.norm(relative_states[:, :2], axis=-1)
        agent_states = agent_states[distances <= self.max_range].copy()
        
        # match lanes
        if self._ref_path is None:
            self._ref_path_id = self.map_data.match_lane(x_ego, y_ego, psi_ego, l_ego, w_ego)
            self._ref_path = self.map_data.lanes[self._ref_path_id].centerline.frenet_path
        
        # get ego frenet coordinates
        v_ego = np.sqrt(vx_ego**2 + vy_ego**2)
        s_condition_ego, d_condition_ego = self._ref_path.cartesian_to_frenet(
            x_ego, y_ego, v_ego, None, psi_ego, None, order=2
        )
        
        if self._fv_track_id is not None and self.track_fv:
            agent_states = agent_states[np.where(agent_states[:, self.id_idx] == self._fv_track_id)]
            if len(agent_states) > 0:
                x_agent = agent_states[0, self.x_idx]
                y_agent = agent_states[0, self.y_idx]
                v_agent = np.sqrt(agent_states[0, self.vx_idx]**2 + agent_states[0, self.vy_idx]**2)
                psi_agent = agent_states[0, self.psi_idx]
                l_agent = agent_states[0, self.l_idx]
                w_agent = agent_states[0, self.w_idx]
                id_agent = agent_states[0, self.id_idx]
                s_condition_agent, d_condition_agent = self._ref_path.cartesian_to_frenet(
                    x_agent, y_agent, v_agent, None, psi_agent, None, order=2
                )
                fv_measurements = self.get_measurements(
                    s_condition_ego, d_condition_ego, s_condition_agent, d_condition_agent, 
                    l_ego, l_agent, w_agent, id_agent
                )
                fv_pos = agent_states[0, :2]
            return fv_measurements, fv_pos

        # filter agent not in the same lane
        agent_path_ids = np.array([self.map_data.match_lane(
            agent_states[i, self.x_idx], 
            agent_states[i, self.y_idx],
            agent_states[i, self.psi_idx],
            agent_states[i, self.l_idx],
            agent_states[i, self.w_idx],
        ) for i in range(len(agent_states))])
        is_same_lane = np.where(agent_path_ids == self._ref_path_id)
        agent_states = agent_states[is_same_lane].copy() 

        # get agent frenet coordinates
        tailway_distance_fv = np.inf
        for i in range(len(agent_states)):
            x_agent = agent_states[i, self.x_idx]
            y_agent = agent_states[i, self.y_idx]
            v_agent = np.sqrt(agent_states[i, self.vx_idx]**2 + agent_states[i, self.vy_idx]**2)
            psi_agent = agent_states[i, self.psi_idx]
            l_agent = agent_states[i, self.l_idx]
            w_agent = agent_states[i, self.w_idx]
            id_agent = agent_states[i, self.id_idx]
            s_condition_agent, d_condition_agent = self._ref_path.cartesian_to_frenet(
                x_agent, y_agent, v_agent, None, psi_agent, None, order=2
            )
            tailway_distance = s_condition_ego[0] - s_condition_agent[0]
            if tailway_distance > 0 and tailway_distance < tailway_distance_fv:
                tailway_distance_fv = tailway_distance
                fv_measurements = self.get_measurements(
                    s_condition_ego, d_condition_ego, s_condition_agent, d_condition_agent, 
                    l_ego, l_agent, w_agent, id_agent
                )
                fv_pos = agent_states[i, :2]
                self._fv_track_id = agent_states[i, self.id_idx]
        
        return fv_measurements, fv_pos 

    def get_measurements(
        self, s_condition_ego, d_condition_ego, s_condition_agent, d_condition_agent, 
        l_ego, l_agent, w_agent, id_agent
        ):
        s_rel = s_condition_ego[0] - s_condition_agent[0]
        ds_rel = s_condition_ego[1] - s_condition_agent[1]
        fv_measurements = np.zeros(6) 
        fv_measurements[0] = s_rel - l_agent / 2 - l_ego / 2
        fv_measurements[1] = ds_rel
        fv_measurements[2] = calc_looming(s_rel, ds_rel, l_agent, w_agent, l_ego)
        fv_measurements[3] = d_condition_agent[0]
        fv_measurements[4] = d_condition_agent[1]
        fv_measurements[5] = id_agent
        return fv_measurements
    
    @staticmethod
    def calc_looming(s_rel, ds_rel, l, w, l0, max_val=10):
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
        inv_tau = np.clip(theta_dot / (theta + 1e-6), -max_val, max_val)
        return inv_tau


class LidarSensor:
    """ 
    adapted from: 
    https://github.com/travelbureau/RareSim/blob/master/simulator/Vehicle.py 
    https://github.com/sisl/InteractionSimulator/blob/main/intersim/envs/intersimple.py
    """
    def __init__(self, num_beams=20, max_range=60., max_angle=np.deg2rad(360/2), state_keys=STATE_KEYS):
        self.num_beams = num_beams
        self.max_range = max_range
        self.state_keys = state_keys
        self.beam_angles = np.linspace(-np.pi, np.pi, num_beams + 1)[1:]
        self.valid_beam_idx = np.where((self.beam_angles >= -max_angle) & (self.beam_angles <= max_angle))[0]

        self.x_idx = state_keys.index("x")
        self.y_idx = state_keys.index("y")
        self.vx_idx = state_keys.index("vx")
        self.vy_idx = state_keys.index("vy")
        self.psi_idx = state_keys.index("psi_rad")
        self.l_idx = state_keys.index("length")
        self.w_idx = state_keys.index("width")

        self.feature_names = np.stack([
            [f"lidar_range_{i}" for i in range(self.num_beams)],
            [f"lidar_range_rate_{i}" for i in range(self.num_beams)]
        ], axis=-1).flatten().tolist() 
    
    def reset(self):
        pass 
    
    def get_obs(self, ego_state, agent_states):
        """ Compute lidar range and range-rate measurements at each angle offset
        
        Args:
            ego_state (np.array): size=[state_dim]
            agent_states (np.array): size=[num_agents, state_dim]
            
        Returns:
            lidar_measurements (np.array): range and range-rate measurements. 
                size=[num_beams, 2]
            lidar_hit_pos (np.array): lidar hit positions. size=[num_beams, 2]
        """
        # default measurements
        lidar_measurements = np.zeros((self.num_beams, 2))
        lidar_measurements[:, 0] = self.max_range
        lidar_hit_pos = np.nan * np.ones((self.num_beams, 2))

        relative_states = agent_states - ego_state
        
        # remove out of range agents
        distances = np.linalg.norm(relative_states[:, :2], axis=-1)
        agent_states = agent_states[distances <= self.max_range].copy()
        relative_states = relative_states[distances <= self.max_range].copy()
        relative_states = relative_states[:, [self.x_idx, self.y_idx, self.vx_idx, self.vy_idx]]
        
        if len(relative_states) == 0:
            return lidar_measurements, lidar_hit_pos

        # transform to ego frame
        psi = ego_state[self.psi_idx]
        x_ego, y_ego = coord_transformation(relative_states[:, 0], relative_states[:, 1], None, None, theta=psi)
        vx_ego, vy_ego = coord_transformation(relative_states[:, 2], relative_states[:, 3], None, None, theta=psi)
        relative_states[:, [0, 1]] = np.stack([x_ego, y_ego], axis=-1)
        relative_states[:, [2, 3]] = np.stack([vx_ego, vy_ego], axis=-1)

        # bin agent angles
        agent_angles = np.arctan2(relative_states[:, 1], relative_states[:, 0])
        offset = np.pi / self.num_beams # offset so zero is in the center
        bin_edges = np.linspace(-np.pi, np.pi, self.num_beams + 1)
        bin_assignments = np.digitize(agent_angles, bins=bin_edges[1:-1] + offset)
        bin_matrix = np.expand_dims(bin_assignments, -2) == np.expand_dims(np.arange(self.num_beams), -1)
        
        # take closest vehicle in each bin
        agent_bounding_boxes = compute_bounding_box(
            agent_states[:, self.x_idx],
            agent_states[:, self.y_idx],
            agent_states[:, self.psi_idx],
            agent_states[:, self.l_idx],
            agent_states[:, self.w_idx]
        )
        distances = agent_bounding_boxes - ego_state[:2].reshape(1, 1, 2)
        distances = np.linalg.norm(distances, axis=-1).min(-1)
        bin_distances = np.where(bin_matrix, np.expand_dims(distances, -2), np.inf)
        ray_hit_idx = np.argmin(bin_distances, axis=-1) # not hit bins are filled with dummy index
        
        # compute range and range rate
        range_ = distances[ray_hit_idx]
        beam_angles = bin_edges[1:]
        relative_vel = relative_states[ray_hit_idx][:, [2, 3]]
        range_rate = relative_vel[:, 0] * np.cos(beam_angles) + relative_vel[:, 1] * np.sin(beam_angles)
        
        # handle not hit bins
        num_hits = np.sum(bin_matrix, axis=-1)
        range_[num_hits == 0] = self.max_range
        range_rate[num_hits == 0] = 0.
        
        lidar_measurements = np.stack([range_, range_rate], axis=-1)
        lidar_hit_pos = agent_states[ray_hit_idx, :2]
        lidar_hit_pos[num_hits == 0] = np.nan

        lidar_measurements = lidar_measurements[self.valid_beam_idx]
        lidar_hit_pos = lidar_hit_pos[self.valid_beam_idx]
        return lidar_measurements, lidar_hit_pos
    
    """ NOTE: code below from RareSim """
    # def get_obs(self, ego_state, agent_state):
    #     """ Compute lidar range and range-rate measurements at each angle offset
        
    #     Args:
    #         ego_state (np.array): size=[state_dim]
    #         agent_state (np.array): size=[num_agents, state_dim]
            
    #     Returns:
    #         lidar_measurements (np.array): size=[num_beams, 2]
    #     """
    #     num_beams = self.num_beams
    #     max_range = self.max_range
    #     state_keys = self.state_keys

    #     x_id = state_keys.index("x")
    #     y_id = state_keys.index("y")
    #     vx_id = state_keys.index("vx")
    #     vy_id = state_keys.index("vy")
    #     psi_id = state_keys.index("psi_rad")
    #     l_id = state_keys.index("length")
    #     w_id = state_keys.index("width")

    #     ego_pose = ego_state[[x_id, y_id, psi_id]]

    #     lidar_measurements = np.empty((num_beams, 2))
    #     lidar_hits = np.nan * np.ones((num_beams, 2)) # position of hit lidar position
    #     beams = np.linspace(0, 2 * np.pi, num_beams + 1)[1:]
    #     for i, angle in enumerate(beams):
    #         ray_angle = angle + ego_state[psi_id]
    #         ray_vector = np.array([np.cos(ray_angle), np.sin(ray_angle)])
    #         range_ = max_range
    #         range_rate = 0.0
    #         for agent_id in range(len(agent_state)):
    #             bounding_box = compute_bounding_box(
    #                 agent_state[agent_id, x_id], 
    #                 agent_state[agent_id, y_id], 
    #                 agent_state[agent_id, psi_id],
    #                 agent_state[agent_id, l_id], 
    #                 agent_state[agent_id, w_id]
    #             )
                
    #             range_temp = _lidar_observation(ego_pose, ray_angle, bounding_box)
    #             if range_temp < range_:
    #                 range_ = range_temp
    #                 ego_speed = ego_state[[vx_id, vy_id]]
    #                 agent_speed = agent_state[agent_id, [vx_id, vy_id]]
    #                 relative_speed = agent_speed - ego_speed
    #                 range_rate = np.dot(relative_speed, ray_vector)

    #                 lidar_hits[i, 0] = agent_state[agent_id, x_id]
    #                 lidar_hits[i, 1] = agent_state[agent_id, y_id]
    #             lidar_measurements[i, 0] = range_
    #             lidar_measurements[i, 1] = range_rate
    #     return lidar_measurements, lidar_hits

    # def get_lidar_observation(self, x, y):
    #     """  """
    #     return
    
# def _lidar_observation(pose, beam_theta, agent_bounding_box):
#     ranges = np.empty((len(agent_bounding_box),))
#     for i in range(len(agent_bounding_box)):
#         ranges[i] = _range_observation(pose, beam_theta, agent_bounding_box[i-1, :], agent_bounding_box[i, :])
#     return np.amin(ranges)

# def _range_observation(pose, beam_theta, line_segment_a, line_segment_b):
#     o = pose[:2]
#     v1 = o - line_segment_a
#     v2 = line_segment_b - line_segment_a
#     v3 = np.array([np.cos(beam_theta + np.pi/2.), np.sin(beam_theta + np.pi/2.)])

#     denom = np.dot(v2, v3)

#     x = np.inf
#     if np.abs(denom) > 0.0:
#         d1 = cross(v2, v1)/denom  # length of ray (law of sines)
#         d2 = np.dot(v1, v3)/denom  # length of seg/v2 (law of sines)
#         if d1 >= 0 and d2 >= 0 and d2 <= 1.0:
#             x = d1
#     elif _are_collinear(pose, line_segment_a, line_segment_b):
#         dist_a = np.linalg.norm(line_segment_a - o)
#         dist_b = np.linalg.norm(line_segment_b - o)
#         x = np.minimum(dist_a, dist_b)
#     return x

# def _are_collinear(pt_a, pt_b, pt_c, tol=1e-8):
#     return np.abs(cross(pt_b-pt_a, pt_a-pt_c)) < tol

# def cross(vec1, vec2):
#     return vec1[0]*vec2[1] - vec1[1]*vec2[0]