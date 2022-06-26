import numpy as np
import pandas as pd
import torch
from src.data.geometry import wrap_angles, coord_transformation

""" TODO: 
make lane feature calculation a part of the observer 
make observer a wrapper for simulation
"""

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
        "psi_rel", # relativ eheading
        "loom_s", 
        "loom_d"
    ],
}

class Observer:
    """ Observer object that computes features in the frenet frame """
    default_ego_features = ["ds", "dd", "d", "psi_error_r", "kappa_r"]
    default_relative_features = ["s_rel", "d_rel", "ds_rel", "dd_rel", "psi_rel", "loom_s"]
    
    def __init__(
        self, map_data, ego_features=default_ego_features, 
        relative_features=default_relative_features, fp_dist=30.):
        """
        Args:
            map_data (MapReader): map data object
            ego_features (list, optional): list of strings for ego features
            lane_features (list, optional): list of strings for lane features
            relative_features (list, optional): list of strings for relative features
            fp_dist (float): far point distance
        """
        assert all([f in FEATURE_SET["ego"] for f in ego_features]), "ego feature not in FEATURE_SET"
        assert all([f in FEATURE_SET["relative"] for f in relative_features]), "relative feature not in FEATURE_SET"
        self.map_data = map_data
        self.feature_set = ego_features + relative_features
        self.fp_dist = fp_dist
        self.eps = 1e-6
        
        self.reset()
    
    def reset(self):
        self._ref_path = None
        self._trajectory = None

    def observe(self, obs_env):
        """ Convert environment observations into a vector 
        
        Args:
            obs_env (np.array): environment state [x, y, vx, vy, psi]
        
        Returns:
            obs (torch.tensor): agent observation [1, obs_dim]
        """
        ego_feature_dict = self.compute_ego_features(obs_env)
        relative_feature_dict = self.compute_relative_features(obs_env)
        
        obs_dict = {**ego_feature_dict, **relative_feature_dict}
        obs = [obs_dict[k] for k in self.feature_set]
        obs = torch.tensor(obs).view(1, -1).to(torch.float32)
        return obs
    
    def compute_ego_features(self, obs_env):
        ego_state = obs_env["ego"]
        [x_ego, y_ego, vx_ego, vy_ego, psi_ego] = ego_state
        
        # match current lane
        if self._ref_path is None:
            ref_lane_id = self.map_data.match_lane(x_ego, y_ego)
            self.ref_path = self.map_data.lanes[ref_lane_id].centerline.frenet_path
        
        # convert to frenet coordinate
        v_ego = np.sqrt(vx_ego**2 + vy_ego**2)
        s_condition_ego, d_condition_ego = self.ref_path.cartesian_to_frenet(
            x_ego, y_ego, v_ego, None, psi_ego, None, order=2
        )
        
        # compute ego features
        [s, ds], [d, dd] = s_condition_ego, d_condition_ego 
        
        # compute lane features
        lbd, rbd = 1.8 - d, 1.8 + d
        psi_tan = self.ref_path.get_tangent(s)
        kappa_r = self.ref_path.get_curvature(s)
        psi_error_r = wrap_angles(psi_ego - psi_tan)
        
        # compute far point features
        fp_dict = {}
        for i in range(4):
            s_fp = min(s + self.fp_dist * (i + 1), self.ref_path.arc_length)
            psi_fp = self.ref_path.get_tangent(s_fp)
            psi_error_fp = wrap_angles(psi_ego - psi_fp)
            fp_dict[f"psi_error_fp_{i}"] = psi_error_fp
        
        obs_dict = {
            "s": s, "d": d, "ds": ds, "dd": dd, "lbd": lbd, "rbd": rbd, 
            "kappa_r": kappa_r, "psi_error_r": psi_error_r, **fp_dict
        }
        return obs_dict

    def compute_relative_features(self, obs_env):
        ego_state = obs_env["ego"]
        agent_state = obs_env["agents"]
        
        [x_ego, y_ego, vx_ego, vy_ego, psi_ego] = ego_state
        [x_agent, y_agent, vx_agent, vy_agent, psi_agent] = agent_state[0]
        
        # convert to frenet coordinate
        v_ego = np.sqrt(vx_ego**2 + vy_ego**2)
        s_condition_ego, d_condition_ego = self.ref_path.cartesian_to_frenet(
            x_ego, y_ego, v_ego, None, psi_ego, None, order=2
        )
        v_agent = np.sqrt(vx_agent**2 + vy_agent**2)
        s_condition_agent, d_condition_agent = self.ref_path.cartesian_to_frenet(
            x_agent, y_agent, v_agent, None, psi_agent, None, order=2
        )
        
        s_condition_rel = s_condition_agent - s_condition_ego
        d_condition_rel = d_condition_agent - d_condition_ego
        [s_rel, ds_rel], [d_rel, dd_rel] = s_condition_rel, d_condition_rel
        psi_rel = wrap_angles(psi_ego - psi_agent)
        
        # compute looming
        loom_s = np.clip(ds_rel / (s_rel + self.eps), -10, 10)
        loom_d = np.clip(dd_rel / (d_rel + self.eps), -10, 10)
        obs_dict = {
            "s_rel": s_rel, "d_rel": d_rel, "psi_rel": psi_rel, 
            "ds_rel": ds_rel, "dd_rel": dd_rel,
            "loom_s": loom_s, "loom_d": loom_d
        }
        return obs_dict


class RelativeObserver:
    def __init__(self, map_data, frame="frenet", fields="rel"):
        """
        Args:
            map_data (MapReader): MapReader class
            frame (str): coordinate frame for agent. One of ["frenet", "carte"]
            fields (str): one of ["rel", "two_point", "lv_fv", "full"]
                "rel": lead vehicle with left and right bound observations
                "two_point": lead vehicle with center line and heading error observations
                "lv_fv": lead and following with center line and heading error observations
                "full": all adjacent vehicles with center line and heading error observations
        """
        self.map_data = map_data
        self.frame = frame
        self.agent_ids = [
            "lead_track_id", "follow_track_id", "left_track_id", "right_track_id",
            "left_lead_track_id", "right_lead_track_id", 
            "left_follow_track_id", "right_follow_track_id"
        ]
        self.state_fields = ["vx_ego", "vy_ego"]        
        self.rel_fields = [
            "x_rel", "y_rel", "vx_rel", "vy_rel", "psi_rel", "loom_x"
        ]
        
        # mean observations
        self.mean_obs = {
            "lead_track_id": [14, -0.3, 0, -0.14, -0.03], 
            "follow_track_id": [-14, -0.3, 0, 0.14, 0.03], 
            "left_track_id": [0, 4.5, 0, 0, 0, 0], 
            "right_track_id": [0, -4.5, 0, 0, 0, 0],
            "left_lead_track_id": [13, 4, 0, -0.14, -0.03], 
            "right_lead_track_id": [13, -4, 0, -0.14, -0.03], 
            "left_follow_track_id": [-13, 4, 0, 0.14, 0.03], 
            "right_follow_track_id": [-13, -4, 0, 0.14, 0.03],
        }
        if fields == "rel":
            self.lane_fields = ["left_bound_dist", "right_bound_dist"]
        else:
            self.lane_fields = ["center_line_dist", "psi_error"]

        if fields == "lv_fv":
            self.agent_ids = self.agent_ids[:2]
        elif fields == "full":
            pass
        else:
            self.agent_ids = self.agent_ids[:1]
        self.agent_fields = [f"{f}_{i}" for i in range(len(self.agent_ids)) for f in self.rel_fields]

        self.ego_fields = self.state_fields + self.lane_fields + self.agent_fields
        self.act_fields = ["ax_ego", "ay_ego"]
        
        self.obs_dim = len(self.ego_fields)
        self.ctl_dim = len(self.act_fields)

        self.reset()
    
    def reset(self):
        self.t = 0
        self.target_lane_id = None
    
    def glob_to_ego(self, x, y, psi_coord):
        return coord_transformation(x, y, None, None, theta=psi_coord, inverse=False)

    def ego_to_glob(self, x, y, psi_coord):
        return coord_transformation(x, y, None, None, theta=psi_coord, inverse=True)
    
    def get_lane_obs(self, x, y, psi, df=None):
        target_lane_id = None if self.target_lane_id is None else self.target_lane_id
        if df is None:
            if self.frame == "carte":
                (lane_id, cell_id, left_bound_dist, right_bound_dist, 
                center_line_dist, cell_headings) = self.map_data.match(
                    x, y, target_lane_id=target_lane_id
                )
                cell_heading = cell_headings[2, 2]
                psi_error = wrap_angles(psi - cell_heading)
                self.target_lane_id = lane_id
                psi_coord = psi
            elif self.frame == "frenet":
                (lane_id, psi_tan, center_line_dist, left_bound_dist, right_bound_dist, 
                wp_coords, cell_headings) = self.map_data.match_frenet(
                    float(x), float(y), target_lane_id=target_lane_id
                )
                cell_heading = cell_headings[2]
                psi_error = wrap_angles(psi - psi_tan)
                self.target_lane_id = lane_id
                psi_coord = psi_tan
        else:
            lane_id = df["lane_id"].values
            left_bound_dist = df["left_bound_dist"].values
            right_bound_dist = df["right_bound_dist"].values
            center_line_dist = df["center_line_dist"].values
            cell_heading = df["cell_heading_2"].values
            if self.frame == "carte":
                psi_error = wrap_angles(df["psi_rad"].values - cell_heading)
            else:
                psi_error = wrap_angles(df["psi_rad"].values - df["psi_tan"])
            psi_coord = df["psi_tan"].values
        
        psi_error = np.nan_to_num(psi_error, nan=0)
        if "center_line_dist" in self.ego_fields:
            obs_dict = {"center_line_dist": center_line_dist, "psi_error": psi_error}
        else:
            obs_dict = {"left_bound_dist": left_bound_dist, "right_bound_dist": right_bound_dist}
        return obs_dict, psi_coord
    
    def get_state_obs(self, obs_ego, psi_coord):
        vx, vy = obs_ego[:, 2], obs_ego[:, 3]
        vx_ego, vy_ego = self.glob_to_ego(vx, vy, psi_coord)
        return {"vx_ego": vx_ego, "vy_ego": vy_ego}

    def get_rel_obs(self, obs_ego, obs_agent, mean_obs, psi_coord):
        """ 
        Args:
            obs_ego (np.array): self obs [x, y, vx, vy, psi]
            obs_agent (np.array): agent obs [x, y, vx, vy, psi]
        
        Returns:
            obs_dict (dict): self and relative obs in ego frame
        """
        vx, vy = obs_ego[:, 2], obs_ego[:, 3]
        psi = obs_ego[:, 4]
        rel_obs = obs_agent - obs_ego
        
        x_rel, y_rel = self.glob_to_ego(rel_obs[:, 0], rel_obs[:, 1], psi_coord)
        vx_rel, vy_rel = self.glob_to_ego(rel_obs[:, 2], rel_obs[:, 3], psi_coord)
        psi_rel = wrap_angles(rel_obs[:, 4])
        
        # handle missing agent
        x_rel = np.nan_to_num(x_rel, nan=mean_obs[0])
        y_rel = np.nan_to_num(y_rel, nan=mean_obs[1])
        vx_rel = np.nan_to_num(vx_rel, nan=mean_obs[2])
        vy_rel = np.nan_to_num(vy_rel, nan=mean_obs[3])
        psi_rel = np.nan_to_num(psi_rel, nan=mean_obs[4])
        loom_x = vx_rel / (x_rel + 1e-6)

        obs_dict = {
            "x_rel": x_rel, "y_rel": y_rel, "vx_rel": vx_rel, 
            "vy_rel": vy_rel, "psi_rel": psi_rel, "loom_x": loom_x
        }
        return obs_dict

    def observe(self, obs):
        obs_ego = obs["ego"].reshape(1, -1).astype(np.float64)
        
        lane_obs, psi_coord = self.get_lane_obs(obs_ego[:, 0], obs_ego[:, 1], obs_ego[:, 4])
        state_obs = self.get_state_obs(obs_ego, psi_coord)
        obs_dict = {**state_obs, **lane_obs}

        for i, agent_id in enumerate(self.agent_ids):
            obs_agent = obs["agents"][i].reshape(1, -1).astype(np.float64)
            rel_obs = self.get_rel_obs(obs_ego, obs_agent, self.mean_obs[agent_id], psi_coord)
            rel_obs = {f"{k}_{i}": v for k, v in rel_obs.items()}
            obs_dict = {**obs_dict, **rel_obs}
        
        df_obs = pd.DataFrame(obs_dict)[self.ego_fields]
        obs = torch.from_numpy(df_obs.values).view(1, -1).to(torch.float32)

        self.t += 1
        self.psi_coord = psi_coord
        return obs
    
    def control(self, ctl, obs):
        """ Project ego centric control to world coordinate """
        vx, vy = obs["ego"][2], obs["ego"][3]
        [ax_ego, ay_ego] = ctl.data.view(-1).numpy().tolist()
        ax, ay = self.ego_to_glob(ax_ego, ay_ego, self.psi_coord)
        ctl = np.hstack([ax, ay])
        return ctl
    
    def observe_df(self, df):
        """ Make relative observations and concat with df """
        obs_fields = ["x", "y", "vx", "vy", "psi_rad"]
        def get_agent_df(df, agent_id):
            df = df.copy()
            df_agent = df.copy()
            df_agent.columns = [c + "_agent" for c in df_agent.columns]
            df_joint = df.merge(
                df_agent,
                left_on=["scenario", "record_id", "frame_id", agent_id],
                right_on=["scenario_agent", "record_id_agent", "frame_id_agent", "track_id_agent"],
                how="left"
            )
            df_agent = df_joint[[f + "_agent" for f in obs_fields]]
            df_agent.columns = obs_fields
            return df_agent
        
        df_ego = df.copy()[obs_fields]
        lane_obs, psi_coord = self.get_lane_obs(None, None, df["psi_rad"], df=df)
        state_obs = self.get_state_obs(df_ego.values, psi_coord)
        obs_dict = {**state_obs, **lane_obs}
        
        for i, agent_id in enumerate(self.agent_ids):
            df_agent = get_agent_df(df, agent_id)
            rel_obs = self.get_rel_obs(df_ego.values, df_agent.values, self.mean_obs[agent_id], psi_coord)
            rel_obs = {f"{k}_{i}": v for k, v in rel_obs.items()}
            obs_dict = {**obs_dict, **rel_obs}
        
        df_obs = pd.DataFrame(obs_dict)[self.ego_fields]
        
        ax, ay = self.glob_to_ego(df["ax"], df["ay"], psi_coord)
        df_obs["ax_ego"] = ax
        df_obs["ay_ego"] = ay
        
        df = df.drop(columns=["left_bound_dist", "right_bound_dist", "center_line_dist"])
        df = pd.concat([df, df_obs], axis=1)
        return df