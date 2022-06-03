import numpy as np
import pandas as pd
import torch
from src.data.geometry import wrap_angles, coord_transformation

class RelativeObserver:
    def __init__(self, map_data, fields="rel"):
        """
        Args:
            map_data (MapReader): MapReader class
            fields (str): one of ["rel", "two_point", "lv_fv", "full"]
                "rel": lead vehicle with left and right bound observations
                "two_point": lead vehicle with center line and heading error observations
                "lv_fv": lead and following with center line and heading error observations
                "full": all adjacent vehicles with center line and heading error observations
        """
        self.map_data = map_data
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
    
    def glob_to_ego(self, x, y, vx, vy, theta=None):
        return coord_transformation(x, y, vx, vy, theta=theta, inverse=False)

    def ego_to_glob(self, x, y, vx, vy, theta=None):
        return coord_transformation(x, y, vx, vy, theta=theta, inverse=True)
    
    def get_lane_obs(self, x, y, psi, df=None):
        target_lane_id = None if self.target_lane_id is None else self.target_lane_id
        if df is None:
            (lane_id, cell_id, left_bound_dist, right_bound_dist, 
            center_line_dist, cell_headings) = self.map_data.match(
                x, y, target_lane_id=target_lane_id
            )
            cell_heading = cell_headings[2, 2]
            psi_error = wrap_angles(psi - cell_heading)
            self.target_lane_id = lane_id
        else:
            lane_id = df["lane_id"].values
            left_bound_dist = df["left_bound_dist"].values
            right_bound_dist = df["right_bound_dist"].values
            center_line_dist = df["center_line_dist"].values
            cell_heading = df["cell_heading_2"].values
            psi_error = wrap_angles(df["psi_rad"].values - cell_heading)
        
        psi_error = np.nan_to_num(psi_error, nan=0)
        if "center_line_dist" in self.ego_fields:
            obs_dict = {"center_line_dist": center_line_dist, "psi_error": psi_error}
        else:
            obs_dict = {"left_bound_dist": left_bound_dist, "right_bound_dist": right_bound_dist}
        return obs_dict
    
    def get_state_obs(self, obs_ego):
        vx, vy = obs_ego[:, 2], obs_ego[:, 3]
        vx_ego, vy_ego = self.glob_to_ego(vx, vy, vx, vy)
        return {"vx_ego": vx_ego, "vy_ego": vy_ego}

    def get_rel_obs(self, obs_ego, obs_agent, mean_obs):
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
        
        x_rel, y_rel = self.glob_to_ego(rel_obs[:, 0], rel_obs[:, 1], vx, vy, theta=psi)
        vx_rel, vy_rel = self.glob_to_ego(rel_obs[:, 2], rel_obs[:, 3], vx, vy, theta=psi)
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
        
        state_obs = self.get_state_obs(obs_ego)
        lane_obs = self.get_lane_obs(obs_ego[:, 0], obs_ego[:, 1], obs_ego[:, 4])
        obs_dict = {**state_obs, **lane_obs}

        for i, agent_id in enumerate(self.agent_ids):
            obs_agent = obs["agents"][i].reshape(1, -1).astype(np.float64)
            rel_obs = self.get_rel_obs(obs_ego, obs_agent, self.mean_obs[agent_id])
            rel_obs = {f"{k}_{i}": v for k, v in rel_obs.items()}
            obs_dict = {**obs_dict, **rel_obs}
        
        df_obs = pd.DataFrame(obs_dict)[self.ego_fields]
        obs = torch.from_numpy(df_obs.values).view(1, -1).to(torch.float32)

        self.t += 1
        return obs

    def control(self, ctl, obs):
        """ Project ego centric control to world coordinate """
        vx, vy = obs["ego"][2], obs["ego"][3]
        [ax_ego, ay_ego] = ctl.data.view(-1).numpy().tolist()
        ax, ay = self.ego_to_glob(ax_ego, ay_ego, vx, vy)
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
        state_obs = self.get_state_obs(df_ego.values)
        lane_obs = self.get_lane_obs(None, None, None, df=df)
        obs_dict = {**state_obs, **lane_obs}

        for i, agent_id in enumerate(self.agent_ids):
            df_agent = get_agent_df(df, agent_id)
            rel_obs = self.get_rel_obs(df_ego.values, df_agent.values, self.mean_obs[agent_id])
            rel_obs = {f"{k}_{i}": v for k, v in rel_obs.items()}
            obs_dict = {**obs_dict, **rel_obs}
        
        df_obs = pd.DataFrame(obs_dict)[self.ego_fields]
        
        ax, ay = self.glob_to_ego(df["ax"], df["ay"], df["vx"], df["vy"], theta=df["psi_rad"])
        df_obs["ax_ego"] = ax
        df_obs["ay_ego"] = ay
        
        df = df.drop(columns=["left_bound_dist", "right_bound_dist", "center_line_dist"])
        df = pd.concat([df, df_obs], axis=1)
        return df