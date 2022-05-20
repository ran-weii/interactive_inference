import numpy as np
import pandas as pd
import torch
from src.data.geometry import wrap_angles, coord_transformation

class RelativeObserver:
    def __init__(self, map_data, fields="rel"):
        self.map_data = map_data
        if fields == "rel":
            self.ego_fields = [
                "vx_ego", "vy_ego", "left_bound_dist", "right_bound_dist", 
                "x_rel_ego", "y_rel_ego", "vx_rel_ego", 
                "vy_rel_ego", "psi_rel_ego", "loom_x"
            ]
        elif fields == "two_point":
            self.ego_fields = [
                "vx_ego", "vy_ego", "center_line_dist", "psi_error", 
                "x_rel_ego", "y_rel_ego", "vx_rel_ego", 
                "vy_rel_ego", "psi_rel_ego", "loom_x"
            ]
        self.act_fields = ["ax_ego", "ay_ego"]
        
        self.obs_dim = len(self.ego_fields)
        self.ctl_dim = len(self.act_fields)

        self.reset()
    
    def reset(self):
        self.t = 0
        self.target_lane_id = None
    
    def glob_to_ego(self, x, y, vx, vy):
        return coord_transformation(x, y, vx, vy, inverse=False)

    def ego_to_glob(self, x, y, vx, vy):
        return coord_transformation(x, y, vx, vy, inverse=True)
    
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
    
    def get_rel_obs(self, obs_ego, obs_agent):
        """ 
        Args:
            obs_ego (np.array): self obs [x, y, vx, vy, psi]
            obs_agent (np.array): agent obs [x, y, vx, vy, psi]
        
        Returns:
            obs_dict (dict): self and relative obs in ego frame
        """
        vx, vy = obs_ego[:, 2], obs_ego[:, 3]
        rel_obs = obs_agent - obs_ego

        vx_ego, vy_ego = self.glob_to_ego(vx, vy, vx, vy)
        x_rel, y_rel = self.glob_to_ego(rel_obs[:, 0], rel_obs[:, 1], vx, vy)
        vx_rel, vy_rel = self.glob_to_ego(rel_obs[:, 2], rel_obs[:, 3], vx, vy)
        psi_rel = wrap_angles(rel_obs[:, 4])
        loom_x = vx_rel / (x_rel + 1e-6)

        obs_dict = {
            "vx_ego": vx_ego, "vy_ego": vy_ego, "x_rel_ego": x_rel, "y_rel_ego": y_rel,
            "vx_rel_ego": vx_rel, "vy_rel_ego": vy_rel, "psi_rel_ego": psi_rel, "loom_x": loom_x
        }
        return obs_dict

    def observe(self, obs):
        obs_ego = obs["ego"].reshape(1, -1).astype(np.float64)
        obs_agent = obs["agents"][0].reshape(1, -1).astype(np.float64)

        lane_obs = self.get_lane_obs(obs_ego[:, 0], obs_ego[:, 1], obs_ego[:, 4])
        rel_obs = self.get_rel_obs(obs_ego, obs_agent)
        
        obs_dict = rel_obs
        obs_dict.update(lane_obs)
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
        df_ego = df.copy()
        df_agent = df.copy()
        df_agent.columns = [c + "_agent" for c in df_agent.columns]
        df_joint = df_ego.merge(
            df_agent,
            left_on=["scenario", "record_id", "frame_id", "lead_track_id"],
            right_on=["scenario_agent", "record_id_agent", "frame_id_agent", "track_id_agent"],
            how="left"
        )
        obs_fields = ["x", "y", "vx", "vy", "psi_rad"]
        df_ego = df_joint[obs_fields]
        df_agent = df_joint[[f + "_agent" for f in obs_fields]]
        df_agent.columns = obs_fields
        
        # print("ego", df_ego.iloc[0])
        lane_obs = self.get_lane_obs(None, None, None, df=df)
        rel_obs = self.get_rel_obs(df_ego.values, df_agent.values)
        
        obs_dict = rel_obs
        obs_dict.update(lane_obs)
        df_obs = pd.DataFrame(obs_dict)[self.ego_fields]
        
        ax, ay = self.glob_to_ego(df["ax"], df["ay"], df["vx"], df["vy"])
        df_obs["ax_ego"] = ax
        df_obs["ay_ego"] = ay
        
        df = df.drop(columns=["left_bound_dist", "right_bound_dist", "center_line_dist"])
        df = pd.concat([df, df_obs], axis=1)
        return df