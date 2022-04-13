import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from src.data.data_filter import (
    filter_car_follow_eps, get_relative_df, get_ego_centric_df)

class EgoDataset(Dataset):
    """ Dataset for general car following """
    def __init__(
        self, df_track, map_data, df_train_labels=None, min_eps_len=8, max_eps_len=100,
        max_dist=50., max_agents=8, car_following=True
        ):
        super().__init__()
        assert all(v in df_track.columns for v in ["track_id", "car_follow_eps"])
        if car_following:
            df_track = filter_car_follow_eps(df_track, min_eps_len)
        else:
            raise NotImplementedError
        
        """ TODO: 
        come up with a better solution to filter test set 
        pd merge wierdly duplicate rows
        """
        # filter test trajectories
        if df_train_labels is not None:
            merge_keys = ["scenario", "record_id", "track_id"]
            df_track = df_track.merge(
                df_train_labels, left_on=merge_keys, right_on=merge_keys, how="left"
            ).drop_duplicates(merge_keys + ["frame_id"], keep="last").reset_index(drop=True)
            df_track["eps_id"].loc[df_track["is_train"] == False] = -1
        
        unique_eps = df_track["eps_id"].unique()
        self.unique_eps = unique_eps[unique_eps != -1]
        self.df_track = df_track.copy()
        self.map_data = map_data
        self.min_eps_len = min_eps_len
        self.max_eps_len = max_eps_len
        self.max_dist = max_dist
        self.max_agents = max_agents
        
        self.meta_fields = ["scenario", "record_id", "track_id", "agent_type"]
        self.ego_fields = [
            "x", "y", "vx", "vy", "psi_rad", "length", "width", 
            "left_bound_dist", "right_bound_dist", "track_id"
        ]
        self.agent_id_fields = [
            "lead_track_id", "follow_track_id", "left_track_id", "right_track_id",
            "left_lead_track_id", "right_lead_track_id",
            "left_follow_track_id", "right_follow_track_id"
        ]
        self.act_fields = ["ax", "ay"]
            
    def __len__(self):
        return len(self.unique_eps)
    
    def __getitem__(self, idx):
        df_ego = self.df_track.loc[
            self.df_track["eps_id"] == self.unique_eps[idx]
        ].reset_index(drop=True)
        
        obs_meta = df_ego[self.meta_fields].iloc[0].to_numpy()
        obs_ego = df_ego[self.ego_fields].to_numpy()
        obs_agents = self.get_agent_obs(df_ego)
        act_ego = df_ego[self.act_fields].to_numpy()
        
        out_dict = {
            "meta": obs_meta,
            "ego": obs_ego,
            "agents": obs_agents,
            "act": act_ego
        }
        return out_dict
    
    def get_agent_obs(self, df_ego):
        scenario = df_ego["scenario"].iloc[0]
        record_id = df_ego["record_id"].iloc[0]
        ego_id = df_ego["track_id"].iloc[0]
        T = len(df_ego)
        
        df_scenario = self.df_track.loc[self.df_track["scenario"] == scenario]
        df_record = df_scenario.loc[df_scenario["record_id"] == record_id]
        df_record = df_record.loc[df_record["track_id"] != ego_id]
        
        obs_agents = -np.nan * np.ones((T, self.max_agents, len(self.ego_fields)))
        for t in range(T):
            frame_id = df_ego["frame_id"].iloc[t]
            df_frame = df_record.loc[df_record["frame_id"] == frame_id]
            
            agent_ids = df_ego.iloc[t][self.agent_id_fields].values
            for i, agent_id in enumerate(agent_ids):
                if not np.isnan(agent_id) and (i + 1) <= self.max_agents:
                    df_agent = df_frame.loc[df_frame["track_id"] == agent_id]
                    obs_agents[t, i] = df_agent[self.ego_fields].to_numpy().flatten()
        return obs_agents


class SimpleEgoDataset(EgoDataset):
    """ Dataset for lead vehicle following only """
    def __init__(self, df_track, df_lanelet, df_train_labels=None, 
                 min_eps_len=10, max_eps_len=100, max_dist=50.):
        super().__init__(
            df_track, df_lanelet, df_train_labels=df_train_labels, 
            min_eps_len=min_eps_len, max_eps_len=max_eps_len,
            max_dist=max_dist, max_agents=1, car_following=True
        )
    
    def get_agent_obs(self, df_ego):
        scenario = df_ego["scenario"].iloc[0]
        record_id = df_ego["record_id"].iloc[0]
        ego_id = df_ego["track_id"].iloc[0]
        lead_track_id = df_ego["lead_track_id"].iloc[0]
        T = len(df_ego)
        
        df_scenario = self.df_track.loc[self.df_track["scenario"] == scenario]
        df_record = df_scenario.loc[df_scenario["record_id"] == record_id]
        df_agents = df_record.loc[df_record["track_id"] == lead_track_id]
        
        df_agents = df_agents.loc[
            df_agents["frame_id"].isin(df_ego["frame_id"].values)
        ].sort_values("frame_id").reset_index(drop=True)
        df_agents["is_lead"] = True
        df_agents["is_left"] = False
        df_agents["dist_to_ego"] = np.sqrt(
            (df_agents["x"] - df_ego["x"])**2 + (df_agents["y"] - df_ego["y"])**2
        )
        
        # lead vehicle df should have the same length as ego
        obs_agents = -1 * np.ones((T, self.max_agents, len(self.agent_fields)))
        obs_agents[:, 0, :] = df_agents[self.agent_fields].values
        return obs_agents
    
class RelativeDataset(EgoDataset):
    def __init__(self, df_track, map_data, df_train_labels=None, 
                 min_eps_len=10, max_eps_len=100, max_dist=50., control_direction="both"):
        super().__init__(
            df_track, map_data, df_train_labels=df_train_labels, 
            min_eps_len=min_eps_len, max_eps_len=max_eps_len,
            max_dist=max_dist, max_agents=1, car_following=True
        )
        """ TODO: temporary data filtering solution """
        df_rel = get_relative_df(self.df_track, self.df_track["lead_track_id"])
        df_ego = pd.concat([self.df_track, df_rel], axis=1)
        self.df_track = get_ego_centric_df(df_ego)
        self.df_track["loom_x"] = self.df_track["vx_rel"] / (self.df_track["x_rel"] + 1e-6)
        
        self.ego_fields = [
            "vx_ego", "vy_ego", "left_bound_dist", "right_bound_dist", 
            "x_rel_ego", "y_rel_ego", "vx_rel_ego", 
            "vy_rel_ego", "psi_rad_rel", "loom_x"
        ]
        self.act_fields = ["ax_ego", "ay_ego"]
        
        """ TODO: temporary solution of feeding only longitudinal control """
        if control_direction == "both":
            pass
        elif control_direction == "lon":
            self.act_fields = ["ax_ego"]
        elif control_direction == "lat":
            self.act_fields = ["ay_ego"]
        else:
            raise ValueError
        
    def __getitem__(self, idx):
        df_ego = self.df_track.loc[
            self.df_track["eps_id"] == self.unique_eps[idx]
        ].reset_index(drop=True)
        
        """ TODO: add seed to max length filtering """
        if len(df_ego) > self.max_eps_len: 
            sample_id = np.random.randint(0, len(df_ego) - self.max_eps_len)
            df_ego = df_ego.iloc[sample_id:sample_id+self.max_eps_len]
        
        obs_meta = df_ego[self.meta_fields].iloc[0].to_numpy()
        obs_ego = df_ego[self.ego_fields].to_numpy()
        act_ego = df_ego[self.act_fields].to_numpy()
        out_dict = {
            "meta": None,
            "ego": torch.from_numpy(obs_ego).to(torch.float32),
            "agents": None,
            "act": torch.from_numpy(act_ego).to(torch.float32)
        }
        return out_dict
        
class RelativeDatasetNew(EgoDataset):
    def __init__(self, df_track, map_data, df_train_labels=None, 
        min_eps_len=10, max_eps_len=100, max_cells=5, lateral_control=True):
        super().__init__(
            df_track, None, df_train_labels=df_train_labels, 
            min_eps_len=min_eps_len, max_eps_len=max_eps_len,
            max_dist=50, max_agents=1, car_following=True
        )
        """ TODO: temporary data filtering solution """
        df_rel = get_relative_df(self.df_track, self.df_track["lead_track_id"])
        df_ego = pd.concat([self.df_track, df_rel], axis=1)
        self.df_track = get_ego_centric_df(df_ego)
        self.map_data = map_data
        self.max_cells = max_cells
        
        self.ego_fields = [
            "x", "y", "vx_ego", "vy_ego", 
            "left_bound_dist", "right_bound_dist", 
            "x_rel_ego", "y_rel_ego", "vx_rel_ego", 
            "vy_rel_ego", "psi_rad_rel", "psi_rad"
        ]
        self.act_fields = ["ax_ego", "ay_ego"]
        
        """ TODO: temporary solution of feeding only longitudinal control """
        if not lateral_control:
            self.act_fields = ["ax_ego"]
        
    def __getitem__(self, idx):
        df_ego = self.df_track.loc[
            self.df_track["eps_id"] == self.unique_eps[idx]
        ].reset_index(drop=True)
        
        """ TODO: add seed to max length filtering """
        if len(df_ego) > self.max_eps_len: 
            sample_id = np.random.randint(0, len(df_ego) - self.max_eps_len)
            df_ego = df_ego.iloc[sample_id:sample_id+self.max_eps_len]
        
        obs_ego = df_ego[self.ego_fields].to_dict(orient="list")
        act_ego = df_ego[self.act_fields].to_numpy()
        
        cell_headings = [self.map_data.match(
            obs_ego["x"][i], obs_ego["y"][i], max_cells=self.max_cells
        )[4] for i in range(len(df_ego))]
        
        obs_ego = {k: torch.tensor(v).to(torch.float32) for k, v in obs_ego.items()}
        obs_ego["cell_headings"] = torch.tensor(cell_headings).to(torch.float32)
        act_ego = torch.from_numpy(act_ego).to(torch.float32)
        
        out_dict = {
            "ego": obs_ego,
            "act": act_ego
        }
        return out_dict
    
    
def relative_collate_fn(batch):
    def pad_sequence(x, max_len):
        padding_shape = [max_len - len(x)] + list(x.shape)[1:]
        padding = torch.zeros(padding_shape)
        return torch.cat([x, padding], dim=0)
    
    obs_keys = [k for k in batch[0]["ego"].keys()]
    batch_len = [len(b["act"]) for b in batch]
    max_len = max(batch_len)
    mask = [torch.ones(l) for l in batch_len]
    
    pad_obs = {k: torch.stack(
        [pad_sequence(b["ego"][k], max_len) for b in batch]
    ).transpose(0, 1) for k in obs_keys}
    pad_act = torch.stack([pad_sequence(b["act"], max_len) for b in batch]).transpose(0, 1)
    mask = torch.stack([pad_sequence(m, max_len) for m in mask]).transpose(0, 1)
    return pad_obs, pad_act, mask