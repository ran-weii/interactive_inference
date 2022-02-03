from enum import unique
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from src.data.agent_filter import get_neighbor_vehicles

def filter_car_follow_eps(df_track, min_eps_len):
    """
    Args:
        df_track (pd.dataframe): track dataframe
        min_eps_len (int): min episode length

    Returns:
        df_track (pd.dataframe): track dataframe with filtered "eps_id" and "eps_len" fields
    """
    df_track["eps_id"] = df_track["scenario"] + '_' + df_track["record_id"].apply(str) + \
        "_" + df_track["track_id"].apply(str) + "_" + df_track["car_follow_eps"].apply(str)
    is_new_eps = df_track["eps_id"].ne(df_track["eps_id"].shift().bfill())
    
    df_eps_len = df_track.groupby("eps_id").size().reset_index()
    df_eps_len.columns = ["eps_id", "eps_len"]
    df_track = df_track.merge(df_eps_len, how="outer", on="eps_id")
    
    slice_id = np.all(np.stack(
        [df_track["eps_len"] >= min_eps_len, df_track["car_follow_eps"] != -1]
    ), axis=0)
    df_track["eps_id"].loc[slice_id] = np.cumsum(is_new_eps[slice_id])
    df_track["eps_id"].loc[slice_id == False] = -1
    df_track["eps_len"].loc[slice_id == False] = -1
    return df_track
    
    
class EgoDataset(Dataset):
    """ Dataset for general car following """
    def __init__(
        self, df_track, df_lanelet, min_eps_len=10, 
        max_dist=50., max_agents=10, car_following=True
        ):
        super().__init__()
        assert all(v in df_track.columns for v in ["track_id", "car_follow_eps"])
        if car_following:
            df_track = filter_car_follow_eps(df_track, min_eps_len)
            unique_eps = df_track["eps_id"].unique()
            self.unique_eps = unique_eps[unique_eps != -1]
        else:
            raise NotImplementedError
            
        self.df_track = df_track.copy()
        self.df_lanelet = df_lanelet.copy()
        self.max_dist = max_dist
        self.max_agents = max_agents
        
        self.meta_fields = ["scenario", "record_id", "track_id", "agent_type"]
        self.ego_fields = [
            "x", "y", "vx", "vy", "psi_rad", "length", "width", "track_id",
            "lane_left_type", "lane_left_min_dist", 
            "lane_right_type", "lane_right_min_dist"
        ]
        self.agent_fields = self.ego_fields[:-4] + ["is_lead", "is_left", "dist_to_ego"]
        self.act_fields = ["ax", "ay"]
            
    def __len__(self):
        return len(self.unique_eps)
    
    def __getitem__(self, idx):
        df_ego = self.df_track.loc[self.df_track["eps_id"] == self.unique_eps[idx]]
        
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
        
        obs_agents = -1 * np.ones((T, self.max_agents, len(self.agent_fields)))
        for t in range(T):
            frame_id = df_ego["frame_id"].iloc[t]
            df_frame = df_record.loc[df_record["frame_id"] == frame_id]
            
            df_neighbor = get_neighbor_vehicles(df_ego.iloc[t], df_frame, self.max_dist)
            df_neighbor = df_neighbor[self.agent_fields]
            
            num_agents = min(len(df_neighbor), self.max_agents)
            obs_agents[t, :num_agents] = df_neighbor.to_numpy()[:num_agents]
        return obs_agents


class SimpleEgoDataset(EgoDataset):
    """ Dataset for lead vehicle following only """
    def __init__(self, df_track, df_lanelet, min_eps_len=10, max_dist=50.):
        super().__init__(
            df_track, df_lanelet, min_eps_len, 
            max_dist, max_agents=1, car_following=True
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
        
        df_agents = df_agents.loc[df_agents["frame_id"].isin(df_ego["frame_id"].values)]
        df_agents["is_lead"] = True
        df_agents["is_left"] = False
        df_agents["dist_to_ego"] = np.sqrt(
            (df_agents["x"] - df_ego["x"])**2 + (df_agents["y"] - df_ego["y"])**2
        )
        
        obs_agents = -1 * np.ones((T, self.max_agents, len(self.agent_fields)))
        obs_agents[:, 0, :] = df_agents[self.agent_fields].values
        return obs_agents