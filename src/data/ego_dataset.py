from enum import unique
import numpy as np
import pandas as pd
import torch
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

""" TODO: implement this for initial model """
class SimpleEgoDataset(Dataset):
    """ Dataset for lead vehicle following only """
    def __init__(self):
        super().__init__()
        
    def __len__(self):
        return 
    
    def __getitem__(self):
        return 

""" TODO: add acc to fields """
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
            "x", "y", "vx", "vy", "psi_rad", "length", "width", 
            "lane_left_type", "lane_left_min_dist", 
            "lane_right_type", "lane_right_min_dist"
        ]
        self.agent_fields = self.ego_fields[:-4] + ["is_lead", "is_left", "dist_to_ego"]
        self.act_fields = [""]
            
    def __len__(self):
        return len(self.unique_eps)
    
    def __getitem__(self, idx):
        df_ego = self.df_track.loc[self.df_track["eps_id"] == self.unique_eps[idx]]
        
        obs_meta = df_ego[self.meta_fields].iloc[0].to_numpy()
        obs_ego = df_ego[self.ego_fields].to_numpy()
        obs_agents = self.get_neighbor_vehicles(df_ego)
        
        out_dict = {
            "meta": obs_meta,
            "ego": obs_ego,
            "agents": obs_agents,
            "act": None
        }
        return out_dict
    
    def get_neighbor_vehicles(self, df_ego):
        scenario = df_ego["scenario"].iloc[0]
        record_id = df_ego["record_id"].iloc[0]
        T = len(df_ego)
        
        df_scenario = self.df_track.loc[self.df_track["scenario"] == scenario]
        df_record = df_scenario.loc[df_scenario["record_id"] == record_id]
        
        obs_agents = -1 * np.ones((T, self.max_agents, len(self.agent_fields)))
        for t in range(T):
            frame_id = df_ego["frame_id"].iloc[t]
            df_frame = df_record.loc[df_record["frame_id"] == frame_id]
            
            df_neighbor = get_neighbor_vehicles(df_ego.iloc[t], df_frame, self.max_dist)
            df_neighbor = df_neighbor[self.agent_fields]
            
            num_agents = min(len(df_neighbor), self.max_agents)
            obs_agents[t, :num_agents] = df_neighbor.to_numpy()[:num_agents]
        return obs_agents
    
if __name__ == "__main__":
    import os
    from src.data.lanelet import load_lanelet_df
    pd.set_option('display.max_columns', None)
    pd.options.mode.chained_assignment = None

    lanelet_path = "../../exp/lanelet"
    data_path = "../../interaction-dataset-master"
    scenario = "DR_CHN_Merging_ZS"
    filename = "vehicle_tracks_007.csv"
    
    df_lanelet = load_lanelet_df(os.path.join(lanelet_path, scenario + ".json"))
    df_processed = pd.read_csv(
        os.path.join(data_path, "processed_trackfiles", scenario, filename)
    )
    df_track = pd.read_csv(
        os.path.join(data_path, "recorded_trackfiles", scenario, filename)
    )
    df_track = df_track.merge(df_processed, on=["track_id", "frame_id"])
    
    min_eps_len = 50
    ego_dataset = EgoDataset(df_track, df_lanelet, min_eps_len)
    obs = ego_dataset[0]