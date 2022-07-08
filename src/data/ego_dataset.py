import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from src.simulation.observers import AUGMENTATION_PARAMETERS

def collate_fn(batch):
    """ Collate batch to have the same sequence length """
    pad_obs = pad_sequence([b["ego"] for b in batch])
    pad_act = pad_sequence([b["act"] for b in batch])
    mask = torch.all(pad_obs != 0, dim=-1).to(torch.float32)
    return pad_obs, pad_act, mask

def sample_sequence(seq_len, max_seq_len, gamma=1.):
    """ Sample a segment of the sequence

    Args:
        seq_len (int): original sequence length
        max_seq_len (int): maximum sequence length
        gamma (float, optional): higher gamma bias towards 
            sampling smaller time steps. Default=1.
    
    Returns:
        sample_id (np.array): sample id
    """
    sample_id = np.arange(seq_len)
    if seq_len <= max_seq_len:
        return sample_id
    else:
        gamma = 1.
        candidates = np.arange(seq_len - max_seq_len)
        p = (candidates + 1) ** -float(gamma)
        p /= p.sum()
        id_start = np.random.choice(candidates, p=p)
        sample_id = sample_id[id_start:id_start+max_seq_len]
    return sample_id

def aug_flip_lr(obs, act, feature_set):
    """ Data augmentation by a left-right flip wrt the road 
    
    Args:
        obs (np.array): observation matrix. size=[T, obs_dim]
        act (np.array): action matrix. size=[T, act_dim]
        feature_set (list): list of strings feature names

    Returns:
        obs_aug (np.array): augmented observations. size=[T, obs_dim]
        act_aug (np.array): augmented actions. size=[T, act_dim]
    """
    obs_aug = obs.copy()
    act_aug = act.copy()
    if np.random.randint(2) == 1: # perform augmentation with 0.5 probability
        for i, f in enumerate(feature_set):
            flippable = AUGMENTATION_PARAMETERS[f]["flip_lr"]
            if flippable == 1:
                obs_aug[:, i] *= -1
            
            # handle special case
            if f == "lbd":
                obs_aug[:, i] = 1.8 + (1.8 - obs_aug[:, i])
            elif f == "rbd":
                obs_aug[:, i] = 1.8 - (obs_aug[:, i] - 1.8)
        act_aug[:, 1] *= -1
    return obs_aug, act_aug

""" NOTE: 
simulation should not be done with fixed neighbors, 
it should be similar to the realtime neighbor graph in sisl interaction simulator,
ignore this fact for now 
"""
class EgoDataset(Dataset):
    """ Dataset for raw state outputs """
    def __init__(
        self, df_track, train_labels_col=None, max_dist=50., max_agents=8
        ):
        super().__init__()
        eps_id = df_track["eps_id"].values.copy()
        if train_labels_col is not None:
            eps_id[df_track[train_labels_col] == 0] = np.nan
        
        unique_eps = np.unique(eps_id)
        self.unique_eps = unique_eps[np.isnan(unique_eps) == False]
        self.df_track = df_track.copy()
        self.max_dist = max_dist
        self.max_agents = max_agents
        
        """ TODO: remove lbd and rbd from ego fields and animation visualizer """
        self.meta_fields = ["track_id", "eps_id"]
        self.ego_fields = [
            "x", "y", "vx", "vy", "psi_rad", "length", "width", "track_id"
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

    
class RelativeDataset(EgoDataset):
    def __init__(self, df_track, feature_set, action_set, train_labels_col=None, 
        min_eps_len=50, max_eps_len=1000, augmentation=[]):
        super().__init__(df_track, train_labels_col)
        assert set(feature_set).issubset(set(df_track.columns))
        self.max_eps_len = max_eps_len
        self.ego_fields = feature_set
        self.act_fields = action_set
        self.augmentation = augmentation
        
    def __getitem__(self, idx):
        df_ego = self.df_track.loc[
            self.df_track["eps_id"] == self.unique_eps[idx]
        ].reset_index(drop=True)
        
        sample_ids = sample_sequence(len(df_ego), self.max_eps_len, gamma=1.)
        df_ego = df_ego.iloc[sample_ids].reset_index(drop=True)
        
        obs_meta = df_ego[self.meta_fields].iloc[0].to_numpy()
        obs_ego = df_ego[self.ego_fields].to_numpy()
        act_ego = df_ego[self.act_fields].to_numpy()
        
        for aug in self.augmentation:
            obs_ego, act_ego = aug(obs_ego, act_ego, self.ego_fields)

        out_dict = {
            "meta": torch.from_numpy(obs_meta).to(torch.float32),
            "ego": torch.from_numpy(obs_ego).to(torch.float32),
            "agents": None,
            "act": torch.from_numpy(act_ego).to(torch.float32)
        }
        return out_dict