import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from src.simulation.observers import AUGMENTATION_PARAMETERS

def collate_fn(batch):
    """ Collate batch of dict to have the same sequence length """
    assert isinstance(batch[0], dict)
    keys = list(batch[0].keys())
    pad_batch = {k: pad_sequence([b[k] for b in batch]) for k in keys}
    mask = pad_sequence([torch.ones(len(b[keys[0]])) for b in batch])
    return pad_batch, mask
    
def sample_sequence(seq_len, max_seq_len, gamma=1.):
    """ Sample a segment of the sequence

    Args:
        seq_len (int): original sequence length
        max_seq_len (int): maximum sequence length
        gamma (float, optional): higher gamma bias towards 
            sampling smaller time steps. 0. is uniform sampling. Default=1.
    
    Returns:
        sample_id (np.array): sample id
    """
    sample_id = np.arange(seq_len)
    if seq_len <= max_seq_len:
        return sample_id
    else:
        candidates = np.arange(seq_len - max_seq_len)
        p = (candidates + 1) ** -float(gamma)
        p /= p.sum()
        id_start = np.random.choice(candidates, p=p)
        sample_id = sample_id[id_start:id_start+max_seq_len]
    return sample_id

def aug_flip_lr(obs, act, feature_set):
    """ Data augmentation by a left-right flip wrt the road 
    
    Args:
        obs ([np.array, torch.tensor]): observation matrix. size=[T, obs_dim]
        act ([np.array, torch.tensor]): action matrix. size=[T, act_dim]
        feature_set (list): list of strings feature names

    Returns:
        obs_aug ([np.array, torch.tensor]): augmented observations. size=[T, obs_dim]
        act_aug ([np.array, torch.tensor]): augmented actions. size=[T, act_dim]
    """
    if isinstance(obs, np.ndarray):
        obs_aug = obs.copy()
        act_aug = act.copy()
    elif isinstance(obs, torch.Tensor):
        obs_aug = obs.clone()
        act_aug = act.clone()
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

def create_svt_from_df(df_track, unique_eps_id, agent_id_fields, state_fields, act_fields, meta_fields, verbose=True):
    """ Create stacked vehicle trajectories from track dataframe
    
    Args:
        df_track (pd.dataframe): track dataframe with field eps_id and agent_id_fields
        unique_eps_id (list): unique episode id that will be processed
        agent_id_fields (list): list of agent id fields in df
        state_fields (list): list of state fields in df
        meta_fields (list): list of meta fields in df
        verbose (bool, optional): whether to verbose during processing. Default=True
    
    Returns:
        svt (list): list of stacked vehicle trajectries. Each svt is a dict with keys:
            ["ego", "agent", "meta"]. ego_size=[T, state_dim], agent_size=[T, num_agent, state_dim],
            meta_size=[meta_dim]
    """
    def concat_agent_df(df_ego, df_track, agent_id):
        df_agent = df_track.copy()
        df_agent.columns = [c + "_agent" for c in df_agent.columns]
        df_joint = df_ego.merge(
            df_agent,
            left_on=["scenario", "record_id", "frame_id", agent_id],
            right_on=["scenario_agent", "record_id_agent", "frame_id_agent", "track_id_agent"],
            how="left"
        )
        df_agent = df_joint[[f + "_agent" for f in state_fields]]
        return df_agent
    
    svt = []
    pbar = tqdm(unique_eps_id) if verbose else unique_eps_id
    for i, eps_id in enumerate(pbar):
        if verbose:
            pbar.set_description(f"create svt for eps: {eps_id}")

        df_eps = df_track.loc[df_track["eps_id"] == eps_id].reset_index(drop=True)
        ego_states = df_eps[state_fields].values
        ego_acts = df_eps[act_fields].values
        
        T = len(df_eps)
        agent_states = np.nan * np.ones((T, len(agent_id_fields), len(state_fields)))
        for j, agent_id in enumerate(agent_id_fields):
            df_agent = concat_agent_df(df_eps, df_track, agent_id)
            agent_states[:, j] = df_agent.values
        
        meta_states = df_eps[meta_fields].values[0]
        svt.append({
            "meta": meta_states,
            "ego": ego_states,
            "agents": agent_states,
            "act": ego_acts
        })
    return svt 


class BaseDataset(Dataset):
    def __init__(self, df_track, train_labels_col=None, max_eps=500, seed=0):
        super().__init__()
        np.random.seed(seed)
        eps_id = df_track["eps_id"].values.copy()
        if train_labels_col is not None:
            eps_id[df_track[train_labels_col] == 0] = np.nan
        unique_eps = np.unique(eps_id)
        self.unique_eps = unique_eps[np.isnan(unique_eps) == False]
        
        if len(self.unique_eps) > max_eps:
            sample_id = np.random.choice(
                np.arange(len(self.unique_eps)), max_eps, replace=False
            )
            self.unique_eps = self.unique_eps[sample_id]

        self.df_track = df_track.copy()
    
    def __len__(self):
        return len(self.unique_eps)

    def __getitem__(self, idx):
        raise NotImplementedError


class EgoDataset(BaseDataset):
    """ Dataset for raw ego and agent state outputs 
        Used to feed the simulator, assume other agents do not change trajectories
    """
    def __init__(self, df_track, train_labels_col=None, max_eps=500, create_svt=False, seed=0):
        """
        Args:
            df_track (pd.dataframe): track dataframe
            train_labels_col (str): columns name of train labels. Used to reduce unique episode ids
            max_eps (int, optional): maximum number of episodes to store. Default=500
            create_svt (bool, optional): whether to create svt in init. Default=False
        """
        super().__init__(df_track, train_labels_col, max_eps, seed)
        self.create_svt = create_svt
        
        self.meta_fields = ["track_id", "eps_id"]
        self.ego_fields = [
            "x", "y", "vx", "vy", "psi_rad", "length", "width", "track_id"
        ]
        # self.agent_id_fields = [
        #     "lead_track_id", "follow_track_id", "left_track_id", "right_track_id",
        #     "left_lead_track_id", "right_lead_track_id",
        #     "left_follow_track_id", "right_follow_track_id"
        # ]
        self.agent_id_fields = ["lv_track_id"]
        self.act_fields = ["ax", "ay"]
        
        if create_svt:
            self.svt = create_svt_from_df(
                self.df_track, self.unique_eps, self.agent_id_fields, 
                self.ego_fields, self.act_fields, self.meta_fields
            )        
            
    def __len__(self):
        return len(self.unique_eps)
    
    def __getitem__(self, idx):
        if self.create_svt:
            out_dict = self.svt[idx]
        else:
            unique_eps = [self.unique_eps[idx]]
            svt = create_svt_from_df(
                self.df_track, unique_eps, self.agent_id_fields, 
                self.ego_fields, self.act_fields, self.meta_fields, verbose=False
            )
            out_dict = svt[0]
        return out_dict

    
class RelativeDataset(BaseDataset):
    def __init__(self, df_track, feature_set, action_set, train_labels_col=None, 
        max_eps=500, max_eps_len=1000, gamma=0., state_action=False, augmentation=[], 
        device=torch.device("cpu"), seed=0):
        """
        Args:
            df_track (pd.dataframe): dataframe storing features and actions
            feature_set (list): list of feature names
            action_set (list): list of action names
            action_bins (int): used to create action discretizer
            train_labels_col (str): training label column
            max_esp (int): maximum number of episode. Exceeded episodes will be subsampled
            max_eps_len (int): maximum number of steps in an episode. Exceeded steps will be subsampled.
            gamma (float): sequence sampling factor.
            state_action (bool): whether to return state action only instead of sequences.
            augmentation (list): list of data augmentation functions.
            device (torch.device): data device.
            seed (int): data sampling seed.
        """
        super().__init__(df_track, train_labels_col, max_eps, seed)
        assert set(feature_set).issubset(set(df_track.columns))
        self.max_eps_len = max_eps_len
        self.gamma = gamma
        self.state_action = state_action
        self.ego_fields = feature_set
        self.act_fields = action_set
        self.meta_fields = ["track_id", "eps_id"]
        self.augmentation = augmentation
        self.device = device
        
        # preload data to device for offline training
        self.data_ego = [
            torch.from_numpy(
                self.df_track.loc[self.df_track["eps_id"] == i][self.ego_fields].values.astype(np.float32)
            ).to(torch.float32).to(self.device) for i in self.unique_eps
        ]
        self.data_act = [
            torch.from_numpy(
                self.df_track.loc[self.df_track["eps_id"] == i][self.act_fields].values.astype(np.float32)
            ).to(torch.float32).to(self.device) for i in self.unique_eps
        ]
        self.data_meta = [
            torch.from_numpy(
                self.df_track.loc[self.df_track["eps_id"] == i][self.meta_fields].values.astype(np.float32)
            ).to(torch.float32).to(self.device) for i in self.unique_eps
        ] 

        if state_action:
            self.data_ego = torch.cat(self.data_ego, dim=0)
            self.data_act = torch.cat(self.data_act, dim=0)
            self.data_meta = torch.cat(self.data_meta, dim=0)
    
    def __len__(self):
        if self.state_action:
            return len(self.df_track)
        else:
            return super().__len__()

    def __getitem__(self, idx):
        obs_meta = self.data_meta[idx]
        obs_ego = self.data_ego[idx]
        act_ego = self.data_act[idx]

        if not self.state_action:
            sample_ids = sample_sequence(len(obs_ego), self.max_eps_len, gamma=self.gamma)
            obs_meta = obs_meta[sample_ids][0]
            obs_ego = obs_ego[sample_ids]
            act_ego = act_ego[sample_ids] 
        
        for aug in self.augmentation:
            obs_ego, act_ego = aug(obs_ego, act_ego, self.ego_fields)

        out_dict = {
            "obs": obs_ego,
            "act": act_ego,
            "meta": obs_meta,
        }
        return out_dict