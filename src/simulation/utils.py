import numpy as np
import pandas as pd
from tqdm import tqdm
from src.simulation.simulator import STATE_KEYS

class VehicleTrajectoreis:
    """ Stacked vehicle trajectories object """
    def __init__(self, svt, track_ids, ego_track_ids, eps_ids, t_range):
        """
        Args:
            svt (list): stack vehicle states for each frame. Each item in the list 
                consists of all vehicles in the frame. size=[num_frames]
            track_ids (list): vehicle track ids in each frame. size=[num_frames]
            ego_track_ids (np.array): track id of each ego vehicle. size=[num_eps]
            eps_ids (np.array): episode id of each trajectory. size=[num_eps]
            t_range (list): start and end frame index of each vehicle. size=[num_eps, 2]
        """
        self.svt = svt
        self.track_ids = track_ids
        self.ego_track_ids = ego_track_ids
        self.eps_ids = eps_ids
        self.t_range = t_range


def create_svt_from_df(df, eps_id_col="track_id"):
    """ Build stacked vehicle trajectories 
    
    Args:
        df (pd.dataframe): dataframe of track records
        eps_id_col (str): column containing episode id

    Returns:
        svt (list): stacked vehicle trajectories object
    """
    frame_ids = np.sort(df["frame_id"].unique())
    svt = [] # stacked vehicle state for each frame
    track_ids = [] # track ids for each frame
    for i in tqdm(frame_ids):
        df_frame = df.loc[df["frame_id"] == i]
        svt.append(df_frame[STATE_KEYS].values)
        track_ids.append(df_frame["track_id"].values)
    
    # get episode retrival indices
    df_eps_head = df.groupby(eps_id_col).head(1)
    df_eps_tail = df.groupby(eps_id_col).tail(1)
    df_eps = df_eps_head.merge(df_eps_tail, on="eps_id")
    df_eps = df_eps.loc[df_eps["eps_id"].isna() == False].sort_values(by="eps_id").reset_index(drop=True)
    ego_track_ids = df_eps["track_id_x"].values.astype(int)
    eps_ids = df_eps["eps_id"].values.astype(int)
    
    # map frame id to svt id
    frame_2_id = {frame_ids[i]: i for i in range(len(frame_ids))}
    t_start = df_eps["frame_id_x"].map(frame_2_id)
    t_end = df_eps["frame_id_y"].map(frame_2_id)
    t_range = np.stack([t_start, t_end]).T
    
    svt_object = VehicleTrajectoreis(svt, track_ids, ego_track_ids, eps_ids, t_range)
    return svt_object
