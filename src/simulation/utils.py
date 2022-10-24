import numpy as np
import pandas as pd
from tqdm import tqdm

class VehicleTrajectoreis:
    def __init__(self, svt, track_ids, ego_track_ids, eps_ids, t_range):
        """
        Properties:
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
    
    Returns:
        svt (list): stack vehicle states for each frame. Each item in the list 
            consists of all vehicles in the frame. size=[num_frames]
        track_ids (list): vehicle track ids in each frame. size=[num_frames]
        t_range (list): start and end frame index of each vehicle. size=[num_tracks, 2]
    """
    state_keys = [
        "x", "y", "vx", "vy", "ax", "ay", "psi_rad", 
        "length", "width", "track_id"
    ]
    svt = [] # stacked vehicle state for each frame
    track_ids = [] # track id for each frame
    for fid in tqdm(np.sort(df["frame_id"].unique())):
        df_frame = df.loc[df["frame_id"] == fid]
        state = df_frame[state_keys].values
        tid = df_frame["track_id"].values

        svt.append(state)
        track_ids.append(tid)
    
    # get episode retrival indices
    df = df.assign(eps_id=df[eps_id_col])
    df_eps_head = df.groupby(eps_id_col).head(1)
    df_eps_tail = df.groupby(eps_id_col).tail(1)
    df_eps = df_eps_head.merge(df_eps_tail, on="eps_id")
    df_eps = df_eps.loc[df_eps["eps_id"].isna() == False].sort_values(by="eps_id").reset_index(drop=True)
    ego_track_ids = df_eps["track_id_x"].values.astype(int)
    eps_ids = df_eps["eps_id"].values.astype(int)
    t_range = df_eps[["frame_id_x", "frame_id_y"]].values.astype(int)
    
    svt_object = VehicleTrajectoreis(svt, track_ids, ego_track_ids, eps_ids, t_range)
    return svt_object
