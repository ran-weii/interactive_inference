import numpy as np
import pandas as pd
from tqdm import tqdm

class VehicleTrajectoreis:
    def __init__(self, svt, track_ids, t_range):
        self.svt = svt
        self.track_ids = track_ids
        self.t_range = t_range


def create_svt_from_df(df):
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
    for fid in tqdm(df["frame_id"].unique()):
        df_frame = df.loc[df["frame_id"] == fid]
        state = df_frame[state_keys].values
        tid = df_frame["track_id"].values

        svt.append(state)
        track_ids.append(tid)

    t_start = df.groupby("track_id")["frame_id"].head(1).values # first time step of every track
    t_end = df.groupby("track_id")["frame_id"].tail(1).values # last time step of every track
    t_range = np.stack([t_start, t_end]).T

    svt_object = VehicleTrajectoreis(svt, track_ids, t_range)
    return svt_object
