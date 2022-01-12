import argparse
import os
import glob
from typing_extensions import ParamSpec
import numpy as np
import pandas as pd

""" TODO
write tests:
check how original repo handle tracks to render
train kalman filter
"""
def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../interaction-dataset-master"
    )
    parser.add_argument("--scenario", type=str, default="DR_CHN_Merging_ZS")
    parser.add_argument("--num_files", type=int, default=3, help="number of files used to sample trajectories, default=3")
    parser.add_argument("--min_len", type=int, default=100, help="min track length, default=100")
    parser.add_argument("--num_tracks", type=int, default=50, help="number of tracks used to train kalman filter, default=50")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def main(arglist):
    np.random.seed(arglist.seed)
    
    track_paths = glob.glob(
        os.path.join(
            arglist.data_path, 
            "recorded_trackfiles", 
            arglist.scenario,
            "*.csv"
        )
    ) 
    track_id = np.random.randint(
        0, len(track_paths), size=(arglist.num_files)
    )
    track_paths = [track_paths[i] for i in track_id]
    
    
    # load data
    df_tracks = []
    num_cars = 0
    for i, track_path in enumerate(track_paths):
        df_track = pd.read_csv(track_path)

        # reset track_id
        df_track["track_id"] = pd.Categorical(df_track["track_id"]).codes
        df_track["track_id"] += num_cars
        
        df_tracks.append(df_track)
        num_cars += len(np.unique(df_track["track_id"]))
        
    df_tracks = pd.concat(df_tracks, axis=0)
    
    # subsample tracks
    df_tracks = df_tracks.groupby("track_id").filter(lambda x: x.shape[0] >= arglist.min_len)
    
    assert len(df_tracks["track_id"].unique()) > arglist.num_tracks
    id = np.random.choice(
        len(df_tracks["track_id"].unique()), 
        size=(arglist.num_tracks,), replace=False
    )
    track_id = df_tracks["track_id"].unique()[id]
    
    df_tracks = df_tracks.loc[df_tracks["track_id"].isin(track_id)].reset_index(drop=True)
    
    df_tracks = df_tracks.groupby("track_id").head(arglist.min_len).reset_index(drop=True)
    
    # derive acceleration
    dt = df_track.groupby("track_id")["timestamp_ms"].diff().mean()
    f_grad = lambda x: pd.DataFrame(np.gradient(x, dt), index=x.index) * 1000
    df_track["vx_grad"] = df_track.groupby("track_id")["vx"].apply(f_grad)
    df_track["vy_grad"] = df_track.groupby("track_id")["vy"].apply(f_grad)
    
    # init parameters
    
    # train
    return 

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)