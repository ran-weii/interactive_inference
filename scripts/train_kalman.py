import argparse
import os
import glob
import time
import numpy as np
import pandas as pd
from pykalman import KalmanFilter

# set pandas display option
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

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
    parser.add_argument("--epochs", type=int, default=1, help="number of em epochs, default=1")
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
    df_all_tracks = []
    num_cars = 0
    for i, track_path in enumerate(track_paths):
        df_track = pd.read_csv(track_path)

        # reset track_id
        df_track["track_id"] = pd.Categorical(df_track["track_id"]).codes
        df_track["track_id"] += num_cars
        
        df_all_tracks.append(df_track)
        num_cars += len(np.unique(df_track["track_id"]))
        
    df_all_tracks = pd.concat(df_all_tracks, axis=0)
    
    # subsample tracks
    df_all_tracks = df_all_tracks.groupby("track_id").filter(lambda x: x.shape[0] >= arglist.min_len)
    
    assert len(df_all_tracks["track_id"].unique()) > arglist.num_tracks
    id = np.random.choice(
        len(df_all_tracks["track_id"].unique()), 
        size=(arglist.num_tracks,), replace=False
    )
    track_id = df_all_tracks["track_id"].unique()[id]
    
    df_all_tracks = df_all_tracks.loc[df_all_tracks["track_id"].isin(track_id)].reset_index(drop=True)
    
    df_all_tracks = df_all_tracks.groupby("track_id").head(arglist.min_len).reset_index(drop=True)

    # derive acceleration
    dt = df_track.groupby("track_id")["timestamp_ms"].diff().mean() / 1000
    f_grad = lambda x: pd.DataFrame(np.gradient(x, dt), index=x.index)
    df_all_tracks["vx_grad"] = df_all_tracks.groupby("track_id")["vx"].apply(f_grad)
    df_all_tracks["vy_grad"] = df_all_tracks.groupby("track_id")["vy"].apply(f_grad)
    
    # normalize data
    f_norm = lambda x: x - x.iloc[0]
    df_all_tracks["x"] = df_all_tracks.groupby("track_id")["x"].apply(f_norm)
    df_all_tracks["y"] = df_all_tracks.groupby("track_id")["y"].apply(f_norm)
    
    # pack observations
    obs = df_all_tracks[["x", "y", "vx", "vy", "vx_grad", "vy_grad"]]
    obs = obs.values
    
    # init parameters with const acc model
    eps = 1e-4
    transition_matrix = np.array(
        [[1, 0, dt,  0,  0,  0],
         [0, 1,  0, dt,  0,  0],
         [0, 0,  1,  0, dt,  0],
         [0, 0,  0,  1,  0, dt],
         [0, 0,  0,  0,  1,  0],
         [0, 0,  0,  0,  0,  1]]
    ) + eps
    observation_matrix = np.eye(6) + eps
    transition_covariance = 0.01 * np.eye(6) + eps
    observation_covariance = 0.01 * np.eye(6) + eps
    transition_offset = np.zeros((6)) + eps
    observation_offset = np.zeros((6)) + eps
    initial_state_mean = np.zeros((6,))
    initial_state_covariance = 0.01 * np.eye(6) + eps
    
    # train
    # kf = KalmanFilter(
    #     transition_matrix,
    #     observation_matrix,
    #     transition_covariance,
    #     observation_covariance,
    #     transition_offset,
    #     observation_offset,
    #     initial_state_mean,
    #     initial_state_covariance,
    #     random_state=arglist.seed,
    #     em_vars=[
    #     "transition_matrices", "observation_matrices",
    #     "transition_covariance", "observation_covariance",
    #     "transition_offsets", "observation_offsets", 
    #     "initial_state_mean", "initial_state_covariance"
    #     ]
    # )
    
    # loss = []
    # for i in range(arglist.epochs):
    #     kf = kf.em(X=obs, n_iter=1)    
    #     loss.append(kf.loglikelihood(obs))
    
    # print(loss)
    return 

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)