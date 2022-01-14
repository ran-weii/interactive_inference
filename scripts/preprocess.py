import argparse
import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.kalman_filter import BatchKalmanFilter
from src.data.utils import derive_acc, normalize_pos

# set pandas display option
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../interaction-dataset-master"
    )
    parser.add_argument(
        "--kf_path", type=str, default="../exp/kalman_filter"
    )
    parser.add_argument("--scenario", type=str, default="Merging")
    parser.add_argument("--debug", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def kalman_smooth(df, kf):
    """ apply kalman smoothing to a dataframe of tracks """
    
    def smooth_one_track(df_one_track):
        df_ = df_one_track.copy()
        init_pos = df_.head(1)[["x", "y"]].values[:2]
        
        df_ = normalize_pos(df_)
        obs = df_[["x", "y", "vx", "vy", "vx_grad", "vy_grad"]].values
        
        s_mean, s_cov = kf.smooth(obs)
        s_mean[:, :2] = s_mean[:, :2] + init_pos.reshape(1, -1)
        
        df_s_mean = pd.DataFrame(
            s_mean,
            columns=["x_kf", "y_kf", "vx_kf", "vy_kf", "ax_kf", "ay_kf"]
        )
        return df_s_mean
        
    df_s_mean = df.groupby("track_id").apply(
        lambda x: smooth_one_track(x)
    ).reset_index().drop(columns="level_1")
    df_out = pd.concat([df, df_s_mean], axis=1)
    return df_out

def main(arglist):
    # load kalman filter 
    with open(os.path.join(arglist.kf_path, "model.p"), "rb") as f:
        kf = pickle.load(f)
        dt = 0.1
    
    # load data
    map_path = os.path.join(arglist.data_path, "maps")
    scenario_paths = glob.glob(
        os.path.join(arglist.data_path, "recorded_trackfiles", "*/")
    )
    
    scenario_counter = 0
    for i, scenario_path in enumerate(scenario_paths):
        scenario = os.path.basename(os.path.dirname(scenario_path))
        if arglist.scenario in scenario:
            print(f"Scenario {i+1}: {scenario}")
            
            track_paths = glob.glob(
                os.path.join(scenario_path, "*.csv")
            )
            
            for j, track_path in enumerate(track_paths):
                df_track = pd.read_csv(track_path)
                
                # take one track
                df_track_sub = df_track.loc[df_track["track_id"].isin([1, 2])]
                df_track_sub = derive_acc(df_track_sub, dt)
                df_track_sub = kalman_smooth(df_track_sub, kf)
                print(df_track_sub)
                
                if arglist.debug:
                    break
            
            scenario_counter += 1
        else:
            pass 
        
        if arglist.debug and scenario_counter > 0:
            break
    
    return 

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)