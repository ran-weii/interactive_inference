import argparse
import os
import glob
import json
import pickle
import numpy as np
from numpy.core.fromnumeric import trace
import pandas as pd
import matplotlib.pyplot as plt
from src.data.lanelet import load_lanelet_df
from src.data.kalman_filter import BatchKalmanFilter
from src.data.agent_filter import get_lane_pos, get_neighbor_vehicles
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
    parser.add_argument(
        "--map_path", type=str, default="../exp/lanelet"
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
    ).reset_index().drop(columns=["level_1", "track_id"])
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
            print(f"\nScenario {i+1}: {scenario}")
            
            track_paths = glob.glob(
                os.path.join(scenario_path, "*.csv")
            )
            
            # df_lanelet = load_map(arglist.map_path, scenario)
            df_lanelet = load_lanelet_df(
                os.path.join(arglist.map_path, scenario + ".json")
            )
            
            for j, track_path in enumerate(track_paths):
                filename = os.path.basename(track_path)
                print(f"track_file: {filename}")
                
                df_track = pd.read_csv(track_path)
                
                if arglist.debug:
                    # df_track = df_track.loc[df_track["track_id"].isin([1, 2, 3])].iloc[:100]
                    df_track = df_track.iloc[:10000]
                
                # preprocess raw data
                df_track["psi_rad"] = np.clip(df_track["psi_rad"], -np.pi, np.pi)
                # df_track = derive_acc(df_track, dt)
                # df_track = kalman_smooth(df_track, kf)
                
                # df_lane_assigned = df_track.apply(
                #     lambda x: assign_lane_pos(
                #         x["x_kf"], x["y_kf"], x["psi_rad"], df_lanelet
                #     ), axis=1
                # )
                # df_track = pd.concat([df_track, df_lane_assigned], axis=1)
                # print(df_track)
                
                # sort track by frame
                # df_track = df_track.sort_values(by=["frame_id"]).reset_index(drop=True)
                # df_track = df_track.sort_values(by=["track_id", "frame_id"]).reset_index(drop=True)
                
                """ test find neighbors """
                df_track = df_track.loc[df_track["frame_id"] == 1].reset_index(drop=True)
                df_lane_assigned = df_track.apply(
                    lambda x: get_lane_pos(
                        x["x"], x["y"], x["psi_rad"], df_lanelet
                    ), axis=1
                )
                df_track = pd.concat([df_track, df_lane_assigned], axis=1)
                
                df_ego = df_track.loc[df_track["track_id"] == 2].iloc[0]
                df_frame = df_track.loc[df_track["frame_id"] == df_ego["frame_id"]]
                
                df_neighbor = get_neighbor_vehicles(df_ego, df_frame, r=50)
                print(df_neighbor)
                # find neighbouring vehicle
                
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