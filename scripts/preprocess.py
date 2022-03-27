import argparse
import os
import glob
import time
import pickle
import numpy as np
import pandas as pd
import swifter
from tqdm.auto import tqdm

from src.map_api.lanelet import MapData
from src.data.kalman_filter import BatchKalmanFilter
from src.data.agent_filter import (
    get_neighbor_vehicle_ids, get_car_following_episode)

# to use apply progress bar
tqdm.pandas()

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
    parser.add_argument("--use_kf", type=bool_, default=True, 
                        help="use kalman filter, default=False")
    parser.add_argument("--cell_len", type=float, default=10, 
                        help="length of drivable cells, default=10")
    parser.add_argument("--max_cells", type=int, default=5, 
                        help="maximum number of lookahead cells, default=5")
    parser.add_argument("--max_dist", type=float, default=50., 
                        help="max dist to be considered neighbor, default=50")
    parser.add_argument("--parallel", type=bool_, default=True, 
                        help="parallel apply, default=True")
    parser.add_argument("--save", type=bool_, default=True)
    parser.add_argument("--debug", type=bool_, default=False)
    arglist = parser.parse_args()
    return arglist

def derive_acc(df, dt):
    """ differentiate vel and add to df """
    df_ = df.copy()
    f_grad = lambda x: pd.DataFrame(np.gradient(x, dt), index=x.index)
    df_["vx_grad"] = df_.groupby("track_id")["vx"].apply(f_grad)
    df_["vy_grad"] = df_.groupby("track_id")["vy"].apply(f_grad)
    return df_

def normalize_pos(df):
    """ subtrack all pos by initial pos """
    df_ = df.copy()
    
    f_norm = lambda x: x - x.iloc[0]
    df_["x"] = df_.groupby("track_id")["x"].apply(f_norm)
    df_["y"] = df_.groupby("track_id")["y"].apply(f_norm)
    return df_

def smooth_one_track(df_one_track, kf):
    """ Apply kalman filter to one track """
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

def match_lane(map_data, x, y, max_cells):
    """ Match point (x, y) to lane """
    lane_id, cell_id, left_bound_dist, right_bound_dist, cell_headings = map_data.match(x, y, max_cells)
    df_lane_pos = pd.Series({
        "lane_id": lane_id, 
        "cell_id": cell_id, 
        "left_bound_dist": left_bound_dist,
        "right_bound_dist": right_bound_dist
    })
    return df_lane_pos

def parallel_apply(df, f, parallel, axis=1, desc=""):
    if parallel:
        return df.swifter.progress_bar(True, desc=desc).apply(f, axis=axis)
    else:
        print(desc)
        return df.progress_apply(f, axis=axis)

def preprocess_pipeline(df_track, map_data, max_cells, max_dist, dt, kf_filter, parallel=False):
    """ Preprocess pipeline: 
    1. Identify ego lane
    2. Identify lead vehicle id
    3. Identify car following episode
    4. Derive acceleration
    5. Filter vehicle dynamics
    """
    df_track["psi_rad"] = np.clip(df_track["psi_rad"], -np.pi, np.pi)
    
    # identify ego lane
    f_ego_lane = lambda x: match_lane(map_data, x["x"], x["y"], max_cells)
    df_ego_lane = parallel_apply(
        df_track, f_ego_lane, parallel, axis=1, desc="identify ego lane"
    )
    df_track = pd.concat([df_track, df_ego_lane], axis=1)
    
    # identify lead vehicle
    f_neighbor_vehicle = lambda x: get_neighbor_vehicle_ids(x, df_track, map_data, max_dist)
    df_lead_vehicle = parallel_apply(
        df_track, f_neighbor_vehicle, parallel, axis=1, desc="identify lead vehicle"
    ).reset_index(drop=True)
    
    print("identify car following episode")
    df_processed = pd.concat([df_ego_lane, df_lead_vehicle], axis=1)
    df_processed.insert(0, "track_id", df_track["track_id"])
    df_processed.insert(1, "frame_id", df_track["frame_id"])
    df_processed["car_follow_eps"] = get_car_following_episode(df_processed)
    
    print("derive acceleration")
    df_track = derive_acc(df_track, dt)
    df_processed["ax"] = df_track["vx_grad"]
    df_processed["ay"] = df_track["vy_grad"]
    
    print("filter vehicle dynamics")
    df_track_smooth = df_track.groupby("track_id").progress_apply(
            lambda x: smooth_one_track(x, kf_filter)
    ).reset_index().drop(columns=["level_1", "track_id"])
    df_processed = pd.concat([df_processed, df_track_smooth], axis=1)
    return df_processed

def main(arglist):
    # make save path
    save_path = os.path.join(arglist.data_path, "processed_trackfiles")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    # load kalman filter 
    with open(os.path.join(arglist.kf_path, "model.p"), "rb") as f:
        kf = pickle.load(f)
        dt = 0.1
    
    # load data
    scenario_paths = glob.glob(
        os.path.join(arglist.data_path, "recorded_trackfiles", "*/")
    )
    
    num_processed = 0
    for i, scenario_path in enumerate(scenario_paths):
        scenario = os.path.basename(os.path.dirname(scenario_path))
        
        if arglist.scenario in scenario:
            print(f"\nScenario {i+1}: {scenario}")
            
            track_paths = glob.glob(
                os.path.join(scenario_path, "*.csv")
            )
            
            map_data = MapData(cell_len=arglist.cell_len)
            map_data.parse(
                os.path.join(arglist.data_path, "maps", f"{scenario}.osm"), 
                verbose=True
            )
        
            for j, track_path in enumerate(track_paths):
                track_filename = os.path.basename(track_path)
                track_record_id = track_filename.replace(".csv", "").split("_")[-1]
                print(f"num_processed: {num_processed}, track_file: {track_filename}")
                
                df_track = pd.read_csv(track_path)
                
                if arglist.debug:
                    df_track = df_track.iloc[:5000]
                
                start_time = time.time()
                df_processed = preprocess_pipeline(
                    df_track, map_data, arglist.max_cells, 
                    arglist.max_dist, dt, kf, arglist.parallel
                )
                df_processed.insert(0, "scenario", scenario)
                df_processed.insert(1, "record_id", track_record_id)
                print(f"time: {np.round(time.time() - start_time, 2)}")
                
                if arglist.save:
                    scenario_save_path = os.path.join(save_path, scenario)
                    if not os.path.exists(scenario_save_path):
                        os.mkdir(scenario_save_path)
                        
                    # df_processed.to_csv(
                    #     os.path.join(scenario_save_path, track_filename), index=False
                    # )
                    df_processed.to_csv(
                        os.path.join(scenario_save_path, "test.csv"), index=False
                    )
            
                num_processed += 1
                if num_processed > 0:
                    return 
                if arglist.debug and num_processed > 0:
                    return

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)