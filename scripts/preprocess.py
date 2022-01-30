import argparse
import os
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data.lanelet import load_lanelet_df
from src.data.kalman_filter import BatchKalmanFilter
from src.data.agent_filter import get_lane_pos, get_neighbor_vehicles
from src.data.utils import derive_acc, normalize_pos

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
    parser.add_argument(
        "--map_path", type=str, default="../exp/lanelet"
    )
    parser.add_argument("--scenario", type=str, default="Merging")
    parser.add_argument("--max_dist", type=float, default=50., help="max dist to be considereed neighbor")
    parser.add_argument("--debug", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def smooth_one_track(df_one_track, kf):
    """ Apply kalman filter to one track """
    df_ = df_one_track.copy()
    df_ = derive_acc(df_, kf.dt)
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
    
def get_lead_vehicle_id(df_ego, df_track, max_dist):
    """ 
    Args:
        df_ego (pd.series): pd series of ego vehicle 
        df_track (pd.dataframe): dataframe of all vehilce in the track record
        max_dist (float): max dist to ego to be considered neighbors

    Returns:
        out (pd.series): lead vehicle track id
    """
    df_frame = df_track.loc[df_track["frame_id"] == df_ego["frame_id"]]
    df_neighbor = get_neighbor_vehicles(df_ego, df_frame, max_dist)
    df_neighbor = df_neighbor.loc[df_neighbor["is_ego"] == False]
    df_lead = df_neighbor.loc[df_neighbor["is_lead"]].sort_values(by="dist_to_ego")
    id_lead = df_lead.head(1)["track_id"].iloc[0] if len(df_lead) > 0 else None
    out = pd.Series({"lead_track_id": id_lead})
    return out

def process(df_track, df_lanelet, max_dist, kf_filter=None):
    """ Preprocess pipeline: 
    get ego lane, get lead vehicle id, filter vehicle dynamics
    """
    df_track["psi_rad"] = np.clip(df_track["psi_rad"], -np.pi, np.pi)
    
    print("identifying ego lane")
    df_ego_lane = df_track.progress_apply(
        lambda x: get_lane_pos(
            x["x"], x["y"], x["psi_rad"], df_lanelet
        ), axis=1
    )
    df_track = pd.concat([df_track, df_ego_lane], axis=1)
    
    print("identifying lead vehicle")
    df_lead_vehicle = df_track.progress_apply(
        lambda x: get_lead_vehicle_id(x, df_track, max_dist), axis=1
    ).reset_index(drop=True)
    
    df_processed = pd.concat([df_ego_lane, df_lead_vehicle], axis=1)
    
    if kf_filter is not None:
        df_track_smooth = df_track.groupby("track_id").apply(
                lambda x: smooth_one_track(x, kf_filter)
        ).reset_index().drop(columns=["level_1", "track_id"])
        df_processed = pd.concat([df_processed, df_track_smooth], axis=1)
    return df_processed

def main(arglist):
    # load kalman filter 
    with open(os.path.join(arglist.kf_path, "model.p"), "rb") as f:
        kf = pickle.load(f)
        kf.dt = 0.1
    
    # load data
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
            
            df_lanelet = load_lanelet_df(
                os.path.join(arglist.map_path, scenario + ".json")
            )
            
            for j, track_path in enumerate(track_paths):
                filename = os.path.basename(track_path)
                record_id = filename.replace(".csv", "").split("_")[-1]
                print(f"track_file: {filename}")
                
                df_track = pd.read_csv(track_path)
                
                if arglist.debug:
                    # df_track = df_track.loc[df_track["track_id"].isin([1, 2, 3])].iloc[:100]
                    df_track = df_track.iloc[:10000]
                
                """ test process """
                df_track = df_track.loc[df_track["frame_id"].isin([1, 2])].reset_index(drop=True)
                df_processed = process(
                    df_track, df_lanelet, arglist.max_dist, kf_filter=None
                )
                df_processed.insert(0, "scenario", scenario)
                df_processed.insert(1, "record_id", record_id)
                print(df_processed)
                
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