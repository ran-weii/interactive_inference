import argparse
import os
import numpy as np
import pandas as pd

from src.data.data_filter import (
    filter_car_follow_eps, filter_tail_merging, filter_lane)

# set pandas display option
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

data_path = "../interaction-dataset-master"
scenario = "DR_CHN_Merging_ZS"
filename = "vehicle_tracks_007.csv"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../interaction-dataset-master"
    )
    parser.add_argument("--scenario", type=str, default="DR_CHN_Merging_ZS")
    arglist = parser.parse_args()
    return arglist

def load_data():
    df_processed = pd.read_csv(
        os.path.join(data_path, "processed_trackfiles", scenario, filename)
    )
    df_track = pd.read_csv(
        os.path.join(data_path, "recorded_trackfiles", scenario, filename)
    )
    df_track = df_track.merge(df_processed, on=["track_id", "frame_id"])
    df_track["psi_rad"] = np.clip(df_track["psi_rad"], -np.pi, np.pi)
    return df_track

def test_data(arglist):
    """ data sanity and quality test """
    file_path = os.path.join(
        arglist.data_path, 
        "recorded_trackfiles", 
        arglist.scenario, 
        "vehicle_tracks_000.csv"
    )
    df_track = pd.read_csv(file_path)
    dt = df_track.groupby("track_id")["timestamp_ms"].diff().mean() / 1000
    
    # get distance traveled
    df_track_head = df_track.groupby(["track_id"]).head(1).\
        reset_index(drop=True).drop(columns=["agent_type"])
    df_track_tail = df_track.groupby(["track_id"]).tail(1).\
        reset_index(drop=True).drop(columns=["agent_type"])
    df_track_diff = df_track_tail - df_track_head
    
    print(10 * "=" + " duration summary " + 10 * "=")
    print(df_track_diff.describe().drop(
        columns=["frame_id", "vx", "vy", "psi_rad", "length", "width"]
    ).round(4))
    
    # differentiate pos
    f_grad = lambda x: pd.DataFrame(np.gradient(x, dt), index=x.index) 
    df_track["x_grad"] = df_track.groupby("track_id")["x"].apply(f_grad)
    df_track["y_grad"] = df_track.groupby("track_id")["y"].apply(f_grad)
    df_track["vx_error"] = np.abs(df_track["x_grad"] - df_track["vx"])
    df_track["vy_error"] = np.abs(df_track["y_grad"] - df_track["vy"])
    
    # get yaw angle
    yaw = np.arctan2(df_track["vy"], df_track["vx"])
    df_track["yaw_error"] = np.abs(yaw - df_track["psi_rad"])
    
    # note: yaw error with track 518 with vx=0, vy=0, psi_rad=-3.103
    
    print(10 * "=" + " velocity grad & yaw summary " + 10 * "=")
    print(df_track[
        ["x_grad", "vx", "vx_error", "y_grad", "vy", "vy_error", "yaw_error"]
    ].describe().drop(index="count").round(4))    
    
    # avg speed test
    avg_vx = np.abs(df_track_diff["x"] / df_track_diff["timestamp_ms"] * 1000)
    avg_vy = np.abs(df_track_diff["y"] / df_track_diff["timestamp_ms"] * 1000)
    
    assert np.all(avg_vx < 50)
    assert np.all(avg_vy < 10)
    print("test data: avg speed test passed")
    
    # velocity test
    assert np.all(df_track["vx"] < 50)
    assert np.all(df_track["vy"] < 10)
    print("test data: instantaneous speed test passed")
    
    # pos grad error test
    max_x_error_percent = df_track["vx_error"].mean() / df_track["vx"].abs().max()
    max_y_error_percent = df_track["vy_error"].mean() / df_track["vy"].abs().max()
    
    assert max_x_error_percent < 0.01
    assert max_y_error_percent < 0.01
    print("test data: pos grad error test passed")

def test_filter_lane():
    df_track = load_data()
    
    filter_lane_id = [1, 6]
    df_track = filter_lane(df_track, lane_ids=filter_lane_id)
    unique_lanes = df_track.loc[df_track["car_follow_eps"] != -1]["lane_id"].unique()
    assert not set(filter_lane_id).issubset(set(unique_lanes))
    
    min_eps_len = 50
    df_track = filter_car_follow_eps(df_track, min_eps_len)
    unique_lanes = df_track.loc[df_track["eps_id"] != -1]["lane_id"].unique()
    assert not set(filter_lane_id).issubset(set(unique_lanes))
    print("test_filter_lane passed")

def test_filter_tail_merging():
    from src.data.geometry import wrap_angles
    
    df_track = load_data()
    
    max_psi_error = 0.05
    df_track = filter_tail_merging(df_track, max_psi_error=max_psi_error, min_bound_dist=1)
    df_valid = df_track.loc[df_track["car_follow_eps"] != -1]
    df_valid = df_valid.assign(psi_error=wrap_angles(df_valid["psi_rad"] - df_valid["psi_tan"]))
    psi_error = df_valid.groupby(["track_id", "car_follow_eps"]).tail(1)["psi_error"]
    assert np.isclose(psi_error.abs().max(), max_psi_error, atol=0.02)
    
    min_eps_len = 50
    df_track = filter_car_follow_eps(df_track, min_eps_len)
    df_valid = df_track.loc[df_track["eps_id"] != -1]
    df_valid = df_valid.assign(psi_error=wrap_angles(df_valid["psi_rad"] - df_valid["psi_tan"]))
    psi_error = df_valid.groupby(["eps_id"]).tail(1)["psi_error"]
    assert np.isclose(psi_error.abs().max(), max_psi_error, atol=0.02)
    print("test_filter_tail_merging passed")

if __name__ == "__main__":
    arglist = parse_args()
    # test_data(arglist)
    # test_filter_lane()
    # test_filter_tail_merging()