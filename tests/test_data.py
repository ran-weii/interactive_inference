import argparse
import os
import numpy as np
import pandas as pd

# set pandas display option
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../interaction-dataset-master"
    )
    parser.add_argument("--scenario", type=str, default="DR_CHN_Merging_ZS")
    arglist = parser.parse_args()
    return arglist

def test_data(arglist):
    """ data sanity and quality test """
    file_path = os.path.join(
        arglist.data_path, 
        "recorded_trackfiles", 
        arglist.scenario, 
        "vehicle_tracks_000.csv"
    )
    df_track = pd.read_csv(file_path)
    dt = df_track.groupby("track_id")["timestamp_ms"].diff().mean()
    
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
    f_grad = lambda x: pd.DataFrame(np.gradient(x, dt), index=x.index) * 1000
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

if __name__ == "__main__":
    arglist = parse_args()
    test_data(arglist)