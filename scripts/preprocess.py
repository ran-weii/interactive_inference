import argparse
import os
import glob
import json
import pickle
import numpy as np
from numpy.core.fromnumeric import trace
import pandas as pd
import matplotlib.pyplot as plt
from src.data.kalman_filter import BatchKalmanFilter
from src.data.utils import derive_acc, normalize_pos
from src.data.utils import dist_two_points, closest_point_on_line
from src.data.utils import is_above_line, get_cardinal_direction

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

def load_map(map_path, scenario):
    map_file = os.path.join(map_path, scenario + ".json")
    with open(map_file, "r") as f:
        lane_dict = json.load(f)
    
    # remove unlabeled lanes
    lane_dict = {
        k: v for (k, v) in lane_dict.items() if v["label"] is not None
    }
    
    # dict to df
    df_lanelet = []
    for i, (lane_id, lane_val) in enumerate(lane_dict.items()):
        label = lane_val["label"]
        num_ways = lane_val["len"]
        way_dict = lane_val["way_dict"]
        
        df_way = []
        for j, (way_id, way_val) in enumerate(way_dict.items()):
            way_val["way_id"] = way_id
            df_way.append(way_val)
            
        df_way = pd.DataFrame(df_way)
        df_way["lane_id"] = lane_id
        df_way["lane_label"] = [label for i in range(len(df_way))]
        df_way["num_ways"] = num_ways
        
        df_lanelet.append(df_way)
        
    df_lanelet = pd.concat(df_lanelet, axis=0).reset_index(drop=True)
    return df_lanelet
    
def min_dist_to_way(x, y, heading, way_x, way_y):
    """ Find the min absolute distance from the vehicle to 
    every line segment in a way

    Args:
        x (float): vehicle x coor
        y (float): vehicle y coor
        heading (float): vehicle heading in radians
        way_x (list): list of x coor of the way
        way_y (list): list of y coor of the way
        
    Returns:
        min_dist (float): signed minimum distance from the vehile to the way
            if right lane: sign=1, if left lane: sign=-1
    """    
    dists = []
    for i in range(len(way_x) - 1):
        a, b = closest_point_on_line(x, y, way_x[i:i+2], way_y[i:i+2])
        card = get_cardinal_direction(x, y, heading, a, b)
        sign = 1 if card < 0 and card > -np.pi else -1
        dist = dist_two_points(x, y, a, b)
        dists.append(sign * dist)
    
    dists = np.array(dists)
    sort_id = np.argsort(dists ** 2)
    min_dist = dists[sort_id[0]]
    return min_dist

def assign_lane_pos(x, y, heading, df_lanelet):
    """ Assign lane position and labels to vehicle

    Args:
        x (float): vehicle x coor
        y (float): vehicle y coor
        heading (float): vehicle heading in radians
        df_lanelet (pd.dataframe): lanelet dataframe, each row is a way

    Returns:
        df_lane_assigned (pd.series): left and right lane attribute dataframe
    """
    df_lanelet = df_lanelet.copy()
    
    # calculate distance between vehicle and all ways
    min_dists = np.zeros((len(df_lanelet)))
    for i in range(len(df_lanelet)):
        min_dists[i] = min_dist_to_way(
            x, y, heading, df_lanelet.iloc[i]["x"], df_lanelet.iloc[i]["y"]
        )
    
    df_lanelet["min_dist"] = min_dists
    df_lanelet["abs_min_dist"] = np.abs(min_dists)
    df_lanelet = df_lanelet.sort_values("abs_min_dist").reset_index(drop=True)
    
    # find closest ways above and below
    df_left = df_lanelet.loc[df_lanelet["min_dist"] > 0].\
        sort_values("abs_min_dist").head(1).reset_index(drop=True)
    df_right = df_lanelet.loc[df_lanelet["min_dist"] <= 0].\
        sort_values("abs_min_dist").head(1).reset_index(drop=True)
    
    label_left = df_left["lane_label"].iloc[0]
    label_right = df_right["lane_label"].iloc[0]
    
    # determine merging lane labels
    label1 = label_left[0]
    label2 = label_right[0]
    if abs(label_left[0] - label_right[0]) > 1:
        if len(label_left) > 1:
            label1 = label_left[1]
            label2 = label_right[0]
        elif len(label_right) > 1:
            label1 = label_left[0]
            label2 = label_right[1]
    
    df_lane_assigned = pd.Series({
        "lane_left_label": label1,
        "lane_left_type": df_left["type"].iloc[0],
        "lane_left_subtype": df_left["subtype"].iloc[0],
        "lane_left_way_id": df_left["way_id"].iloc[0],
        "lane_left_min_dist": df_left["min_dist"].iloc[0],
        "lane_right_label": label2,
        "lane_right_type": df_right["type"].iloc[0],
        "lane_right_subtype": df_right["subtype"].iloc[0],
        "lane_right_way_id": df_right["way_id"].iloc[0],
        "lane_right_min_dist": df_right["min_dist"].iloc[0],
    })
    return df_lane_assigned

def find_neighbor_vehicles(df_ego, df_track, r):
    """ Find all vehicle in adjacent lane within a radius

    Args:
        df_ego (pd.series): ego vehicle series
        df_track (pd.dataframe): entire track dataframe
        r (float): scan radius centered on ego vehicle. 
            Vehicles outside of the radus will be removed
            
    Returns:
        dict_frame (list): list of observation dict. Each element in list is a vehicle.
            The ego vehicle is the first element. 
    """ 
    frame_id = df_ego["frame_id"]
    df_frame = df_track.loc[df_track["frame_id"] == frame_id]
    
    df_frame["dist_to_ego"] = np.sqrt(
        (df_frame["x"] - df_ego["x"]) ** 2 + (df_frame["y"] - df_ego["y"]) ** 2
    )
    df_frame = df_frame.loc[df_frame["dist_to_ego"] <= r]
    
    # compare avg lanes and remove non neighbors
    df_ego["avg_lane"] = 0.5 * (df_ego["lane_left_label"] + df_ego["lane_right_label"])
    df_frame["avg_lane"] = 0.5 * (df_frame["lane_left_label"] + df_frame["lane_right_label"])
    df_frame = df_frame.loc[np.abs(df_frame["avg_lane"] - df_ego["avg_lane"]) <= 1]
    df_frame = df_frame.sort_values(by="dist_to_ego").reset_index(drop=True)
    
    # find neighbor cardianl directions
    df_frame["card"] = [
        get_cardinal_direction(
            df_ego["x"], df_ego["y"], df_ego["psi_rad"], 
            df_frame.iloc[i]["x"], df_frame.iloc[i]["y"]
        ) for i in range(len(df_frame))
    ]
    df_frame["is_ego"] = df_frame["dist_to_ego"] == 0
    df_frame["is_lead"] = (df_frame["card"] < np.pi/2) * (df_frame["card"] > -np.pi/2)
    df_frame["is_left"] = (df_frame["card"] > 0) * (df_frame["card"] < np.pi)
    
    dict_frame = df_frame.to_dict(orient="records")
    return dict_frame

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
            
            df_lanelet = load_map(arglist.map_path, scenario)
            
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
                    lambda x: assign_lane_pos(
                        x["x"], x["y"], x["psi_rad"], df_lanelet
                    ), axis=1
                )
                df_track = pd.concat([df_track, df_lane_assigned], axis=1)
                
                df_ego = df_track.loc[df_track["track_id"] == 2].iloc[0]
                
                find_neighbor_vehicles(df_ego, df_track, r=50)
                
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