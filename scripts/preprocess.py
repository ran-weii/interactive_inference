import argparse
import os
import glob
import json
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
    ).reset_index().drop(columns="level_1")
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
    df_lane = []
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
        
        df_lane.append(df_way)
        
    df_lane = pd.concat(df_lane, axis=0).reset_index(drop=True)
    return df_lane

def dist_two_points(x1, y1, x2, y2):
    """ Two point distance formula """
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def closest_point_on_line(x, y, x_line, y_line):
    """ Find the closest point (a, b) to an external point (x, y) on a line segement
    
    Args:
        x (float): x coor of target point
        y (float): y coor of target point
        x_line (list): x coors of target line
        y_line (list): y coors of target line
        
    Returns:
        a (float): x coor of closest point
        b (float): y coor of closest point
    """
    [x1, x2] = x_line
    [y1, y2] = y_line
    
    px = x2 - x1
    py = y2 - y1
    norm = px * px + py * py
    
    # fraction of tangent point on line
    u = ((x - x1) * px + (y - y1) * py) / norm
    
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    
    # closest point (a, b)
    a = x1 + u * px
    b = y1 + u * py
    return a, b

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
        min_dist (float): minimum distance from the vehile to the way
    """
    def get_dist_sign(x, y, heading, a, b):
        """ Determine if point (a, b) is above (1) or below (-1) the line 
        defined by point (x, y) and heading 
        """
        # extend x, y along heading
        heading_slope = np.tan(heading)
        intercept = y - heading_slope * x
        b_pred = heading_slope * a + intercept
        
        if b_pred > b: 
            return -1
        else: 
            return 1
    
    dists = []
    for i in range(len(way_x) - 1):
        a, b = closest_point_on_line(x, y, way_x[i:i+2], way_y[i:i+2])
        sign = get_dist_sign(x, y, heading, a, b)
        dist = dist_two_points(x, y, a, b)
        dists.append(sign * dist)
        
    dists = np.array(dists)
    sort_id = np.argsort(dists ** 2)
    min_dist = dists[sort_id[0]]
    return min_dist

""" TODO
move some functions to data/utils
implement lane distance finding x distance ahead of ego vehicle (maybe read markkula new paper?)
find all vehicles in the same frame in adjacent lanes within a radius
"""
def assign_lane_pos(x, y, heading, df_lane):
    """ Assign lane position and labels to vehicle

    Args:
        x (float): vehicle x coor
        y (float): vehicle y coor
        heading (float): vehicle heading in radians
        df_lane (pd.dataframe): lanelet dataframe, each row is a way

    Returns:
        df_out (pd.dataframe): left and right lane attribute dataframe
    """
    df_lane = df_lane.copy()
    
    # calculate distance between vehicle and all ways
    min_dists = np.zeros((len(df_lane)))
    for i in range(len(df_lane)):
        min_dists[i] = min_dist_to_way(
            x, y, heading, df_lane.iloc[i]["x"], df_lane.iloc[i]["y"]
        )
    
    df_lane["min_dist"] = min_dists
    df_lane["abs_min_dist"] = np.abs(min_dists)
    
    # find closest ways above and below
    df_above = df_lane.loc[df_lane["min_dist"] > 0].\
        sort_values("abs_min_dist").head(1).reset_index(drop=True)
    df_below = df_lane.loc[df_lane["min_dist"] <= 0].\
        sort_values("abs_min_dist").head(1).reset_index(drop=True)
    
    label_above = df_above["lane_label"].iloc[0]
    label_below = df_below["lane_label"].iloc[0]
    
    # determine merging lane labels
    label1 = label_above[0]
    label2 = label_below[0]
    if abs(label_above[0] - label_below[0]) > 1:
        if len(label_above) > 1:
            label1 = label_above[1]
            label2 = label_below[0]
        elif len(label_below) > 1:
            label1 = label_above[0]
            label2 = label_below[1]
    
    df_out = pd.DataFrame([{
        "lane_1_label": label1,
        "lane_1_type": df_above["type"].iloc[0],
        "lane_1_subtype": df_above["subtype"].iloc[0],
        "lane_1_way_id": df_above["way_id"].iloc[0],
        "lane_1_min_dist": df_above["min_dist"].iloc[0],
        "lane_2_label": label2,
        "lane_2_type": df_below["type"].iloc[0],
        "lane_2_subtype": df_below["subtype"].iloc[0],
        "lane_2_way_id": df_below["way_id"].iloc[0],
        "lane_2_min_dist": df_below["min_dist"].iloc[0],
    }])
    
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
            
            df_lane = load_map(arglist.map_path, scenario)
            
            for j, track_path in enumerate(track_paths):
                df_track = pd.read_csv(track_path)
                
                if arglist.debug:
                    df_track = df_track.loc[df_track["track_id"].isin([1, 2, 3])]
                
                print(df_track.columns)
                # test lane assignment
                # x = df_track.iloc[0]["x"]
                # y = df_track.iloc[0]["y"]
                # heading = df_track.iloc[0]["psi_rad"]
                # k = np.tan(heading)
                
                x, y = 1114, 960
                # x, y = 1055, 960
                # x, y = 1074, 962
                # x, y = 1094, 939
                # x, y = 1109, 940
                # x, y = 1126, 947
                k = np.tan(np.deg2rad(30))
                heading = np.arctan2(k, 1)
                
                assign_lane_pos(x, y, heading, df_lane)
                
                """ debug calc min dist """
                import xml.etree.ElementTree as xml
                from src.data.lanelet import find_all_points, plot_all_lanes
                file_path = os.path.join(
                    arglist.data_path, 
                    "maps", 
                    scenario + ".osm"
                )
                
                e = xml.parse(file_path).getroot()
                
                point_dict = find_all_points(e, lat_origin=0, lon_origin=0)
                map_file = os.path.join(arglist.map_path, scenario + ".json")
                with open(map_file, "r") as f:
                    lane_dict = json.load(f)
                fig, ax, _ = plot_all_lanes(point_dict, lane_dict)
                ax.plot([x, x + 1], [y, y + k], "o")
                plt.show()
                
                # df_track = derive_acc(df_track, dt)
                # df_track = kalman_smooth(df_track, kf)
                # print(df_track)
                
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