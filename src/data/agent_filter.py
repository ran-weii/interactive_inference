import numpy as np
import pandas as pd
from .utils import closest_point_on_line, get_cardinal_direction, dist_two_points

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

def get_lane_pos(x, y, heading, df_lanelet):
    """ Assign lane position and labels to vehicle

    Args:
        x (float): vehicle x coor
        y (float): vehicle y coor
        heading (float): vehicle heading in radians
        df_lanelet (pd.dataframe): lanelet dataframe, each row is a way

    Returns:
        df_lane_assigned (pd.series): left and right lane attribute dataframe
    """
    assert all(
        v in df_lanelet.columns for v in 
            ["x", "y", "lane_label", "type", "subtype", "way_id"]
    )
    
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
        "lane_label_diff": label1 - label2,
        "lane_label_avg": 0.5 * (label1 + label2)
    })
    return df_lane_assigned

def get_neighbor_vehicles(df_ego, df_frame, max_dist):
    """ Find all vehicle in adjacent lane within a radius

    Args:
        df_ego (pd.series): pd series of ego vehicle 
        df_frame (pd.dataframe): dataframe of all vehicels in the same frame
        max_dist (float): max dist to ego to be considered neighbors
            
    Returns:
        df_neighbors (pd.dataframe): dataframe of neighbor vehicles.
            The ego vehicle is the first row. 
    """ 
    assert all(v in df_ego.index for v in ["x", "y", "psi_rad"])
    assert all(v in df_frame.columns for v in ["x", "y", "psi_rad"])
    assert df_ego["frame_id"] == df_frame["frame_id"].iloc[0]
    assert len(df_frame["frame_id"].unique()) == 1
    
    df_neighbors = df_frame.copy()
    
    df_neighbors["dist_to_ego"] = np.sqrt(
        (df_neighbors["x"] - df_ego["x"]) ** 2 + (df_neighbors["y"] - df_ego["y"]) ** 2
    )
    df_neighbors = df_neighbors.loc[df_neighbors["dist_to_ego"] <= max_dist]
    
    # compare avg lanes and remove non neighbors
    df_neighbors = df_neighbors.loc[np.abs(df_ego["lane_label_avg"] - df_neighbors["lane_label_avg"]) <= 1]
    df_neighbors = df_neighbors.sort_values(by="dist_to_ego").reset_index(drop=True)
    
    # find neighbor cardianl directions
    df_neighbors["card"] = [
        get_cardinal_direction(
            df_ego["x"], df_ego["y"], df_ego["psi_rad"], 
            df_neighbors.iloc[i]["x"], df_neighbors.iloc[i]["y"]
        ) for i in range(len(df_neighbors))
    ]
    df_neighbors["is_ego"] = df_neighbors["dist_to_ego"] == 0
    df_neighbors["is_lead"] = (df_neighbors["card"] < np.pi/2) * (df_neighbors["card"] > -np.pi/2)
    df_neighbors["is_lead"] *= df_neighbors["lane_label_avg"] == df_ego["lane_label_avg"]
    
    df_neighbors["is_left"] = (df_neighbors["card"] > 0) * (df_neighbors["card"] < np.pi)
    df_neighbors["is_left"] *= df_neighbors["lane_label_avg"] != df_ego["lane_label_avg"]
    return df_neighbors

def get_car_following_episode(df_track_processed):
    """
    Args:
        df_track_processed (pd.dataframe): processed track dataframe
        
    Returns:
        episode_id (np.array): car following episode id, -1 for not car following
    """
    assert all(v in df_track_processed.columns for v in ["track_id", "lead_track_id"])
    
    df = df_track_processed.copy()
    df["eps_change"] = np.any(np.stack([
        df.groupby("track_id")["lead_track_id"].fillna(-1).diff() != 0,
        df.groupby("track_id")["lane_label_avg"].fillna(-1).diff() != 0,
    ]), axis=0)
    df["episode"] = df.groupby("track_id")["eps_change"].cumsum()
    df["episode"].loc[df["lead_track_id"].isna()] = -1
    episode_id = df["episode"].values
    return episode_id