import argparse
import os
import pickle
import numpy as np
import pandas as pd
import swifter
from tqdm.auto import tqdm

from src.map_api.lanelet import MapReader
from src.data.kalman_filter import BatchKalmanFilter
from src.data.agent_filter import get_neighbor_vehicle_ids
from src.data.data_filter import (
    get_trajectory_segment_id, filter_segment_by_length,
    classify_tail_merging)
from src.data.geometry import coord_transformation
from src.map_api.frenet import Trajectory
from src.simulation.observers import FEATURE_SET, Observer

# to use apply progress bar
tqdm.pandas()

# set pandas display option
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../interaction-dataset-master")
    parser.add_argument("--kf_path", type=str, default="../exp/kalman_filter")
    parser.add_argument("--scenario", type=str, default="DR_CHN_Merging_ZS")
    parser.add_argument("--filename", type=str, default="vehicle_tracks_007.csv", 
        help="track filename. default=vehicle_tracks_007.csv")
    parser.add_argument("--task", type=str, choices=["kalman_filter", "neighbors", "features", "train_labels"])
    parser.add_argument("--cell_len", type=float, default=10, 
                        help="length of drivable cells, default=10")
    parser.add_argument("--max_dist", type=float, default=50., 
                        help="max dist to be considered neighbor, default=50")
    parser.add_argument("--min_seg_len", type=int, default=50, 
                        help="minimum trajectory segment length, default=50")
    parser.add_argument("--invalid_lane_ids", type=str, default="1,6", 
                        help="invalid lane ids to be filtered. str no space separated by comma default=1,6")
    parser.add_argument("--train_ratio", type=float, default=0.7, 
                        help="train ratio, default=0.7")
    parser.add_argument("--parallel", type=bool_, default=True, 
                        help="parallel apply, default=True")
    parser.add_argument("--seed", type=int, default=0)
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

def parallel_apply(df, f, parallel, axis=1, desc=""):
    if parallel:
        return df.swifter.progress_bar(True, desc=desc).apply(f, axis=axis)
    else:
        print(desc)
        return df.progress_apply(f, axis=axis)

def run_kalman_filter(df, dt, kf_filter):
    """ Return dataframe with fields 
    ["track_id", "frame_id", "ax", "ay", "ax_ego", "ay_ego", 
    "x_kf", "y_kf", "vx_kf", "vy_kf", "ax_kf", "ay_kf"]
    """
    df = derive_acc(df, dt)
    df_track_smooth = df.groupby("track_id").progress_apply(
            lambda x: smooth_one_track(x, kf_filter)
    ).reset_index().drop(columns=["level_1"])
    df_track_smooth.insert(1, "frame_id", df["frame_id"])
    df_track_smooth.insert(2, "ax", df["vx_grad"])
    df_track_smooth.insert(3, "ay", df["vy_grad"])
    
    # compute ego centric accelerations
    ax_ego, ay_ego = coord_transformation(
        df_track_smooth["ax"].values,
        df_track_smooth["ay"].values,
        None, None, theta=df["psi_rad"].values
    )
    df_track_smooth.insert(4, "ax_ego", ax_ego)
    df_track_smooth.insert(5, "ay_ego", ay_ego)
    return df_track_smooth

def find_neighbors(df, map_data, max_dist, parallel):
    """ Return dataframe with fields
    ["track_id", "frame_id", "lane_id", 
    "lead_track_id", "follow_track_id", 
    "left_track_id", "right_track_id",
    "left_lead_track_id", "right_lead_track_id", 
    "left_follow_track_id", "right_follow_track_id"]
    """
    df = df.assign(psi_rad=np.clip(df["psi_rad"], -np.pi, np.pi))
    
    # match lanes
    f_match_lane = lambda x: map_data.match_lane(x["x"], x["y"])
    lane_ids = parallel_apply(df, f_match_lane, parallel, axis=1, desc="match lane")
    df = df.assign(lane_id=lane_ids)
    
    # find neighbors
    f_neighbor_vehicle = lambda x: get_neighbor_vehicle_ids(x, df, map_data, max_dist)
    df_neighbors = parallel_apply(
        df, f_neighbor_vehicle, parallel, axis=1, desc="identify lead vehicle"
    )
    df_neighbors.insert(0, "track_id", df["track_id"])
    df_neighbors.insert(1, "frame_id", df["frame_id"])
    return df_neighbors

def compute_features(df, map_data, min_seg_len, parallel):
    """ Return dataframe with fields defined in FEATURE_SET """
    seg_id = get_trajectory_segment_id(df)
    seg_id, seg_len = filter_segment_by_length(seg_id, min_seg_len)
    df = df.assign(seg_id=seg_id)
    df = df.assign(seg_len=seg_len)
    
    # concat with neighbors
    df_ego = df.copy()
    df_agent = df.copy()
    df_agent.columns = [c + "_agent" for c in df_agent.columns]
    df_joint = df_ego.merge(
        df_agent, left_on=["frame_id", "lead_track_id"], 
        right_on=["frame_id_agent", "track_id_agent"], how="left"
    )
    
    feature_set = FEATURE_SET["ego"] + FEATURE_SET["relative"]
    print(f"features set: {feature_set}")
    def f_compute_features(x, **kwargs): 
        """ Compute features for a time step """
        observer = Observer(
            map_data, ego_features=FEATURE_SET["ego"],
            relative_features=FEATURE_SET["relative"]
        )
        observer.reset()
        
        if np.isnan(x["lead_track_id"]):
            obs = np.nan * np.ones(len(feature_set))
        else:
            ego_state = x[["x", "y", "vx", "vy", "psi_rad"]].values
            agent_state = x[["x_agent", "y_agent", "vx_agent", "vy_agent", "psi_rad_agent"]].values.reshape(1, -1)
            state = {"ego": ego_state, "agents": agent_state}
            obs = observer.observe(state).numpy().flatten()
        return pd.Series(obs)
    
    df_features = parallel_apply(
        df_joint, f_compute_features, parallel, axis=1, desc="computing features"
    )
    df_features.columns = feature_set
    df_features.insert(0, "track_id", df["track_id"])
    df_features.insert(1, "frame_id", df["frame_id"])
    df_features.insert(2, "seg_id", df_joint["seg_id"])
    df_features.insert(3, "seg_len", df_joint["seg_len"])
    
    def compute_trajectory_features(x):
        """ Compute features that require the entire trajectory """
        if x["seg_id"].values[0] == -1:
            features = np.nan * np.ones((len(x), 3))
        else:
            trajectory = Trajectory(
                x["x"].values, x["y"].values, 
                x["vy"].values, x["vy"].values, 
                x["ax"].values, x["ay"].values, x["psi_rad"].values
            )
            
            ref_lane_id = x["lane_id"].values[0]
            ref_path = map_data.lanes[ref_lane_id].centerline.frenet_path
            trajectory.get_frenet_trajectory(ref_path)
            
            dds = trajectory.s_condition[:, 2]
            ddd = trajectory.d_condition[:, 2]
            kappa_ego = trajectory.kappa
            norm_ego = trajectory.norm
            a = trajectory.a # signed frenet acceleration
            features = np.stack([dds, ddd, kappa_ego, norm_ego, a]).T
        
        df_out = pd.DataFrame(
            features, columns=["dds", "ddd", "kappa", "norm", "a"], index=x.index
        )
        df_out = df_out.assign(track_id=x["track_id"].values)
        df_out = df_out.assign(frame_id=x["frame_id"].values)
        return df_out
    
    df_traj_features = df_joint.groupby("seg_id").progress_apply(
        compute_trajectory_features
    ).reset_index(drop=True)
    df_features = df_features.merge(
        df_traj_features, on=["track_id", "frame_id"], how="left"
    )
    return df_features

def get_train_labels(df, train_ratio, min_seg_len, invalid_lane_ids):
    """ Return dataframe with fields 
    ["track_id", "frame_id", "is_tail", "is_tail_merging", "eps_id", "eps_len", "is_train"] 
    """
    df_follow = df.loc[df["seg_id"].isna() == False]
    
    # classify tail merging using logistic regression
    is_tail, is_tail_merging, cmat = classify_tail_merging(
        df_follow, p_tail=0.3, max_d=1.2, class_weight={0:1, 1:2}
    )
    df_follow = df_follow.assign(is_tail=is_tail)
    df_follow = df_follow.assign(is_tail_merging=is_tail_merging)
    
    # remove tail merging from seg_id and seg_len
    new_seg_id = df_follow["seg_id"].values.copy()
    new_seg_id[df_follow["is_tail_merging"] == 1] = np.nan
    new_seg_id[df_follow["lane_id"].isin(invalid_lane_ids)] = np.nan
    new_seg_id, new_seg_len = filter_segment_by_length(new_seg_id, min_seg_len)
    
    df_follow = df_follow.assign(eps_id=new_seg_id)
    df_follow = df_follow.assign(eps_len=new_seg_len)
    
    # assign train id
    unique_eps_id = np.unique(new_seg_id)
    unique_eps_id = unique_eps_id[np.isnan(unique_eps_id) == False]
    unique_eps_id = np.random.permutation(unique_eps_id)
    num_eps = len(unique_eps_id)
    num_train = np.ceil(train_ratio * num_eps).astype(int)
    train_eps_id = unique_eps_id[:num_train]
    
    is_train = np.zeros(len(df_follow))
    is_train[df_follow["eps_id"].isin(train_eps_id)] = 1
    df_follow = df_follow.assign(is_train=is_train)
    
    out_fields = [
        "track_id", "frame_id", "is_tail", "is_tail_merging", 
        "eps_id", "eps_len", "is_train"
    ]
    df_train_labels = df_follow[out_fields]
    df = df.merge(df_train_labels, on=["track_id", "frame_id"], how="left")
    df_train_labels = df[out_fields]
    return df_train_labels

def main(arglist):
    np.random.seed(arglist.seed)
    
    # make save path
    data_path = os.path.join(arglist.data_path, "processed_trackfiles")
    task_path = os.path.join(data_path, arglist.task)
    save_path = os.path.join(task_path, arglist.scenario)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists(task_path):
        os.mkdir(task_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # load raw data
    map_path = os.path.join(
        arglist.data_path,
        "maps",
        f"{arglist.scenario}.osm"
    )
    file_path = os.path.join(
        arglist.data_path, 
        "recorded_trackfiles", 
        arglist.scenario, 
        arglist.filename
    )
    df = pd.read_csv(file_path)
    
    if arglist.task == "kalman_filter":
        if arglist.debug:
            df = df.loc[df["track_id"] <= 30]
    
        # load kalman filter 
        with open(os.path.join(arglist.kf_path, "model.p"), "rb") as f:
            kf = pickle.load(f)
            dt = 0.1
        
        df = run_kalman_filter(df, dt, kf)
    
    elif arglist.task == "neighbors":
        map_data = MapReader(cell_len=arglist.cell_len)
        map_data.parse(map_path, verbose=True)
        df = find_neighbors(df, map_data, arglist.max_dist, arglist.parallel)
    
    elif arglist.task == "features":
        map_data = MapReader(cell_len=arglist.cell_len)
        map_data.parse(map_path, verbose=True)
        
        # load neighbor df
        kf_path = os.path.join(
            data_path, "kalman_filter", arglist.scenario, arglist.filename
        )
        neighbor_path = os.path.join(
            data_path, "neighbors", arglist.scenario, arglist.filename
        )
        df_kf = pd.read_csv(kf_path)
        df_neighbor = pd.read_csv(neighbor_path)
        df = df.merge(df_kf, on=["track_id", "frame_id"], how="right")
        df = df.merge(df_neighbor, on=["track_id", "frame_id"], how="right")
        if arglist.debug:
            df = df.loc[df["track_id"] <= 30]
        
        df = compute_features(df, map_data, arglist.min_seg_len, arglist.parallel)
    
    elif arglist.task == "train_labels":
        neighbor_path = os.path.join(
            data_path, "neighbors", arglist.scenario, arglist.filename
        )
        feature_path = os.path.join(
            data_path, "features", arglist.scenario, arglist.filename
        )
        df_neighbor = pd.read_csv(neighbor_path)
        df_feature = pd.read_csv(feature_path)
        df = df.merge(df_neighbor, on=["track_id", "frame_id"], how="right")
        df = df.merge(df_feature, on=["track_id", "frame_id"], how="right")
        if arglist.debug:
            df = df.loc[df["track_id"] <= 30]
        
        invalid_lane_ids = [int(i) for i in arglist.invalid_lane_ids.split(",")]
        df = get_train_labels(
            df, arglist.train_ratio, arglist.min_seg_len, invalid_lane_ids
        )
    
    if arglist.save:
        df.to_csv(os.path.join(save_path, arglist.filename), index=False)
        
        print(f"processed file saved at: {save_path}/{arglist.filename}")

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)