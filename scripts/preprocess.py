import argparse
import os
import pickle
import numpy as np
import pandas as pd
import swifter
from tqdm.auto import tqdm

from src.map_api.lanelet import MapReader
from src.data.kalman_filter import BatchKalmanFilter
from src.data.data_filter import (
    get_trajectory_segment_id, filter_segment_by_length,
    classify_tail_merging)
from src.data.geometry import coord_transformation
from src.map_api.frenet import Trajectory
# from src.simulation.observers import FEATURE_SET, Observer

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
    parser.add_argument("--task", type=str, choices=["kalman_filter", "features", "train_labels"])
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
    parser.add_argument("--num_cores", type=int, default=10, 
                        help="number of parallel processing cores, default=10")
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

def compute_features(df, map_data, dt, min_seg_len, parallel):
    """ Return dataframe with fields defined in FEATURE_SET """
    from src.simulation.sensors import STATE_KEYS
    from src.simulation.sensors import EgoSensor, LeadVehicleSensor, LidarSensor
    from src.simulation.observers import Observer
    
    # compute actions
    df = derive_acc(df, dt)
    df = df.assign(ax=df["vx_grad"].values)
    df = df.assign(ay=df["vy_grad"].values)

    ego_sensor = EgoSensor(map_data)
    lv_sensor = LeadVehicleSensor(map_data)
    lidar_sensor = LidarSensor()
    feature_set = ego_sensor.feature_names + lv_sensor.feature_names + lidar_sensor.feature_names

    def f_compute_features_episode(df_ego, **kwargs):
        ego_sensor = EgoSensor(map_data)
        lv_sensor = LeadVehicleSensor(map_data, track_lv=False)
        lidar_sensor = LidarSensor()
        observer = Observer(
            map_data, [ego_sensor, lv_sensor, lidar_sensor], 
            feature_set=ego_sensor.feature_names + lv_sensor.feature_names + lidar_sensor.feature_names
        )

        track_id = df_ego["track_id"]
        frame_id = df_ego["frame_id"]
        df_agents = df.loc[(df["frame_id"] == frame_id) & (df["track_id"] != track_id) ]
        ego_state = df_ego[STATE_KEYS].values.astype(np.float64)
        agent_states = df_agents[STATE_KEYS].values.astype(np.float64)

        sensor_obs = {}
        sensor_obs["EgoSensor"], _ = ego_sensor.get_obs(ego_state, agent_states)
        sensor_obs["LeadVehicleSensor"], _ = lv_sensor.get_obs(ego_state, agent_states)
        sensor_obs["LidarSensor"], _ = lidar_sensor.get_obs(ego_state, agent_states)
        obs = observer.observe(sensor_obs).numpy()
        return obs
    
    features = parallel_apply(
        df, f_compute_features_episode, parallel, axis=1, desc="computing features"
    )
    df_features = pd.DataFrame(np.vstack(features.values), columns=feature_set)
    df_features = df[["track_id", "frame_id"]].merge(
        df_features, how="left", left_index=True, right_index=True
    )
    
    # compute drive segment id based on features
    seg_id = get_trajectory_segment_id(df_features, ["track_id", "ego_lane_id", "lv_track_id"])
    seg_id, seg_len = filter_segment_by_length(seg_id, min_seg_len)
    
    df_features.insert(2, "seg_id", seg_id)
    df_features.insert(3, "seg_len", seg_len)

    df.insert(2, "seg_id", seg_id)
    df.insert(3, "seg_len", seg_len)
    df.insert(4, "ego_lane_id", df_features["ego_lane_id"])

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
            
            ref_lane_id = x["ego_lane_id"].values[0].astype(int)
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
    
    df_traj_features = df.groupby("seg_id").progress_apply(
        compute_trajectory_features
    ).reset_index(drop=True)
    df_features = df_features.merge(
        df_traj_features, on=["track_id", "frame_id"], how="left"
    )
    
    # append actions 
    df_features = df_features.assign(ax=df["vx_grad"].values)
    df_features = df_features.assign(ay=df["vy_grad"].values)
    return df_features

def get_train_labels(df, train_ratio, min_seg_len, invalid_lane_ids):
    """ Return dataframe with fields 
    ["track_id", "frame_id", "is_tail", "is_tail_merging", "eps_id", "eps_len", "is_train"] 
    """
    df_follow = df.loc[(df["seg_id"].isna() == False) & (df["lv_track_id"] != 0)]
    
    feature_names = ["ego_d", "ego_dd", "ddd", "ego_psi_error_r"]

    # classify tail merging using logistic regression
    is_tail, is_tail_merging, cmat = classify_tail_merging(
        df_follow, feature_names, tail=True, p_tail=0.3, max_d=1.2, class_weight={0:1, 1:2}
    )
    df_follow = df_follow.assign(is_tail=is_tail)
    df_follow = df_follow.assign(is_tail_merging=is_tail_merging)

    # classify head merging using logistic regression
    is_head, is_head_merging, cmat = classify_tail_merging(
        df_follow, feature_names, tail=False, p_tail=0.3, max_d=1.2, class_weight={0:1, 1:2}
    )
    df_follow = df_follow.assign(is_head=is_head)
    df_follow = df_follow.assign(is_head_merging=is_head_merging)

    # reassign segment labels
    df_follow["seg_id"] = get_trajectory_segment_id(
        df_follow, ["track_id", "seg_id", "is_tail_merging"]
    )
    
    # remove tail merging from seg_id and seg_len
    new_seg_id = df_follow["seg_id"].values.copy().astype(float)
    new_seg_id[df_follow["is_tail_merging"] == 1] = np.nan
    new_seg_id[df_follow["is_head_merging"] == 1] = np.nan
    new_seg_id[df_follow["ego_lane_id"].isin(invalid_lane_ids)] = np.nan
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
        "is_head", "is_head_merging", "eps_id", "eps_len", "is_train"
    ]
    df_train_labels = df_follow[out_fields]
    df = df.merge(df_train_labels, on=["track_id", "frame_id"], how="left")
    df_train_labels = df[out_fields]
    return df_train_labels

def main(arglist):
    np.random.seed(arglist.seed)
    swifter.set_defaults(
        npartitions=arglist.num_cores,
        dask_threshold=1,
        scheduler="processes",
        progress_bar=True,
        progress_bar_desc=None,
        allow_dask_on_strings=False,
        force_parallel=arglist.parallel,
    )
    
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
    
    elif arglist.task == "features":
        map_data = MapReader(cell_len=arglist.cell_len)
        map_data.parse(map_path, verbose=True)
        
        # load neighbor df
        # kf_path = os.path.join(
        #     data_path, "kalman_filter", arglist.scenario, arglist.filename
        # )
        # df_kf = pd.read_csv(kf_path)
        # df = df.merge(df_kf, on=["track_id", "frame_id"], how="right")
        if arglist.debug:
            df = df.loc[df["track_id"] <= 30]
        
        dt = 0.1
        df = compute_features(df, map_data, dt, arglist.min_seg_len, arglist.parallel)
    
    elif arglist.task == "train_labels":
        neighbor_path = os.path.join(
            data_path, "neighbors", arglist.scenario, arglist.filename
        )
        feature_path = os.path.join(
            data_path, "features", arglist.scenario, arglist.filename
        )
        df_feature = pd.read_csv(feature_path)
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