import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from src.data.geometry import vector_projection, wrap_angles

def filter_car_follow_eps(df_track, min_eps_len):
    """ Set car_follow_eps column to -1 if episode length is less than min_eps_len
    
    Args:
        df_track (pd.dataframe): track dataframe
        min_eps_len (int): min episode length

    Returns:
        df_track (pd.dataframe): track dataframe with filtered "eps_id" and "eps_len" fields
    """
    df_track = df_track.assign(
        eps_label=df_track["scenario"] + '_' + df_track["record_id"].apply(str) + \
        "_" + df_track["track_id"].apply(str) + "_" + df_track["car_follow_eps"].apply(str)
    )
    
    df_eps_len = df_track.groupby("eps_label").size().reset_index()
    df_eps_len.columns = ["eps_label", "eps_len"]
    df_eps_len = df_eps_len.sort_values(by="eps_len").reset_index(drop=True)
    
    eps_id = np.zeros(len(df_eps_len))
    eps_id[df_eps_len["eps_len"] < min_eps_len] = -1
    eps_id[df_eps_len["eps_label"].str.contains("_-1")] = -1
    eps_id[eps_id != -1] = np.arange(sum(eps_id != -1))
    df_eps_len = df_eps_len.assign(eps_id=eps_id)
    
    df_track = df_track.merge(df_eps_len, how="outer", on="eps_label")
    df_track["eps_id"].loc[df_track["lead_track_id"].isna()] = -1
    return df_track

def filter_lane(df_track, lane_ids):
    """ Set car_follow_eps column to -1 if lane in lane_ids

    Args:
        df_track (pd.dataframe): track dataframe
        lane_ids (list): list of lane ids to be filtered

    Returns:
        df_track_ (pd.dataframe): track dataframe with updated car_follow_eps column
    """
    car_follow_eps = df_track["car_follow_eps"].values.copy()
    car_follow_eps[df_track["lane_id"].isin(lane_ids)] = -1
    df_track = df_track.assign(car_follow_eps=car_follow_eps) 
    return df_track

def filter_tail_merging(df_track, max_psi_error=0.05, min_bound_dist=1):
    """ Filter out merging trajectories, set car_follow_eps column to -1
    
    Args:
        df_track (pd.dataframe): track dataframe
        max_psi_error (float, optional): maximum allowed heading error. Default=0.05
        min_bound_dist (float, optional): minimum allowed distance to lane bounds. Default=1

    Returns:
        df_track_ (pd.dataframe): track dataframe with updated car_follow_eps column
    """
    def filter_heading(df):
        if df.head(1)["car_follow_eps"].values[0] != -1:
            # lane deviation based identification
            left_bound_dist = df["left_bound_dist"].values
            right_bound_dist = df["right_bound_dist"].values
            psi_error = wrap_angles(df["psi_rad"] - df["psi_tan"]).values
            
            is_merge = any(
                [np.abs(psi_error)[-1] > max_psi_error, 
                np.abs(left_bound_dist)[-1] < min_bound_dist,
                np.abs(right_bound_dist)[-1] < min_bound_dist]
            )
            if is_merge:
                merge_start_id = np.where(np.stack(
                    [np.abs(left_bound_dist) > min_bound_dist,
                    np.abs(right_bound_dist) > min_bound_dist,
                    np.abs(psi_error) < max_psi_error]
                ).all(0))[0]
                merge_start_id = 0 if len(merge_start_id) == 0 else merge_start_id[-1]
                
                car_follow_eps = df["car_follow_eps"].copy().values
                car_follow_eps[merge_start_id:] = -1
                df = df.assign(car_follow_eps=car_follow_eps)
        return df
    df_track_ = df_track.groupby(["track_id", "car_follow_eps"]).apply(filter_heading)
    return df_track_

def get_relative_df(df, agent_id, vars=["x", "y", "vx", "vy", "psi_rad"]):
    """
    Args:
        df (pd.dataframe): track dataframe
        agent_id (list): agent track id to concat with ego and compute relative state
        vars (list): variables to compute relative state: agent_state - ego_state

    Returns:
        df_rel (pd.dataframe): dataframe with relative states
    """
    df_ego = df.copy()
    df_ego["agent_id"] = agent_id

    df_agent = df.copy()
    df_agent.columns = [c + "_agent" for c in df_agent.columns]
    
    df_rel = df_ego.merge(
        df_agent,
        left_on=["scenario", "record_id", "frame_id", "agent_id"],
        right_on=["scenario_agent", "record_id_agent", "frame_id_agent", "track_id_agent"],
        how="left"
    )
    
    vars=["x", "y", "vx", "vy", "psi_rad"]
    agent_vars = [v + "_agent" for v in vars]
    rel_state = df_rel[agent_vars].to_numpy() - df_rel[vars].to_numpy()
    df_rel = pd.DataFrame(rel_state, columns=[v + "_rel" for v in vars])
    # df_rel["psi_rad_rel"].loc[df_rel["psi_rad_rel"] > np.pi] -= 2 * np.pi
    # df_rel["psi_rad_rel"].loc[df_rel["psi_rad_rel"] < -np.pi] += 2 * np.pi
    df_rel["psi_rad_rel"] = wrap_angles(df_rel["psi_rad_rel"])
    return df_rel

def get_ego_centric_df(df):
    df_ego = df.copy()

    # get initial condition
    df_init = df_ego.groupby(["scenario", "record_id", "track_id"]).head(1)
    df_init["x0"] = df_init["x"] 
    df_init["y0"] = df_init["y"] 
    df_init["psi0"] = df_init["psi_rad"]
    df_init = df_init[[
        "scenario", "record_id", "track_id", 
        "car_follow_eps", "x0", "y0", "psi0"
    ]]

    df_ego = df_ego.merge(
        df_init, 
        on=["scenario", "record_id", "track_id", "car_follow_eps"], 
        how="left"
    ).reset_index(drop=True)

    # convert heading angle to unit vector
    psi_x = np.cos(df_ego["psi_rad"]).to_numpy()
    psi_y = np.sin(df_ego["psi_rad"]).to_numpy()
    
    # route progress wrt initial ego coordinate
    progress_x = df_ego["x"] - df_ego["x0"]
    progress_y = df_ego["y"] - df_ego["y0"]
    df_ego["progress_x"], df_ego["progress_y"] = vector_projection(
        progress_x.to_numpy(), progress_y.to_numpy(), psi_x, psi_y
    )
    
    # instantaneous ego coordinate
    vars = [("vx", "vy"), ("ax", "ay"), ("x_rel", "y_rel"), ("vx_rel", "vy_rel")]
    for i, v in enumerate(vars):
        df_ego[v[0] + "_ego"], df_ego[v[1] + "_ego"] = vector_projection(
            df_ego[v[0]], df_ego[v[1]], psi_x, psi_y
        )
    return df_ego

def get_trajectory_segment_id(df, colnames):
    """ Divide trajectory into segments based on change in columns
    
    Args:
        df (pd.dataframe): track dataframe
        colnames (list): list of colum names to detect change
    
    Returns:
        seg_id (np.array): segment episode id. Invalid segment id equals nan. dim=[len(df)]
    """
    col_diff = [
        df.groupby("track_id")[colname].fillna(-1).diff() != 0 for colname in colnames
    ]
    df = df.assign(seg_change=np.any(np.stack(col_diff), axis=0))
    
    # get segment id per track
    track_seg_id = np.hstack([
        np.cumsum(1 * x[-1].values) for x in df.groupby("track_id")["seg_change"]
    ]).astype(float)
    for colname in colnames:
        track_seg_id[df[colname].isna()] = np.nan
    
    # get total segment id
    track_id = df["track_id"].values
    df_labels = pd.DataFrame(
        np.stack([track_id, track_seg_id]).T, 
        columns=["track_id", "track_seg_id"]
    )
    
    # concat seg labels
    is_valid_seg = np.isnan(track_seg_id) == False
    seg_labels = (df_labels["track_id"].apply(str) + "_" + df_labels["track_seg_id"].apply(str)).values
    seg_labels[is_valid_seg == False] = np.nan
    df_labels = df_labels.assign(seg_labels=seg_labels)
    
    valid_seg_labels = seg_labels[is_valid_seg]
    unique_valid_seg_labels = np.unique(valid_seg_labels)
    df_labels = df_labels.assign(
        seg_labels=df_labels["seg_labels"].map(
            dict(zip(unique_valid_seg_labels, np.arange(len(unique_valid_seg_labels))))
        )
    )
    seg_id = df_labels["seg_labels"].values
    return seg_id

def filter_segment_by_length(seg_id, min_seg_len):
    """ Filter track segment by length 
    
    Args:
        seg_id (np.array): track segment id. Invalid segment id equals nan
        min_seg_len (int): minimum segment length
    
    Returns:
        new_seg_id (np.array): new track segment id. Invalid segment id equals nan. dim=[len(seg_id)]
        new_seg_len (np.array): new track segment length. Invalid segment len equals nan. dim=[len(seg_id)]
    """
    unique_seg_id, seg_len = np.unique(seg_id, return_counts=True)
    df_seg_len = pd.DataFrame(
        np.stack([unique_seg_id, seg_len]).T,
        columns=["seg_id", "seg_len"]
    )
    df_seg_id = pd.DataFrame(seg_id, columns=["seg_id"])
    df_seg_id = df_seg_id.merge(df_seg_len, on="seg_id", how="left")
    
    new_seg_id = df_seg_id["seg_id"].values
    new_seg_id[df_seg_id["seg_len"] < min_seg_len] = np.nan
    new_seg_len = df_seg_id["seg_len"].values
    new_seg_len[df_seg_id["seg_len"] < min_seg_len] = np.nan
    new_seg_len[np.isnan(df_seg_id["seg_id"])] = np.nan
    df_seg_id = df_seg_id.assign(new_seg_id=new_seg_id)
    df_seg_id = df_seg_id.assign(new_seg_len=new_seg_len)
    
    unique_new_seg_id, new_seg_count = np.unique(new_seg_id, return_counts=True)
    idx_not_nan = np.where(np.isnan(unique_new_seg_id) == False)[0]
    unique_new_seg_id = unique_new_seg_id[idx_not_nan]
    new_seg_count = new_seg_count[idx_not_nan]
    df_seg_id = df_seg_id.assign(
        new_seg_id=df_seg_id["new_seg_id"].map(
            dict(zip(unique_new_seg_id, np.arange(len(unique_new_seg_id))))
        )
    )
    
    new_seg_id = df_seg_id["new_seg_id"].values
    new_seg_len = df_seg_id["new_seg_len"].values
    return new_seg_id, new_seg_len

def classify_tail_merging(df, tail=True, p_tail=0.3, max_d=1.2, class_weight={0: 1, 1: 2}):
    """ Classify tail merging using logistic regression
    
    Args:
        df (pd.dataframe): track dataframe with fields ["seg_id", "d", "dd", "ddd"]
        tail (bool, optional): whether to classify the tail of an episode. If false classify head. Default=True
        p_tail (float, optional): proportion of trajectory to be considered tail. Default=0.3
        max_d (float, optional): maximum distance from centerline. Tail trajectories with the last step 
            exceeding max_d will be labeled as merging for classifier training. Default=1.2
        class weight (dict, optional): classification class weight. Default={0: 1, 1:2}
    
    Returns:
        is_tail (np.array): indicator array for tail trajectories
        is_tail_merging (np.array): indicator array for tail merging
        cmat (np.array): confusion matrix
    """
    assert all([v in df.columns for v in ["seg_id", "ego_d", "ego_dd", "ddd"]])
    if "is_tail" in df.columns:
        df = df.drop(columns=["is_tail"])
    if "is_tail_merging" in df.columns:
        df = df.drop(columns=["is_tail_merging"])
    
    # get tail trajectories
    if tail:
        func = lambda x: x.tail(np.ceil(p_tail * len(x)).astype(int))
    else:
        func = lambda x: x.head(np.ceil(p_tail * len(x)).astype(int))
    df_tail = df.groupby("seg_id").apply(func).reset_index(drop=True)
    
    def label_merging(x):
        is_merging = np.zeros(len(x))
        if np.abs(x["ego_d"].values[-1]) > max_d:
            is_merging = np.ones(len(x))
        return pd.DataFrame(is_merging)
    
    df_merging_labels = df_tail.groupby("seg_id").apply(
        label_merging
    ).reset_index(drop=True)
    df_tail["merging_labels"] = df_merging_labels[0]
    
    # logistic regression features
    clf_inputs = np.abs(df_tail[["ego_d", "ego_dd", "ddd"]].values)
    clf_targets = df_tail["merging_labels"].values
    
    clf = LogisticRegression(class_weight=class_weight)
    clf.fit(clf_inputs, clf_targets)
    pred = clf.predict(clf_inputs)
    cmat = confusion_matrix(clf_targets, pred)
    
    # create labels
    df_tail["is_tail_merging"] = pred
    df_labels = df_tail[["track_id", "frame_id", "is_tail_merging"]]
    df_labels["is_tail"] = 1
    
    df = df.merge(df_labels, on=["track_id", "frame_id"], how="left")
    df["is_tail"] = df["is_tail"].fillna(0)
    df["is_tail_merging"] = df["is_tail_merging"].fillna(0)
    
    is_tail = df["is_tail"].values
    is_tail_merging = df["is_tail_merging"].values
    return is_tail, is_tail_merging, cmat