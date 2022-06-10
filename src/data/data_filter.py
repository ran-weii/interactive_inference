import numpy as np
import pandas as pd
from src.data.geometry import vector_projection, wrap_angles

""" debug episode tail merging """
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
