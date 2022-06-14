import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.map_api.lanelet import MapReader
from src.data.frenet import FrenetPath, Trajectory

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

lanelet_path = "../exp/lanelet"
data_path = "../interaction-dataset-master"
save_path = "../doc/fig"
if not os.path.exists(save_path):
    os.mkdir(save_path)
scenario = "DR_CHN_Merging_ZS"
filename = "vehicle_tracks_007.csv"

# load map
map_data = MapReader(cell_len=10)
map_data.parse(os.path.join(data_path, "maps", scenario + ".osm"))

def load_data():
    # load tracks
    df_processed = pd.read_csv(
        os.path.join(data_path, "processed_trackfiles", scenario, filename)
    )
    df_track = pd.read_csv(
        os.path.join(data_path, "recorded_trackfiles", scenario, filename)
    )
    df_track = df_track.merge(df_processed, on=["track_id", "frame_id"])
    return df_track

def dynamics(x, u):
    """ Linear dynamics model """
    dt = 0.1
    _A = np.array(
        [[1, 0, dt,  0],
        [0, 1,  0, dt],
        [0, 0,  1,  0],
        [0, 0,  0,  1]]
    )
    
    _C = np.array(
        [[0.5 * dt**2, 0],
        [0, 0.5 * dt**2],
        [dt, 0],
        [0, dt]]
    )

    x_t = np.sum(_A.reshape(1, 4, 4) * x.reshape(-1, 1, 4), axis=-1)
    u_t = np.sum(_C.reshape(1, 4, 2) * u.reshape(-1, 1, 2), axis=-1)
    x_t += u_t
    return x_t

def simulate_frenet_trajectory(s0, u, T, ref_path):
    """
    simulate a trajectory in frenet frame
    convert to cartesian frame
    convert back to frenet frame 

    Args:
        s0 (np.array): initial frenet state [s, d, ds, dd]
        u (np.array): control vector [2]
        T (int): simulation time steps
        ref_path (FrenetPath): the reference path object

    Returns:
        s_traj (np.array): frenet frame trajectory [T, 4]
        x_traj (np.array): cartesian frame trajectory [T, 6]
        s_traj_reverse (np.array): inversed frenet frame trajectory [T, 6]
    """
    s_traj = [np.hstack([s0, u])]
    s_traj_reverse = []
    x_traj = []
    for t in range(T):
        s_t = dynamics(s_traj[t][:4], u).reshape(-1)
        s_traj.append(np.hstack([s_t, u]))

        s_condition = np.array([s_t[0], s_t[2], u[0]])
        d_condition = np.array([s_t[1], s_t[3], u[1]])
        x_t = ref_path.frenet_to_cartesian(s_condition, d_condition)
        x_traj.append(np.array(x_t))

        s_t_reverse = ref_path.cartesian_to_frenet(
            x_t[0], x_t[1], x_t[2], x_t[3], x_t[4], x_t[5]
        )
        s_traj_reverse.append(np.hstack(s_t_reverse))
        # break

    s_traj = np.array(s_traj)
    s_traj = np.stack([
        s_traj[:, 0], s_traj[:, 2], s_traj[:, 4],
        s_traj[:, 1], s_traj[:, 3], s_traj[:, 5],
    ]).T
    x_traj = np.array(x_traj)
    s_traj_reverse = np.array(s_traj_reverse)
    return s_traj, x_traj, s_traj_reverse

def plot_cartesian_trajectory(x_traj, figsize=(6, 4)):
    """
    Args:
        x_traj (np.array): cartesian trajectory [x, y, v, a, theta, kappa]
    """
    fig, ax = plt.subplots(3, 2, figsize=figsize)
    ax[0, 0].plot(x_traj[:, 0], label="x")
    ax[0, 1].plot(x_traj[:, 1], label="y")

    ax[1, 0].plot(x_traj[:, 2], label="v")
    ax[1, 1].plot(x_traj[:, 3], label="a")

    ax[2, 0].plot(x_traj[:, 4], label="theta")
    ax[2, 1].plot(x_traj[:, 5], label="kappa")

    for x in ax.flat:
        x.legend()
        x.set_xlabel("time")

    plt.tight_layout()
    return fig, ax

def plot_frenet_trajectory(s_traj, s_traj_reverse, figsize=(6, 4)):
    """
    Args:
        s_traj (np.array): frenet trajectory [s, ds, dds, d, dd, ddd]
        s_traj_reverse (np.array): inversed frenet trajectory [s, ds, dds, d, dd, ddd]
    """
    fig, ax = plt.subplots(3, 2, figsize=figsize)
    ax[0, 0].plot(s_traj[1:, 0], label="s")
    ax[0, 0].plot(s_traj_reverse[:, 0], label="rev")

    ax[0, 1].plot(s_traj[1:, 3], label="d")
    ax[0, 1].plot(s_traj_reverse[:, 3], label="rev")

    ax[1, 0].plot(s_traj[1:, 1], label="ds")
    ax[1, 0].plot(s_traj_reverse[:, 1], label="rev")

    ax[1, 1].plot(s_traj[1:, 4], label="dd")
    ax[1, 1].plot(s_traj_reverse[:, 4], label="rev")
    
    ax[2, 0].plot(s_traj[:, 2], label="dds")
    ax[2, 0].plot(s_traj_reverse[:, 2], label="rev")
    
    ax[2, 1].plot(s_traj[:, 5], label="ddd")
    ax[2, 1].plot(s_traj_reverse[:, 5], label="rev")

    for x in ax.flat:
        x.legend()
        x.set_xlabel("time")

    plt.tight_layout()
    return fig, ax

def test_frenet_path():
    # test forward lane
    lane_id = 0
    ref_coords = np.array(map_data.lanes[lane_id].centerline.linestring.coords)
    ref_path = FrenetPath(ref_coords)
    
    T = 18
    s0 = np.array([0, 0, 40, 0])
    u = np.array([40, -2])
    s_traj, x_traj, s_traj_reverse = simulate_frenet_trajectory(s0, u, T, ref_path)
    
    plot_cartesian_trajectory(x_traj)
    plot_frenet_trajectory(s_traj, s_traj_reverse)
    
    # plot generated trajectory on map
    fig, ax = map_data.plot(option="lanes", annot=True, figsize=(15, 8))
    ax.plot(ref_path.cubic_spline[:, 0], ref_path.cubic_spline[:, 1], "g-")
    ax.plot(x_traj[:, 0], x_traj[:, 1], "o", label="f2c")
    ax.legend()
    ax.set_title("test forward lane")
    
    # test reverse lane
    lane_id = 3
    ref_coords = np.array(map_data.lanes[lane_id].centerline.linestring.coords)
    ref_path = FrenetPath(ref_coords)
    
    T = 18
    s0 = np.array([0, 0, 40, 0])
    u = np.array([40, -2])
    s_traj, x_traj, s_traj_reverse = simulate_frenet_trajectory(s0, u, T, ref_path)
    
    plot_cartesian_trajectory(x_traj)
    plot_frenet_trajectory(s_traj, s_traj_reverse)
    
    # plot generated trajectory on map
    fig, ax = map_data.plot(option="lanes", annot=True, figsize=(15, 8))
    ax.plot(ref_path.cubic_spline[:, 0], ref_path.cubic_spline[:, 1], "g-")
    ax.plot(x_traj[:, 0], x_traj[:, 1], "o", label="f2c")
    ax.legend()
    ax.set_title("test reverse lane")
    plt.show()
    print("test_frenet_path passed")

def test_frenet_trajectory():
    df_track = load_data()

    track_id = 1
    car_follow_eps = 1
    df_trajectory = df_track.loc[
        (df_track["track_id"] == track_id) & (df_track["car_follow_eps"] == car_follow_eps)
    ]
    lane_id = df_trajectory["lane_id"].values[0]

    # get ego trajectory in cartesian frame
    x = df_trajectory["x"].values
    y = df_trajectory["y"].values
    vx = df_trajectory["vx"].values
    vy = df_trajectory["vy"].values
    ax = df_trajectory["ax"].values
    ay = df_trajectory["ay"].values
    theta = df_trajectory["psi_rad"].values
    ego_trajectory = Trajectory(x, y, vx, vy, ax, ay, theta)
    
    # convert ego trajectory to frenet frame
    ref_coords = np.array(map_data.lanes[lane_id].centerline.linestring.coords)
    ref_path = FrenetPath(ref_coords)
    ego_trajectory.get_frenet_trajectory(ref_path)
    
    # plot converted trajectory
    x_traj = np.stack([
        ego_trajectory.x, ego_trajectory.y, ego_trajectory.v, 
        ego_trajectory.a, ego_trajectory.theta, ego_trajectory.kappa
    ]).T
    s_traj = np.hstack([ego_trajectory.s_condition, ego_trajectory.d_condition])
    plot_cartesian_trajectory(x_traj)
    plot_frenet_trajectory(s_traj, s_traj)
    plt.show()
    print("test_frenet_trajectory passed")

if __name__ == "__main__":
    test_frenet_path()
    test_frenet_trajectory()