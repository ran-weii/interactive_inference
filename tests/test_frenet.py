import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.map_api.lanelet import MapReader
from src.map_api.frenet import FrenetPath, Trajectory

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

def simulate_trajectory(pos0, vel0, acc, T):
    """ Simulate a trajectory in the cartesian frame 
    
    Returns:
        x (np.array): [x, y, vy, vy, ax, ay]
    """
    x = [np.hstack([pos0, vel0])]
    for t in range(T):
        x.append(dynamics(x[t], acc).reshape(-1))
    x = np.stack(x)
    x = np.hstack([x, acc.reshape(1, 2) * np.ones((T + 1, 2))])
    return x

def plot_cartesian_trajectory(x_traj, fig_ax=None, figsize=(6, 4)):
    """
    Args:
        x_traj (np.array): cartesian trajectory [x, y, v, a, theta, kappa]
    """
    if fig_ax is None:
        fig, ax = plt.subplots(3, 2, figsize=figsize)
    else:
        fig, ax = fig_ax

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

def plot_frenet_trajectory(s_traj, fig_ax=None, figsize=(6, 4)):
    """
    Args:
        s_traj (np.array): frenet trajectory [s, ds, dds, d, dd, ddd]
    """
    if fig_ax is None:
        fig, ax = plt.subplots(3, 2, figsize=figsize)
    else:
        fig, ax = fig_ax

    ax[0, 0].plot(s_traj[:, 0], label="s")
    ax[0, 1].plot(s_traj[:, 3], label="d")

    ax[1, 0].plot(s_traj[:, 1], label="ds")
    ax[1, 1].plot(s_traj[:, 4], label="dd")
    
    ax[2, 0].plot(s_traj[:, 2], label="dds")
    ax[2, 1].plot(s_traj[:, 5], label="ddd")

    for x in ax.flat:
        x.legend()
        x.set_xlabel("time")

    plt.tight_layout()
    return fig, ax

def plot_velocity_acceleration_norms(x_traj, s_traj, figsize=(6, 1.5)):
    v_norm_x = np.abs(x_traj[:, 2])
    v_norm_s = np.sqrt(s_traj[:, 1]**2 + s_traj[:, -2]**2)

    a_norm_x = np.abs(x_traj[:, 3])
    a_norm_s = np.sqrt(s_traj[:, 2]**2 + s_traj[:, -1]**2)

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].plot(v_norm_x, label="carte")
    ax[0].plot(v_norm_s, label="frenet")
    ax[0].set_ylabel("v")
    ax[0].set_xlabel("time")
    ax[0].legend()
    
    ax[1].plot(a_norm_x, label="carte")
    ax[1].plot(a_norm_s, label="frenet")
    ax[1].set_ylabel("a")
    ax[1].set_xlabel("time")
    ax[1].legend()

    plt.tight_layout()
    return fig, ax

def test_frenet_path():
    # test forward lane
    lane_id = 0
    ref_coords = np.array(map_data.lanes[lane_id].centerline.linestring.coords)
    ref_path = FrenetPath(ref_coords)
    
    # generate a trajectory in the cartesian frame
    pos0 = ref_coords[0]
    vel0 = np.array([40, -15])
    acc = np.array([40, 10])
    T = 10
    dt = 0.1
    sim_trajectory = simulate_trajectory(pos0, vel0, acc, T)
    
    # pack sim data
    x = sim_trajectory[:, 0]
    y = sim_trajectory[:, 1]
    vx = sim_trajectory[:, 2]
    vy = sim_trajectory[:, 3]
    ax = np.gradient(vx) / dt
    ay = np.gradient(vy) / dt
    theta = np.arctan2(vy, vx)

    traj = Trajectory(
        x, y, vx, vy, ax, ay, theta
    )
    traj.get_frenet_trajectory(ref_path)
    cartesian_trajectory = np.stack([traj.x, traj.y, traj.v, traj.a, traj.theta, traj.kappa]).T
    
    # convert back to cartesian frame
    cartesian_trajectory_reverse = []
    for t in range(traj.length):
        x_t = ref_path.frenet_to_cartesian(traj.s_condition[t], traj.d_condition[t])
        cartesian_trajectory_reverse.append(np.array(x_t))
    cartesian_trajectory_reverse = np.stack(cartesian_trajectory_reverse)
    
    # plot cartesian trajectory
    fig, ax = plot_cartesian_trajectory(cartesian_trajectory)
    fig, ax = plot_cartesian_trajectory(cartesian_trajectory_reverse, fig_ax=(fig, ax))
    ax[0, 0].set_title("test forward lane")

    # plot frenet trajectory
    frenet_trajectory = np.hstack([traj.s_condition, traj.d_condition])
    fig, ax = plot_frenet_trajectory(frenet_trajectory)
    ax[0, 0].set_title("test forward lane")

    # compare norms
    fig, ax = plot_velocity_acceleration_norms(cartesian_trajectory, frenet_trajectory)
    ax[0].set_title("test forward lane")
    
    # plot tangent and norm vectors
    fig, ax = map_data.plot(option="lanes", annot=True, figsize=(15, 8))
    ax.plot(cartesian_trajectory[:, 0], cartesian_trajectory[:, 1], "ro-")
    ax.quiver(cartesian_trajectory[:, 0], cartesian_trajectory[:, 1], traj.tan_vec[:, 0], traj.tan_vec[:, 1], color="r", label="tan")
    ax.quiver(cartesian_trajectory[:, 0], cartesian_trajectory[:, 1], traj.norm_vec[:, 0], traj.norm_vec[:, 1], color="k", label="norm")
    ax.quiver(cartesian_trajectory[:, 0], cartesian_trajectory[:, 1], traj.ax, traj.ay, color="g", label="a")
    ax.legend()
    ax.set_title("test forward lane")

    fig, ax = map_data.plot(option="lanes", annot=True, figsize=(15, 8))
    ax.plot(cartesian_trajectory[:, 0], cartesian_trajectory[:, 1], "ro-")
    ax.quiver(cartesian_trajectory[:, 0], cartesian_trajectory[:, 1], traj.tan_vec[:, 0], traj.tan_vec[:, 1], color="r", label="tan")
    ax.quiver(cartesian_trajectory[:, 0], cartesian_trajectory[:, 1], traj.norm_vec[:, 0], traj.norm_vec[:, 1], color="k", label="norm")
    ax.quiver(cartesian_trajectory[:, 0], cartesian_trajectory[:, 1], traj.vx, traj.vy, color="g", label="v")
    plt.legend()
    ax.set_title("test forward lane")
    plt.show()
    
    # # test reverse lane    
    lane_id = 3
    ref_coords = np.array(map_data.lanes[lane_id].centerline.linestring.coords)
    ref_path = FrenetPath(ref_coords)
    
    # generate a trajectory in the cartesian frame
    pos0 = ref_coords[0]
    vel0 = np.array([-40, -15])
    acc = np.array([-40, 12])
    T = 10
    dt = 0.1
    sim_trajectory = simulate_trajectory(pos0, vel0, acc, T)
    
    # pack sim data
    x = sim_trajectory[:, 0]
    y = sim_trajectory[:, 1]
    vx = sim_trajectory[:, 2]
    vy = sim_trajectory[:, 3]
    ax = np.gradient(vx) / dt
    ay = np.gradient(vy) / dt
    theta = np.arctan2(vy, vx)

    traj = Trajectory(
        x, y, vx, vy, ax, ay, theta
    )
    traj.get_frenet_trajectory(ref_path)
    cartesian_trajectory = np.stack([traj.x, traj.y, traj.v, traj.a, traj.theta, traj.kappa]).T
    
    # convert back to cartesian frame
    cartesian_trajectory_reverse = []
    for t in range(traj.length):
        x_t = ref_path.frenet_to_cartesian(traj.s_condition[t], traj.d_condition[t])
        cartesian_trajectory_reverse.append(np.array(x_t))
    cartesian_trajectory_reverse = np.stack(cartesian_trajectory_reverse)
    
    # plot cartesian trajectory
    fig, ax = plot_cartesian_trajectory(cartesian_trajectory)
    fig, ax = plot_cartesian_trajectory(cartesian_trajectory_reverse, fig_ax=(fig, ax))
    ax[0, 0].set_title("test reverse lane")

    # plot frenet trajectory
    frenet_trajectory = np.hstack([traj.s_condition, traj.d_condition])
    fig, ax = plot_frenet_trajectory(frenet_trajectory)
    ax[0, 0].set_title("test reverse lane")

    # compare norms
    fig, ax = plot_velocity_acceleration_norms(cartesian_trajectory, frenet_trajectory)
    ax[0].set_title("test reverse lane")
    
    # plot tangent and norm vectors
    fig, ax = map_data.plot(option="lanes", annot=True, figsize=(15, 8))
    ax.plot(cartesian_trajectory[:, 0], cartesian_trajectory[:, 1], "ro-")
    ax.quiver(cartesian_trajectory[:, 0], cartesian_trajectory[:, 1], traj.tan_vec[:, 0], traj.tan_vec[:, 1], color="r", label="tan")
    ax.quiver(cartesian_trajectory[:, 0], cartesian_trajectory[:, 1], traj.norm_vec[:, 0], traj.norm_vec[:, 1], color="k", label="norm")
    ax.quiver(cartesian_trajectory[:, 0], cartesian_trajectory[:, 1], traj.ax, traj.ay, color="g", label="a")
    ax.legend()
    ax.set_title("test reverse lane")

    fig, ax = map_data.plot(option="lanes", annot=True, figsize=(15, 8))
    ax.plot(cartesian_trajectory[:, 0], cartesian_trajectory[:, 1], "ro-")
    ax.quiver(cartesian_trajectory[:, 0], cartesian_trajectory[:, 1], traj.tan_vec[:, 0], traj.tan_vec[:, 1], color="r", label="tan")
    ax.quiver(cartesian_trajectory[:, 0], cartesian_trajectory[:, 1], traj.norm_vec[:, 0], traj.norm_vec[:, 1], color="k", label="norm")
    ax.quiver(cartesian_trajectory[:, 0], cartesian_trajectory[:, 1], traj.vx, traj.vy, color="g", label="v")
    plt.legend()
    ax.set_title("test reverse lane")
    plt.show()

    print("test_frenet_path passed")

if __name__ == "__main__":
    test_frenet_path()