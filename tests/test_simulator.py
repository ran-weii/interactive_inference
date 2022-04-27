import os
import torch
import numpy as np
import pandas as pd

from src.map_api.lanelet import MapReader
from src.data.ego_dataset import EgoDataset, RelativeDataset
from src.simulation.simulator import InteractionSimulator
from src.visualization.animation import animate, save_animation
from src.simulation.observers import RelativeObserver
from src.simulation.controllers import AuxiliaryController

import warnings
warnings.filterwarnings("ignore")

lanelet_path = "../exp/lanelet"
data_path = "../interaction-dataset-master"
scenario1 = "DR_CHN_Merging_ZS"
scenario2 = "DR_DEU_Merging_MT"
scenario = scenario1
filename = "vehicle_tracks_007.csv"

# load map
map_data = MapReader(cell_len=10)
map_data.parse(os.path.join(data_path, "maps", scenario + ".osm"))

# load tracks
df_processed = pd.read_csv(
    os.path.join(data_path, "processed_trackfiles", scenario, filename)
)
df_track = pd.read_csv(
    os.path.join(data_path, "recorded_trackfiles", scenario, filename)
)
df_track = df_track.merge(df_processed, on=["track_id", "frame_id"])

def test_simulator():
    min_eps_len = 50
    max_eps_len = 1000
    ego_dataset = EgoDataset(
        df_track, map_data, 
        min_eps_len=min_eps_len, max_eps_len=max_eps_len
    )
    track_data = ego_dataset[0]
    
    env = InteractionSimulator(ego_dataset, map_data)
    observer = RelativeObserver(map_data)
    aux_controller = AuxiliaryController(map_data, ctl_direction="lat")
    
    observer.reset()
    obs_env = env.reset(11)
    obs = observer.observe(obs_env)
    for t in range(env.T):
        # get true agent control
        ctl_env = env.get_action()
        ctl_agent = observer.glob_to_ego(
            ctl_env[0], ctl_env[1], obs_env["ego"][2], obs_env["ego"][3]
        )[0]
        ctl_agent = torch.tensor([ctl_agent])

        # get aux controll
        ctl_aux = aux_controller.choose_action(obs_env, ctl_agent, ctl_env)
        
        ctl = torch.stack([ctl_agent, ctl_aux], dim=-1)
        ctl_env = observer.control(ctl, obs_env)
        
        obs_env, r, done, info = env.step(ctl_env)
        obs = observer.observe(obs_env)
        if done:
            break
    ani = animate(map_data, env._states, env._track, title="test", annot=False)
    save_animation(ani, "/Users/rw422/Documents/render_ani.mp4")

def test_observer():
    min_eps_len = 50
    max_eps_len = 1000
    ego_dataset = EgoDataset(
        df_track, map_data, 
        min_eps_len=min_eps_len, max_eps_len=max_eps_len
    )
    rel_dataset = RelativeDataset(
        df_track, map_data, 
        min_eps_len=min_eps_len, max_eps_len=max_eps_len
    )
    test_track_ids = [771]
    eps_id = rel_dataset.df_track.loc[rel_dataset.df_track["track_id"].isin(test_track_ids)]["eps_id"].unique()

    """ simulate episode """
    env = InteractionSimulator(ego_dataset, map_data)
    observer = RelativeObserver(map_data)
    eps_id = 771 #12
    obs_env = env.reset(eps_id)
    
    # get correct relative dataset
    rel_track = rel_dataset[eps_id]
    rel_obs = rel_track["ego"]
    rel_act = rel_track["act"]
    rel_keys = rel_dataset.ego_fields
    
    ctl_env_true = [np.empty(0)] * (env.T - 1)
    ctl_env_pred = [np.empty(0)] * (env.T - 1)
    
    obs_true = [torch.empty(0)] * env.T
    obs_pred = [torch.empty(0)] * env.T 
    obs_true[0] = rel_obs[0]
    obs_pred[0] = observer.observe(obs_env)
    for t in range(env.T):
        ctl = rel_act[t].view(1, -1)
        ctl_observer = observer.control(ctl, obs_env)
        # ctl_env = env.get_action()
        ctl_env = ctl_observer
        obs_env, r, done, info = env.step(ctl_env)
        
        obs_true[t + 1] = rel_obs[t]
        obs_pred[t + 1] = observer.observe(obs_env)

        ctl_env_true[t] = env.get_action()
        ctl_env_pred[t] = ctl_observer
        if done:
            break
    
    obs_true = torch.stack(obs_true)
    obs_pred = torch.stack(obs_pred).squeeze(1)

    ctl_env_true = np.stack(ctl_env_true)
    ctl_env_pred = np.stack(ctl_env_pred)

    # print(obs_true.shape)
    # print(obs_pred.shape)

    # print()
    # print(ctl_env_true.shape)
    # print(ctl_env_pred.shape)

    # ani = animate(map_data, env._states, env._track, title="test", annot=False)
    # save_animation(ani, "/Users/rw422/Documents/render_ani.mp4")
    
    """ TODO: while most measures are good lane deviation seem to be off, is it because of dynamics mismatch? """
    # compare with relative dataset observations
    import matplotlib.pyplot as plt
    
    # plot observer observations
    fig, ax = plt.subplots(obs_true.shape[1], 1, figsize=(6, 10))
    for i in range(obs_true.shape[1]):
        ax[i].plot(obs_true[:, i], label="true")
        ax[i].plot(obs_pred[:, i], label="pred")
        ax[i].set_ylabel(rel_keys[i])
        ax[i].legend()
    plt.tight_layout()
    
    # plot observer controls
    fig, ax = plt.subplots(3, 1, figsize=(6, 4))
    ax[0].plot(ctl_env_true[:, 0], label="true")
    ax[0].plot(ctl_env_pred[:, 0], label="pred")
    ax[0].plot(env._track["act"][:, 0], label="raw")
    ax[0].set_ylabel("x")
    ax[0].legend()

    ax[1].plot(ctl_env_true[:, 1], label="true")
    ax[1].plot(ctl_env_pred[:, 1], label="pred")
    ax[1].plot(env._track["act"][:, 1], label="raw")
    ax[1].set_ylabel("y")
    ax[1].legend()
    
    ax[2].plot(env._states[:, 4])
    ax[2].set_ylabel("psi")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # test_simulator()
    test_observer()