import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.train_utils import load_data
from src.map_api.lanelet import MapReader
from src.data.data_filter import filter_lane, filter_tail_merging
from src.data.ego_dataset import EgoDataset, RelativeDataset
from src.simulation.simulator import InteractionSimulator
from src.visualization.animation import animate, save_animation

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

def test_simulator():
    df_track = load_data(data_path, scenario, filename)
    
    min_eps_len = 50
    max_eps_len = 1000
    ego_dataset = EgoDataset(df_track)
    track_data = ego_dataset[0]
    
    env = InteractionSimulator(ego_dataset, map_data)
    
    obs_env = env.reset(11)
    for t in range(env.T):
        # get true agent control
        ctl_env = env.get_action()
        obs_env, r, done, info = env.step(ctl_env)
        
        if done:
            break
    
    print("true", env._track_data["ego"][:, 5])
    print("est", env._sim_states[:, 5])

    fig, ax = plt.subplots(3, 2, figsize=(6, 4))
    ax = ax.flat
    for i in range(6):
        ax[i].plot(env._track_data["ego"][:, i])
        ax[i].plot(env._sim_states[:, i])
    plt.tight_layout()
    plt.show()
    # ani = animate(map_data, env._sim_states, env._track_data, title="test", annot=True)
    # save_animation(ani, "/Users/rw422/Documents/render_ani.mp4")

def test_observer():
    from src.simulation.observers import Observer
    from src.simulation.observers import FEATURE_SET
    
    observer = Observer(
        map_data, ego_features=FEATURE_SET["ego"],
        relative_features=FEATURE_SET["relative"]
    )
    
    # create synthetic data
    # ego_state = np.array([1065.303, 958.918, -10, -0.01, np.pi, 0])
    ego_state = np.array([1091.742, 950.918, -10, -0.01, np.pi, 0])
    agent_state = np.array([[1060.303, 960.918, -10, -0.01, np.pi]])
    state = {"ego": ego_state, "agents": agent_state}
    obs = observer.observe(state)
    assert list(obs.shape) == [1, len(observer.feature_set)]
    print("test_pbserver_new passed")

if __name__ == "__main__":
    # test_simulator()
    test_observer()