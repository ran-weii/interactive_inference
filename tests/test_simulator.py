import os
import torch
import pandas as pd

from src.data.lanelet import load_lanelet_df
from src.data.ego_dataset import EgoDataset
from src.map_api.lanelet import MapReader
from src.simulation.simulator import InteractionSimulator
from src.visualization.animation import animate, save_animation
from src.simulation.observers import RelativeObserver

lanelet_path = "../exp/lanelet"
data_path = "../interaction-dataset-master"
scenario1 = "DR_CHN_Merging_ZS"
scenario2 = "DR_DEU_Merging_MT"
scenario = scenario1
filename = "vehicle_tracks_007.csv"

def test_simulator():
    # load map
    filepath = os.path.join(data_path, "maps", scenario + ".osm")
    map_data = MapReader(cell_len=10)
    map_data.parse(filepath, verbose=True)
    df_lanelet = load_lanelet_df(os.path.join(lanelet_path, scenario + ".json"))

    # load tracks
    df_processed = pd.read_csv(
        os.path.join(data_path, "processed_trackfiles", scenario, filename)
    )
    df_track = pd.read_csv(
        os.path.join(data_path, "recorded_trackfiles", scenario, filename)
    )
    df_track = df_track.merge(df_processed, on=["track_id", "frame_id"])

    ego_dataset = EgoDataset(df_track, df_lanelet, max_eps_len=1000)
    track_data = ego_dataset[0]
    
    env = InteractionSimulator(ego_dataset, df_lanelet)
    observer = RelativeObserver(map_data)
    obs_env = env.reset(1)
    obs = observer.observe(obs_env)
    for t in range(env.T):
        ctl = torch.tensor([0, -0.5])
        ctl_env = observer.control(ctl, obs_env)
        obs_env, r, done, info = env.step(ctl_env)
        obs = observer.observe(obs_env)
        if done:
            break
    ani = animate(map_data, env._states, env._track)
    save_animation(ani, "/Users/rw422/Documents/render_ani.mp4")

if __name__ == "__main__":
    test_simulator()