import os
import numpy as np
import pandas as pd
from src.data.lanelet import load_lanelet_df
from src.data.ego_dataset import EgoDataset
from src.map_api.lanelet import MapData
from src.simulation.simulator import InteractionSimulator
from src.visualization.animation import animate

lanelet_path = "../exp/lanelet"
data_path = "../interaction-dataset-master"
scenario1 = "DR_CHN_Merging_ZS"
scenario2 = "DR_DEU_Merging_MT"
scenario = scenario2
filename = "vehicle_tracks_007.csv"

def test_simulator():
    # load map
    # filepath = os.path.join(data_path, "maps", scenario + ".osm")
    # map_data = MapData(cell_len=10)
    # map_data.parse(filepath, verbose=True)
    df_lanelet = load_lanelet_df(os.path.join(lanelet_path, scenario + ".json"))

    # load tracks
    # df_processed = pd.read_csv(
    #     os.path.join(data_path, "processed_trackfiles", scenario, "test.csv")
    # )
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
    obs = env.reset(0)
    for t in range(env.T):
        action = env.get_action()
        obs, r, done, info = env.step(action)
        if done:
            break
    
    osm_path = os.path.join(data_path, "maps", scenario + ".osm")
    animate(osm_path, env._states, env._track)

if __name__ == "__main__":
    test_simulator()