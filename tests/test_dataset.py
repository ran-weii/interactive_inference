import os
import numpy as np
import pandas as pd
from src.data.ego_dataset import (
    EgoDataset, SimpleEgoDataset, RelativeDataset, RelativeDatasetNew)
from src.data.lanelet import load_lanelet_df
from src.visualization.visualizer import build_bokeh_sources, visualize_scene
from bokeh.plotting import save

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

lanelet_path = "../exp/lanelet"
data_path = "../interaction-dataset-master"
save_path = "../doc/fig"
if not os.path.exists(save_path):
    os.mkdir(save_path)
scenario1 = "DR_CHN_Merging_ZS"
scenario2 = "DR_DEU_Merging_MT"
filename = "vehicle_tracks_007.csv"

def test_ego_dataset(scenario):
    df_lanelet = load_lanelet_df(os.path.join(lanelet_path, scenario + ".json"))
    df_processed = pd.read_csv(
        os.path.join(data_path, "processed_trackfiles", scenario, filename)
    )
    df_track = pd.read_csv(
        os.path.join(data_path, "recorded_trackfiles", scenario, filename)
    )
    df_track = df_track.merge(df_processed, on=["track_id", "frame_id"])

    min_eps_len = 50
    ego_dataset = EgoDataset(df_track, df_lanelet, min_eps_len)
    track_data = ego_dataset[0]
    
    frames, lanelet_source = build_bokeh_sources(
        track_data, df_lanelet, ego_dataset.ego_fields, ego_dataset.agent_fields
    )
    fig = visualize_scene(frames, lanelet_source)
    save(fig, os.path.join(save_path, f"{scenario}_Ego.html"))
    
def test_simple_ego_dataset(scenario):
    df_lanelet = load_lanelet_df(os.path.join(lanelet_path, scenario + ".json"))
    df_processed = pd.read_csv(
        os.path.join(data_path, "processed_trackfiles", scenario, filename)
    )
    df_track = pd.read_csv(
        os.path.join(data_path, "recorded_trackfiles", scenario, filename)
    )
    df_track = df_track.merge(df_processed, on=["track_id", "frame_id"])
    
    min_eps_len = 50
    ego_dataset = SimpleEgoDataset(df_track, df_lanelet, min_eps_len)
    track_data = ego_dataset[0]
    
    frames, lanelet_source = build_bokeh_sources(
        track_data, df_lanelet, ego_dataset.ego_fields, ego_dataset.agent_fields
    )
    fig = visualize_scene(frames, lanelet_source)
    save(fig, os.path.join(save_path, f"{scenario}_SimpleEgo.html"))

def test_relative_dataset(scenario):
    df_lanelet = load_lanelet_df(os.path.join(lanelet_path, scenario + ".json"))
    df_processed = pd.read_csv(
        os.path.join(data_path, "processed_trackfiles", scenario, filename)
    )
    df_track = pd.read_csv(
        os.path.join(data_path, "recorded_trackfiles", scenario, filename)
    )
    df_track = df_track.merge(df_processed, on=["track_id", "frame_id"])
    df_track["psi_rad"] = np.clip(df_track["psi_rad"], -np.pi, np.pi)
    
    min_eps_len = 50
    rel_dataset = RelativeDataset(df_track, df_lanelet, min_eps_len)
    track_data = rel_dataset[0]
    
    assert track_data["ego"].shape[1] == len(rel_dataset.ego_fields)
    assert track_data["act"].shape[1] == len(rel_dataset.act_fields)
    print("test_relative_dataset passed")

def test_relative_dataset_new():
    from torch.utils.data import DataLoader
    from src.map_api.lanelet import MapData
    from src.data.ego_dataset import relative_collate_fn
    
    scenario = "DR_CHN_Merging_ZS"
    filename_track = "vehicle_tracks_007.csv"
    filename_processed = "test.csv"
    df_processed = pd.read_csv(
        os.path.join(data_path, "processed_trackfiles", scenario, filename_processed)
    )
    df_track = pd.read_csv(
        os.path.join(data_path, "recorded_trackfiles", scenario, filename_track)
    )
    df_track = df_track.merge(df_processed, on=["track_id", "frame_id"])
    
    map_data = MapData(cell_len=10)
    map_data.parse(
        os.path.join(data_path, "maps", f"{scenario}.osm")
    )
    
    min_eps_len = 50
    max_eps_len = 100
    ego_dataset = RelativeDatasetNew(
        df_track, map_data, min_eps_len=min_eps_len, max_eps_len=max_eps_len
    )
    loader = DataLoader(ego_dataset, batch_size=32, collate_fn=relative_collate_fn)
    track_data = ego_dataset[0]
    batch = next(iter(loader))
    
    print("test_relative_dataset_new passed")

if __name__ == "__main__":
    test_ego_dataset(scenario1)
    test_ego_dataset(scenario2)
    test_simple_ego_dataset(scenario1)
    test_simple_ego_dataset(scenario2)
    test_relative_dataset(scenario1)
    test_relative_dataset_new()