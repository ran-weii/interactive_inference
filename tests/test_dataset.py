import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src.data.ego_dataset import EgoDataset, RelativeDataset
from src.data.lanelet import load_lanelet_df
from src.data.train_utils import load_data
from src.map_api.lanelet import MapReader
from src.simulation.observers import RelativeObserver
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
    df_track = load_data(data_path, scenario, filename)
    
    map_data = MapReader()
    map_data.parse(os.path.join(data_path, "maps", scenario + ".osm"))
    
    ego_dataset = EgoDataset(
        df_track, train_labels_col="is_train",
        seed=0
    )
    track_data = ego_dataset[11]
    
    print("meta", track_data["meta"])
    frames, lanelet_source = build_bokeh_sources(
        track_data, map_data, ego_dataset.ego_fields
    )
    fig = visualize_scene(frames, lanelet_source)
    save(fig, os.path.join(save_path, f"{scenario}_Ego.html"))

def test_relative_dataset(scenario):
    from src.simulation.observers import FEATURE_SET
    df_track = load_data(data_path, scenario, filename)
    
    map_data = MapReader(cell_len=10)
    map_data.parse(os.path.join(data_path, "maps", scenario + ".osm"))
    
    feature_set = FEATURE_SET["ego"] + FEATURE_SET["relative"]
    action_set = ["dds", "ddd"]
    max_eps_len = 50
    rel_dataset = RelativeDataset(
        df_track, feature_set, action_set, max_eps_len=max_eps_len, state_action=False
    )
    track_data = rel_dataset[0]

    loader = DataLoader(rel_dataset, batch_size=64)
    batch = next(iter(loader))
    
    assert track_data["ego"].shape[-1] == len(rel_dataset.ego_fields)
    assert track_data["act"].shape[-1] == len(rel_dataset.act_fields)
    print("test_relative_dataset passed")

def test_observer(scenario):
    df_processed = pd.read_csv(
        os.path.join(data_path, "processed_trackfiles", scenario, filename)
    )
    df_track = pd.read_csv(
        os.path.join(data_path, "recorded_trackfiles", scenario, filename)
    )
    df_track = df_track.merge(df_processed, on=["track_id", "frame_id"])
    
    map_data = MapReader(cell_len=10)
    map_data.parse(os.path.join(data_path, "maps", scenario + ".osm"))
    
    observer = RelativeObserver(map_data, fields="lv_fv")

    # test offline
    df_track_obs = observer.observe_df(df_track)
    obs_offline = df_track_obs.iloc[0][observer.ego_fields].values

    # test online
    ego_dataset = EgoDataset(
        df_track, observer, 
        min_eps_len=50, max_eps_len=1000
    )
    track_data = ego_dataset[0]
    
    obs = observer.observe({"ego": track_data["ego"][0], "agents": track_data["agents"][0]})
    diff_obs = obs_offline[:4] - obs.view(-1).numpy()[:4]
    assert np.all(diff_obs < 1e-5)

    print("test_observer passed")

if __name__ == "__main__":
    # test_ego_dataset(scenario1)
    # test_ego_dataset(scenario2)
    test_relative_dataset(scenario1)
    # test_observer(scenario1)