import os
import pandas as pd
from src.data.ego_dataset import EgoDataset, SimpleEgoDataset
from src.data.lanelet import load_lanelet_df
from src.visualization.visualizer import build_bokeh_sources, visualize_scene
from bokeh.plotting import show

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

lanelet_path = "../exp/lanelet"
data_path = "../interaction-dataset-master"
scenario = "DR_CHN_Merging_ZS"
filename = "vehicle_tracks_007.csv"

def test_ego_dataset():
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
    show(fig)
    
def test_simple_ego_dataset():
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
    show(fig)

if __name__ == "__main__":
    test_ego_dataset()
    test_simple_ego_dataset()