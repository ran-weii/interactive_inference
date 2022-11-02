import os
import pandas as pd
from torch.utils.data import DataLoader
from src.data.ego_dataset import EgoDataset, RelativeDataset
from src.data.ego_dataset import collate_fn
from src.data.train_utils import load_data

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

lanelet_path = "../exp/lanelet"
data_path = "../interaction-dataset-master"
scenario1 = "DR_CHN_Merging_ZS"
scenario2 = "DR_DEU_Merging_MT"
filename = "vehicle_tracks_007.csv"

def test_ego_dataset(scenario):
    df_track = load_data(data_path, scenario, filename)
    
    ego_dataset = EgoDataset(
        df_track, train_labels_col="is_train"
    )
    track_data = ego_dataset[11]

    print("test_ego_dataset passed")

def test_relative_dataset(scenario):
    df_track = load_data(data_path, scenario, filename)
    
    feature_set = ["ego_ds", "lv_s_rel", "lv_ds_rel"]
    action_set = ["dds"]
    max_eps_len = 50
    batch_size = 64
    
    # test recurrent
    rel_dataset = RelativeDataset(
        df_track, feature_set, action_set, train_labels_col="is_train",
        max_eps_len=max_eps_len, state_action=False
    )
    loader = DataLoader(rel_dataset, batch_size=batch_size, collate_fn=collate_fn)
    batch, mask = next(iter(loader))
    
    assert list(batch["obs"].shape) == [max_eps_len, batch_size, len(feature_set)]
    assert list(batch["act"].shape) == [max_eps_len, batch_size, len(action_set)]
    assert list(mask.shape) == [max_eps_len, batch_size]

    # test non recurrent
    rel_dataset = RelativeDataset(
        df_track, feature_set, action_set, train_labels_col="is_train",
        max_eps_len=max_eps_len, state_action=True
    )
    loader = DataLoader(rel_dataset, batch_size=batch_size)
    batch = next(iter(loader))
    
    assert list(batch["obs"].shape) == [batch_size, len(feature_set)]
    assert list(batch["act"].shape) == [batch_size, len(action_set)]

    print("test_relative_dataset passed")

if __name__ == "__main__":
    test_ego_dataset(scenario1)
    test_relative_dataset(scenario1)