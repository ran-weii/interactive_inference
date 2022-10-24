import os
import re
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, random_split

def load_data(data_path, scenario, filename, train=True, load_raw=True):
    """ Load processed data 
    
    Args:
        data_path (str): root data path. e.g. ../interaction-dataset-master
        scenario (str): scenario name. e.g. DR_CHN_Merging_ZS
        filename (str): track file name. e.g. vehicle_tracks_007.csv
        train (bool, optional): whether in training mode. If False flip is_train labels. Default=True
        load_raw (bool, optional): whether to load the raw track file. Default=True
    
    Returns:
        df_track (pd.dataframe): concatenated processed track data
    """

    df_features = pd.read_csv(
        os.path.join(data_path, "processed_trackfiles", "features", scenario, filename)
    )
    df_labels = pd.read_csv(
        os.path.join(data_path, "processed_trackfiles", "train_labels", scenario, filename)
    ).drop(columns=["track_id", "frame_id"])

    df_track = [df_features, df_labels]

    if load_raw:
        df_raw = pd.read_csv(
            os.path.join(data_path, "recorded_trackfiles", scenario, filename)
        ).drop(columns=["track_id", "frame_id"])
        df_raw["psi_rad"] = np.clip(df_raw["psi_rad"], -np.pi, np.pi)
        df_track.append(df_raw)

    df_track = pd.concat(df_track, axis=1)
    
    if not train:
        df_track["is_train"] = 1 - df_track["is_train"]
        
    # add scenario and record id
    record_id = re.compile(r"\d\d\d").search(filename).group()
    df_track.insert(0, "scenario", scenario)
    df_track.insert(0, "record_id", record_id)
    return df_track

def train_test_split(dataset, train_ratio, batch_size, collate_fn=None, seed=0):
    gen = torch.Generator()
    gen.manual_seed(seed)
    
    train_size = np.ceil(train_ratio * len(dataset)).astype(int)
    test_size = len(dataset) - train_size
    
    train_set, test_set = random_split(
        dataset, [train_size, test_size], generator=gen
    )
    
    train_loader = DataLoader(
        train_set, batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False
    )
    return train_loader, test_loader

def count_parameters(model):
    """ Count active parameters in model """
    num_params = 0
    for n, p in model.named_parameters():
        if p.requires_grad == True:
            num_params += np.prod(list(p.shape))
    return num_params