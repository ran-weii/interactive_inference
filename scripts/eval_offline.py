import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.simulation.observers import FEATURE_SET
from src.data.train_utils import load_data, train_test_split, count_parameters
from src.data.ego_dataset import RelativeDataset, aug_flip_lr, collate_fn
from src.agents.vin_agents import VINAgent
from src.algo.irl import BehaviorCloning
from src.visualization.utils import set_plotting_style

import warnings
warnings.filterwarnings("ignore")

set_plotting_style()

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../interaction-dataset-master"
    )
    parser.add_argument("--exp_path", type=str, default="../exp")
    parser.add_argument("--scenario", type=str, default="DR_CHN_Merging_ZS")
    parser.add_argument("--filename", type=str, default="vehicle_tracks_007.csv")
    parser.add_argument("--agent", type=str, choices=["vin"], default="vin", 
        help="agent type, default=vin")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def main(arglist):
    torch.manual_seed(arglist.seed)
    exp_path = os.path.join(arglist.exp_path, "agents", arglist.agent, arglist.exp_name)
    print(f"evalusting offline exp: {arglist.agent}/{arglist.exp_name}")
    
    df_track = load_data(arglist.data_path, arglist.scenario, arglist.filename)
    df_track["is_train"] = 1 - df_track["is_train"]
    df_track = df_track.loc[df_track["is_train"] == 1]
    
    """ TODO: add code to adapt input feature set """
    feature_set = [
        "d", "ds", "dd", "kappa_r", "psi_error_r", 
        "s_rel", "d_rel", "ds_rel", "dd_rel", "loom_s"
    ]
    assert set(feature_set).issubset(set(FEATURE_SET["ego"] + FEATURE_SET["relative"]))
    dataset = RelativeDataset(
        df_track, feature_set, train_labels_col="is_train", augmentation=[aug_flip_lr]
    )
    loader = DataLoader(dataset, len(dataset), shuffle=False, collate_fn=collate_fn)
    obs_dim, ctl_dim = len(feature_set), 2 

    print(f"feature set: {feature_set}")
    print(f"test size: {len(loader.dataset)}")
    
    # load config 
    with open(os.path.join(exp_path, "args.json"), "r") as f:
        config = json.load(f)
    
    # init agent
    if arglist.agent == "vin":
        agent = VINAgent(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["horizon"],
            hmm_rank=config["hmm_rank"], obs_cov=config["obs_cov"], ctl_cov=config["ctl_cov"]
        )

    # init model
    if config["algo"] == "bc":
        model = BehaviorCloning(agent)
    
    # load state dict
    state_dict = torch.load(os.path.join(exp_path, "model.pt"))
    model.load_state_dict(state_dict)
    agent = model.agent
    print(agent)


if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)