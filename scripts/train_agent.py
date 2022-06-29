import argparse
import os
import json
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 

from src.simulation.observers import FEATURE_SET
from src.data.train_utils import load_data, train_test_split, count_parameters
from src.data.ego_dataset import RelativeDataset, aug_flip_lr, collate_fn
from src.agents.vin_agents import VINAgent
# from src.irl.algorithms import 
from src.visualization.utils import plot_history

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../interaction-dataset-master"
    )
    parser.add_argument("--exp_path", type=str, default="../exp")
    parser.add_argument("--scenario", type=str, default="DR_CHN_Merging_ZS")
    parser.add_argument("--filename", type=str, default="vehicle_tracks_007.csv")
    # agent args
    parser.add_argument("--state_dim", type=int, default=30, help="agent state dimension, default=30")
    parser.add_argument("--act_dim", type=int, default=45, help="agent action dimension, default=45")
    parser.add_argument("--horizon", type=int, default=30, help="agent planning horizon, default=30")
    parser.add_argument("--obs_cov", type=str, default="full", help="agent observation covariance, default=full")
    parser.add_argument("--ctl_cov", type=str, default="full", help="agent control covariance, default=full")
    parser.add_argument("--hmm_rank", type=int, default=0, help="agent hmm rank, 0 for full rank, default=0")
    parser.add_argument("--agent", type=str, choices=["vin"], default="vin", help="agent type, default=vin")
    parser.add_argument("--dynamics_path", type=str, default="none", help="pretrained dynamics path, default=none")
    parser.add_argument("--train_dynamics", type=bool_, default=True, help="whether to train dynamics, default=True")
    # training args
    parser.add_argument("--algo", type=str, choices=["il"], default="il", help="training algorithm, default=il")
    parser.add_argument("--min_eps_len", type=int, default=50, help="min track length, default=50")
    parser.add_argument("--max_eps_len", type=int, default=200, help="max track length, default=200")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="ratio of training dataset, default=0.7")
    parser.add_argument("--batch_size", type=int, default=64, help="training batch size, default=64")
    parser.add_argument("--epochs", type=int, default=3, help="number of training epochs, default=10")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate, default=0.01")
    parser.add_argument("--decay", type=float, default=1e-5, help="weight decay, default=0")
    parser.add_argument("--grad_clip", type=float, default=None, help="gradient clipping, default=None")
    parser.add_argument("--plot_history", type=bool_, default=True, help="plot learning curve, default=True")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def main(arglist):
    torch.manual_seed(arglist.seed)
    
    df_track = load_data(arglist.data_path, arglist.scenario, arglist.filename)
    df_track = df_track.loc[df_track["is_train"] == 1]
    
    feature_set = [
        "d", "ds", "dd", "kappa_r", "psi_error_r", 
        "s_rel", "d_rel", "ds_rel", "dd_rel", "loom_s"
    ]
    assert set(feature_set).issubset(set(FEATURE_SET["ego"] + FEATURE_SET["relative"]))
    dataset = RelativeDataset(
        df_track, feature_set, train_labels_col="is_train",
        min_eps_len=arglist.min_eps_len, max_eps_len=arglist.max_eps_len,
        augmentation=[aug_flip_lr]
    )
    train_loader, test_loader = train_test_split(
        dataset, arglist.train_ratio, arglist.batch_size, 
        collate_fn=collate_fn, seed=arglist.seed
    )
    obs_dim, ctl_dim = len(feature_set), 2 

    print(f"feature set: {feature_set}")
    print(f"train size: {len(train_loader.dataset)}, test size: {len(test_loader.dataset)}")
    
    # init agent
    if arglist.agent == "vin":
        agent = VINAgent(
            arglist.state_dim, arglist.act_dim, obs_dim, ctl_dim, arglist.horizon,
            arglist.hmm_rank, arglist.obs_cov, arglist.ctl_cov
        )
    
    # load dynamics
    if arglist.dynamics_path != "none":
        state_dict = torch.load(os.path.join(
            arglist.exp_path, "dynamics", arglist.dynamics_path, "model.pt"
        ))
        agent.load_dynamics_model(state_dict, requires_grad=arglist.train_dynamics)
    
    # print(f"num parameters: {count_parameters(model)}")
    # print(model)

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)