import argparse
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 

# setup imports
from src.data.train_utils import load_data, update_record_id
from src.data.train_utils import train_test_split, count_parameters
from src.data.data_filter import filter_segment_by_length
from src.data.ego_dataset import RelativeDataset

# model imports
from src.agents.rule_based import IDM
from src.agents.nn_agents import MLPAgent
from src.algo.bc import BehaviorCloning

# training imports
from src.algo.utils import train, SaveCallback

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    str_list_ = lambda x: x.replace(" ", "").split(",")
    
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument(
        "--data_path", type=str, default="../interaction-dataset-master"
    )
    parser.add_argument("--exp_path", type=str, default="../exp")
    parser.add_argument("--scenario", type=str, default="DR_CHN_Merging_ZS")
    parser.add_argument("--filenames", type=str_list_, default=["vehicle_tracks_007.csv"])
    parser.add_argument("--checkpoint_path", type=str, default="none", 
        help="if entered train agent from check point")
    parser.add_argument("--valid_lanes", type=str_list_, default=["3", "4"])
    parser.add_argument("--feature_set", type=str_list_, default=["ego_ds", "lv_s_rel", "lv_ds_rel"], help="agent feature set")
    parser.add_argument("--action_set", type=str_list_, default="dds", help="agent action set, default=dds")
    # # agent args
    parser.add_argument("--agent", type=str, choices=["idm", "mlp"], default="mlp", help="agent type, default=mlp")
    parser.add_argument("--act_dim", type=int, default=15, help="agent action dimension, default=15")
    parser.add_argument("--hidden_dim", type=int, default=64, help="nn hidden dimension, default=64")
    parser.add_argument("--num_hidden", type=int, default=2, help="number of hidden layers, default=2")
    parser.add_argument("--activation", type=str, default="relu", help="nn activation, default=relu")
    parser.add_argument("--norm_obs", type=bool_, default=False, help="whether to normalize observation for discriminator, default=False")
    # algo args
    parser.add_argument("--train_mode", type=str, choices=["prior", "post", "marginal"], help="training mode for hbc")
    # training args
    parser.add_argument("--min_eps_len", type=int, default=50, help="min track length, default=50")
    parser.add_argument("--max_eps_len", type=int, default=200, help="max track length, default=200")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="ratio of training dataset, default=0.7")
    parser.add_argument("--batch_size", type=int, default=64, help="training batch size, default=64")
    parser.add_argument("--epochs", type=int, default=3, help="number of training epochs, default=10")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate, default=0.01")
    parser.add_argument("--decay", type=float, default=1e-5, help="weight decay, default=0")
    parser.add_argument("--grad_clip", type=float, default=None, help="gradient clipping, default=None")
    parser.add_argument("--cp_every", type=int, default=1000, help="checkpoint interval, default=1000")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def main(arglist):
    torch.manual_seed(arglist.seed)
    
    # load data files
    df_track = []
    for filename in arglist.filenames:
        df_track.append(load_data(arglist.data_path, arglist.scenario, filename, load_raw=False))
    df_track = pd.concat(df_track, axis=0)
    df_track = update_record_id(df_track)
    
    # filter invalid lanes
    valid_lanes = [int(l) for l in arglist.valid_lanes]
    is_train = df_track["is_train"].values
    is_train[df_track["ego_lane_id"].isin(valid_lanes) == False] = np.nan
    df_track = df_track.assign(is_train=is_train)

    # filter episode length
    eps_id, df_track["eps_len"] = filter_segment_by_length(
        df_track["eps_id"].values, arglist.min_eps_len
    )
    df_track = df_track.assign(eps_id=eps_id.astype(float))    

    df_track = df_track.loc[(df_track["is_train"] == 1) & (df_track["eps_id"] != np.nan)]
    
    feature_set = arglist.feature_set
    action_set = arglist.action_set
    
    # compute obs and ctl mean and variance stats
    obs_mean = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][feature_set].mean().values).to(torch.float32)
    obs_var = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][feature_set].var().values).to(torch.float32)
    
    # init dataset
    dataset = RelativeDataset(
        df_track, feature_set, action_set, train_labels_col="is_train",
        max_eps_len=arglist.max_eps_len, max_eps=10000, state_action=True, seed=arglist.seed
    )
    train_loader, test_loader = train_test_split(
        dataset, arglist.train_ratio, arglist.batch_size, 
        collate_fn=None, seed=arglist.seed
    )
    obs_dim, ctl_dim = len(feature_set), len(action_set)

    print(f"feature set: {feature_set}")
    print(f"action set: {action_set}")
    print(f"train size: {len(train_loader.dataset)}, test size: {len(test_loader.dataset)}")
    
    # init agent
    if arglist.agent == "idm":
        agent = IDM(feature_set)

    elif arglist.agent == "mlp":
        agent = MLPAgent(
            obs_dim, ctl_dim, arglist.act_dim, arglist.hidden_dim, 
            arglist.num_hidden, arglist.activation
        )
    
        # preload stats
        if arglist.norm_obs:
            agent.obs_mean.data = obs_mean
            agent.obs_variance.data = obs_var

        if (arglist.action_set == ["dds"] or arglist.action_set == ["dds_smooth"]):
            # load ctl gmm parameters
            with open(os.path.join(arglist.exp_path, "agents", "ctl_model", "model.p"), "rb") as f:
                [ctl_means, ctl_covs, weights] = pickle.load(f)

            agent.ctl_model.init_params(ctl_means, ctl_covs)
            print("action model loaded")

    # init trainer
    model = BehaviorCloning(
        agent, lr=arglist.lr, decay=arglist.decay, grad_clip=arglist.grad_clip
    )
    
    print(f"num parameters: {count_parameters(model)}")
    print(model)

    # load from check point
    cp_history = None
    if arglist.checkpoint_path != "none":
        cp_path = os.path.join(
            arglist.exp_path, "agents", 
            arglist.agent, arglist.checkpoint_path
        )
        # load state dict
        state_dict = torch.load(os.path.join(cp_path, "model.pt"))

        # load history
        cp_history = pd.read_csv(os.path.join(cp_path, "history.csv"))

        model.load_state_dict(state_dict, strict=False)
        print(f"loaded checkpoint from {cp_path}")
    
    callback = None
    if arglist.save:
        callback = SaveCallback(arglist, cp_history)

    model, df_history = train(
        model, train_loader, test_loader, arglist.epochs, callback=callback, verbose=1
    )
    if arglist.save:
        callback.save_model(model)
        callback.save_history(model, df_history)
    
if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)