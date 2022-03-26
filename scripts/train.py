import argparse
import os
import json
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch 
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from src.data.lanelet import load_lanelet_df
from src.data.ego_dataset import RelativeDataset
from src.irl.algorithms import MLEIRL, BayesianIRL, ImitationLearning

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../interaction-dataset-master"
    )
    parser.add_argument(
        "--lanelet_path", type=str, default="../exp/lanelet"
    )
    parser.add_argument(
        "--save_path", type=str, default="../exp/"
    )
    parser.add_argument("--scenario", type=str, default="DR_CHN_Merging_ZS")
    parser.add_argument("--filename", type=str, default="vehicle_tracks_007.csv")
    # agent args
    parser.add_argument("--state_dim", type=int, default=10, help="agent state dimension, default=10")
    parser.add_argument("--act_dim", type=int, default=5, help="agent action dimension, default=5")
    parser.add_argument("--horizon", type=int, default=10, help="agent max planning horizon, default=10")
    parser.add_argument("--obs_dist", type=str, default="mvn", help="agent observation distribution, default=mvn")
    parser.add_argument("--obs_cov", type=str, default="diag", help="agent observation covariance, default=diag")
    parser.add_argument("--ctl_dist", type=str, default="mvn", help="agent control distribution, default=mvn")
    parser.add_argument("--ctl_cov", type=str, default="diag", help="agent control covariance, default=diag")
    parser.add_argument("--method", type=str, choices=["mleirl", "birl", "il"], 
        default="active_inference", help="algorithm, default=mleirl")
    parser.add_argument("--lateral_control", type=bool_, default=True, help="predict lateral control, default=True")
    # training args
    parser.add_argument("--min_eps_len", type=int, default=50, help="min track length, default=50")
    parser.add_argument("--max_eps_len", type=int, default=100, help="max track length, default=200")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="ratio of training dataset, default=0.7")
    parser.add_argument("--batch_size", type=int, default=64, help="training batch size, default=64")
    parser.add_argument("--epochs", type=int, default=3, help="number of training epochs, default=10")
    parser.add_argument("--obs_penalty", type=float, default=0, help="observation likelihood penalty, default=0")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate, default=1e-3")
    parser.add_argument("--decay", type=float, default=0, help="weight decay, default=0")
    parser.add_argument("--grad_clip", type=float, default=None, help="gradient clipping, default=None")
    parser.add_argument("--plot_history", type=bool_, default=True, help="plot em learning curve, default=True")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def collate_fn(batch):
    pad_obs = pad_sequence([b["ego"] for b in batch])
    pad_act = pad_sequence([b["act"] for b in batch])
    mask = torch.all(pad_obs != 0, dim=-1).to(torch.float32)
    return pad_obs, pad_act, mask

def train_test_split(dataset, train_ratio, batch_size, seed):
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

def plot_history(df_history):
    df_train = df_history.loc[df_history["train"] == "train"]
    df_test = df_history.loc[df_history["train"] == "test"]
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(df_train["epoch"], df_train["logp_pi_mean"], label="train")
    ax[0].plot(df_test["epoch"], df_test["logp_pi_mean"], label="test")
    ax[0].legend()
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("logp_u")
    ax[0].grid()
    
    ax[1].plot(df_train["epoch"], df_train["logp_obs_mean"], label="train")
    ax[1].plot(df_test["epoch"], df_test["logp_obs_mean"], label="test")
    ax[1].legend()
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("logp_o")
    ax[1].grid()
    
    plt.tight_layout()
    return fig, ax

def main(arglist):
    torch.manual_seed(arglist.seed)
    
    # load raw data
    df_lanelet = load_lanelet_df(os.path.join(arglist.lanelet_path, arglist.scenario + ".json"))
    df_processed = pd.read_csv(
        os.path.join(arglist.data_path, "processed_trackfiles", arglist.scenario, arglist.filename)
    )
    df_track = pd.read_csv(
        os.path.join(arglist.data_path, "recorded_trackfiles", arglist.scenario, arglist.filename)
    )
    df_track = df_track.merge(df_processed, on=["track_id", "frame_id"])
    df_track["psi_rad"] = np.clip(df_track["psi_rad"], -np.pi, np.pi)
    
    # load test labels
    merge_keys = ["scenario", "record_id", "track_id"]
    df_train_labels = pd.read_csv(
        os.path.join(arglist.data_path, "test_labels", arglist.scenario, arglist.filename)
    )
    
    dataset = RelativeDataset(
        df_track, df_lanelet, df_train_labels=df_train_labels,
        min_eps_len=arglist.min_eps_len, max_eps_len=arglist.max_eps_len,
        lateral_control=arglist.lateral_control
    )
    train_loader, test_loader = train_test_split(
        dataset, arglist.train_ratio, arglist.batch_size, arglist.seed
    )
    
    # get dimensions from data 
    [obs, ctl, mask] = next(iter(train_loader))
    obs_dim = obs.shape[-1]
    ctl_dim = ctl.shape[-1]
    if not arglist.lateral_control:
        ctl_dim = 1
    
    # init model
    if arglist.method == "mleirl":
        model = MLEIRL(
            arglist.state_dim, arglist.act_dim, obs_dim, ctl_dim, arglist.horizon,
            obs_dist=arglist.obs_dist, obs_cov=arglist.obs_cov, 
            ctl_dist=arglist.ctl_dist, ctl_cov=arglist.ctl_cov,
            obs_penalty=arglist.obs_penalty, lr=arglist.lr, 
            decay=arglist.decay, grad_clip=arglist.grad_clip
        )
    elif arglist.method == "birl":
        model = BayesianIRL(
            arglist.state_dim, arglist.act_dim, obs_dim, ctl_dim, arglist.horizon,
            obs_dist=arglist.obs_dist, obs_cov=arglist.obs_cov, 
            ctl_dist=arglist.ctl_dist, ctl_cov=arglist.ctl_cov,
            obs_penalty=arglist.obs_penalty, lr=arglist.lr,
            decay=arglist.decay, grad_clip=arglist.grad_clip
        )
    elif arglist.method == "il":
        model = ImitationLearning(
            arglist.act_dim, obs_dim, ctl_dim, 
            obs_penalty=arglist.obs_penalty, lr=arglist.lr, 
            decay=arglist.decay, grad_clip=arglist.grad_clip
        )
    else:
        raise NotImplementedError
    
    print(model)
    
    start_time = time.time()
    history = []
    for e in range(arglist.epochs):
        train_stats = model.train_epoch(train_loader)
        test_stats = model.test_epoch(test_loader)
        tnow = time.time() - start_time
        
        history.append(
            pd.Series({"epoch": e+1, "time": tnow, "train": "train"}).append(train_stats)
        )
        history.append(
            pd.Series({"epoch": e+1, "time": tnow, "train": "test"}).append(test_stats)
        )
        
        print("epoch: {}, train, logp_pi: {:.4f}, logp_obs: {:.4f}, t: {:.2f}".format(
            e + 1, train_stats["logp_pi_mean"], train_stats["logp_obs_mean"], tnow
        ))
    
    df_history = pd.DataFrame(history)
    
    # save results
    if arglist.save:
        date_time = datetime.datetime.now().strftime("%m-%d-%Y %H-%M-%S")
        exp_path = os.path.join(arglist.save_path, arglist.method)
        save_path = os.path.join(exp_path, date_time)
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        # save args
        with open(os.path.join(save_path, "args.json"), "w") as f:
            json.dump(vars(arglist), f)
        
        # save model
        torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
        
        # save history
        df_history.to_csv(os.path.join(save_path, "history.csv"), index=False)
        
        # save history plot
        fig_history, _ = plot_history(df_history)
        fig_history.savefig(os.path.join(save_path, "history.png"), dpi=100)
        
        print(f"\nmodel saved at: ./exp/{arglist.method}/{date_time}")
        

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)