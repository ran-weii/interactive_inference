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
from src.data.train_utils import load_data, train_test_split
from src.data.ego_dataset import RelativeDataset, aug_flip_lr, collate_fn
from src.agents.legacy.active_inference import ActiveInference
from src.agents.baseline import FullyRecurrentAgent
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
    parser.add_argument("--obs_model", type=str, choices=["gmm"], default="gmm", help="agent observation model class, default=gmm")
    parser.add_argument("--obs_dist", type=str, default="mvn", help="agent observation distribution, default=mvn")
    parser.add_argument("--obs_cov", type=str, default="diag", help="agent observation covariance, default=diag")
    parser.add_argument("--ctl_model", type=str, choices=["gmm", "glm"], default="gmm", help="agent control model class, default=gmm")
    parser.add_argument("--ctl_dist", type=str, default="mvn", help="agent control distribution, default=mvn")
    parser.add_argument("--ctl_cov", type=str, default="diag", help="agent control covariance, default=diag")
    parser.add_argument("--rwd_model", type=str, choices=["efe", "gfe"], default="efe", help="agent reward model, default=efe")
    parser.add_argument("--hmm_rank", type=int, default=0, help="agent hmm rank, 0 for full rank, default=0")
    parser.add_argument("--planner", type=str, choices=["qmdp", "mcvi"], default="qmdp", help="agent planner, default=qmdp")
    parser.add_argument("--tau", type=float, default=0.9, help="agent planner discount, default=0.9")
    parser.add_argument("--hidden_dim", type=int, default=32, help="neural network hidden dim, default=32")
    parser.add_argument("--num_hidden", type=int, default=2, help="neural network hidden layers, default=2")
    parser.add_argument("--activation", type=str, default="relu", help="neural network activation, default=relu")
    parser.add_argument("--method", type=str, choices=["mleirl", "birl", "il", "frnn"], 
        default="active_inference", help="algorithm, default=mleirl")
    parser.add_argument("--frame", type=str, choices=["frenet", "carte"], 
        default="rel", help="ego coordinate frame, one of [frenet, carte], default=frenet")
    parser.add_argument("--obs_fields", type=str, choices=["rel", "two_point", "lv_fv", "full"], 
        default="rel", help="ego observation fields, default=rel")
    # training args
    parser.add_argument("--min_eps_len", type=int, default=50, help="min track length, default=50")
    parser.add_argument("--max_eps_len", type=int, default=100, help="max track length, default=200")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="ratio of training dataset, default=0.7")
    parser.add_argument("--batch_size", type=int, default=64, help="training batch size, default=64")
    parser.add_argument("--epochs", type=int, default=3, help="number of training epochs, default=10")
    parser.add_argument("--obs_penalty", type=float, default=0, help="observation likelihood penalty, default=0")
    parser.add_argument("--plan_penalty", type=float, default=0, help="planner loss penalty, default=0")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate, default=1e-3")
    parser.add_argument("--decay", type=float, default=0, help="weight decay, default=0")
    parser.add_argument("--grad_clip", type=float, default=None, help="gradient clipping, default=None")
    parser.add_argument("--plot_history", type=bool_, default=True, help="plot em learning curve, default=True")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def plot_history(df_history, keys):
    df_train = df_history.loc[df_history["train"] == "train"]
    df_test = df_history.loc[df_history["train"] == "test"]
    
    num_cols = len(keys)
    width = min(4 * num_cols, 15)
    fig, ax = plt.subplots(1, num_cols, figsize=(width, 4))
    for i in range(num_cols):
        ax[i].plot(df_train["epoch"], df_train[keys[i]], label="train")
        ax[i].plot(df_test["epoch"], df_test[keys[i]], label="test")
        ax[i].legend()
        ax[i].set_xlabel("epoch")
        ax[i].set_ylabel(keys[i])
        ax[i].grid()
    
    plt.tight_layout()
    return fig, ax

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
    
    # init model
    if arglist.method == "mleirl":
        agent = ActiveInference(
            arglist.state_dim, arglist.act_dim, obs_dim, ctl_dim, arglist.horizon,
            obs_model=arglist.obs_model, obs_dist=arglist.obs_dist, obs_cov=arglist.obs_cov, 
            ctl_model=arglist.ctl_model, ctl_dist=arglist.ctl_dist, ctl_cov=arglist.ctl_cov,
            rwd_model=arglist.rwd_model, hmm_rank=arglist.hmm_rank, planner=arglist.planner, tau=arglist.tau, hidden_dim=arglist.hidden_dim, 
            num_hidden=arglist.num_hidden, activation=arglist.activation
        )
        model = MLEIRL(
            agent, obs_penalty=arglist.obs_penalty, plan_penalty=arglist.plan_penalty, 
            lr=arglist.lr, decay=arglist.decay, grad_clip=arglist.grad_clip
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
    elif arglist.method == "frnn":
        agent = FullyRecurrentAgent(
            arglist.state_dim, arglist.act_dim, obs_dim, ctl_dim, arglist.horizon, 
            ctl_dist=arglist.ctl_dist, ctl_cov=arglist.ctl_cov,
            hidden_dim=arglist.hidden_dim, num_hidden=arglist.num_hidden
        )
        model = ImitationLearning(
            agent,
            obs_penalty=arglist.obs_penalty, lr=arglist.lr, 
            decay=arglist.decay, grad_clip=arglist.grad_clip
        )
    else:
        raise NotImplementedError
    
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
        
        print("epoch: {}, train, logp_pi: {:.4f}, logp_obs: {:.4f}, plan_error: {:.4f}, t: {:.2f}".format(
            e + 1, 
            train_stats["logp_pi_mean"], 
            train_stats["logp_obs_mean"], 
            train_stats["loss_plan_mean"] if "loss_plan" in train_stats.keys() else 0, 
            tnow
        ))
        print("epoch: {}, test , logp_pi: {:.4f}, logp_obs: {:.4f}, plan_error: {:.4f}, t: {:.2f} \n{}".format(
            e + 1, 
            test_stats["logp_pi_mean"], 
            test_stats["logp_obs_mean"], 
            test_stats["loss_plan_mean"] if "loss_plan" in test_stats.keys() else 0, 
            tnow, "=="*40
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
        fig_history, _ = plot_history(df_history, model.loss_keys)
        fig_history.savefig(os.path.join(save_path, "history.png"), dpi=100)
        
        print(f"\nmodel saved at: ./exp/{arglist.method}/{date_time}")
        

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)