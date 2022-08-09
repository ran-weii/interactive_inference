import argparse
import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 

# setup imports
from src.simulation.observers import ACTION_SET, FEATURE_SET
from src.data.train_utils import load_data, train_test_split, count_parameters
from src.data.ego_dataset import RelativeDataset, aug_flip_lr, collate_fn

# model imports
from src.agents.vin_agent import VINAgent
from src.agents.hyper_vin_agent import HyperVINAgent
from src.agents.rule_based import IDM
from src.agents.mlp_agents import MLPAgent
from src.algo.irl import BehaviorCloning, HyperBehaviorCloning
from src.algo.irl import ReverseBehaviorCloning

# training imports
from src.algo.utils import train
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
    parser.add_argument("--checkpoint_path", type=str, default="none", 
        help="if entered train agent from check point")
    # agent args
    parser.add_argument("--agent", type=str, choices=["vin", "hvin", "idm", "mlp"], default="vin", help="agent type, default=vin")
    parser.add_argument("--state_dim", type=int, default=30, help="agent state dimension, default=30")
    parser.add_argument("--act_dim", type=int, default=45, help="agent action dimension, default=45")
    parser.add_argument("--horizon", type=int, default=30, help="agent planning horizon, default=30")
    parser.add_argument("--obs_cov", type=str, default="full", help="agent observation covariance, default=full")
    parser.add_argument("--ctl_cov", type=str, default="full", help="agent control covariance, default=full")
    parser.add_argument("--hmm_rank", type=int, default=32, help="agent hmm rank, default=32")
    parser.add_argument("--action_set", type=str, choices=["ego", "frenet", "disc"], default="frenet", help="agent action set, default=frenet")
    parser.add_argument("--discrete_action", type=bool_, default=False)
    parser.add_argument("--use_tanh", type=bool_, default=True, help="whether to use tanh transform, default=True")
    parser.add_argument("--hyper_dim", type=int, default=4, help="number of latent factor, default=4")
    # nn args
    parser.add_argument("--hidden_dim", type=int, default=64, help="nn hidden dimension, default=64")
    parser.add_argument("--num_hidden", type=int, default=2, help="number of hidden layers, default=2")
    parser.add_argument("--gru_layers", type=int, default=1, help="number of gru layers, default=1")
    parser.add_argument("--activation", type=str, default="relu", help="nn activation, default=relu")
    parser.add_argument("--norm_obs", type=bool_, default=False, help="whether to normalize observation for discriminator, default=False")
    parser.add_argument("--use_state", type=bool_, default=False, help="whether to use state for discriminator, default=False")
    # algo args
    parser.add_argument("--algo", type=str, choices=["bc", "hbc", "rbc"], default="bc", help="training algorithm, default=bc")
    parser.add_argument("--d_batch_size", type=int, default=200, help="discriminator batch size, default=200")
    parser.add_argument("--bptt_steps", type=int, default=30, help="bptt truncation steps, default=30")
    parser.add_argument("--d_steps", type=int, default=50, help="discriminator steps, default=50")
    parser.add_argument("--grad_penalty", type=float, default=1., help="discriminator gradient penalty, default=1.")
    parser.add_argument("--bc_penalty", type=float, default=0., help="behavior cloning penalty, default=0.")
    parser.add_argument("--obs_penalty", type=float, default=0., help="observation penalty, default=0.")
    # training args
    parser.add_argument("--min_eps_len", type=int, default=50, help="min track length, default=50")
    parser.add_argument("--max_eps_len", type=int, default=200, help="max track length, default=200")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="ratio of training dataset, default=0.7")
    parser.add_argument("--batch_size", type=int, default=64, help="training batch size, default=64")
    parser.add_argument("--epochs", type=int, default=3, help="number of training epochs, default=10")
    parser.add_argument("--lr_d", type=float, default=0.001, help="discriminator learning rate, default0.001")
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
    
    """ TODO: add code to adapt input feature set """
    # define feature set
    ego_features = ["d", "ds", "dd", "kappa_r", "psi_error_r", ]
    relative_features = ["s_rel", "d_rel", "ds_rel", "dd_rel", "loom_s"]
    feature_set = ego_features + relative_features
    assert set(ego_features).issubset(set(FEATURE_SET["ego"]))
    assert set(relative_features).issubset(set(FEATURE_SET["relative"]))
    
    # define action set
    if arglist.action_set == "frenet":
        action_set = ["dds", "ddd"]
    elif arglist.action_set == "ego":
        action_set = ["ax_ego", "ay_ego"]
    
    action_bins = None
    if arglist.discrete_action:
        action_bins = arglist.act_dim

    assert set(action_set).issubset(set(ACTION_SET))
    
    # compute obs and ctl mean and variance stats
    obs_mean = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][feature_set].mean().values).to(torch.float32)
    obs_var = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][feature_set].var().values).to(torch.float32)
    ctl_mean = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].mean().values).to(torch.float32)
    ctl_var = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].var().values).to(torch.float32)
    
    ctl_max = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].max().values).to(torch.float32)
    ctl_min = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].min().values).to(torch.float32)
    ctl_lim = torch.max(torch.abs(ctl_max), torch.abs(ctl_min)) * 1.2
    
    dataset = RelativeDataset(
        df_track, feature_set, action_set, action_bins=action_bins, train_labels_col="is_train",
        max_eps_len=arglist.max_eps_len, augmentation=[aug_flip_lr], seed=arglist.seed
    )
    train_loader, test_loader = train_test_split(
        dataset, arglist.train_ratio, arglist.batch_size, 
        collate_fn=collate_fn, seed=arglist.seed
    )
    obs_dim, ctl_dim = len(feature_set), 2 

    print(f"feature set: {feature_set}")
    print(f"action set: {action_set}")
    print(f"train size: {len(train_loader.dataset)}, test size: {len(test_loader.dataset)}")

    # init agent
    if arglist.agent == "vin":
        agent = VINAgent(
            arglist.state_dim, arglist.act_dim, obs_dim, ctl_dim, arglist.hmm_rank,
            arglist.horizon, obs_cov=arglist.obs_cov, ctl_cov=arglist.ctl_cov, 
            use_tanh=arglist.use_tanh, ctl_lim=ctl_lim
        )
        agent.obs_model.init_batch_norm(obs_mean, obs_var)
        if not arglist.use_tanh:
            agent.ctl_model.init_batch_norm(ctl_mean, ctl_var)

    elif arglist.agent == "hvin":
        agent = HyperVINAgent(
            arglist.state_dim, arglist.act_dim, obs_dim, ctl_dim, arglist.hmm_rank,
            arglist.horizon, arglist.hyper_dim, arglist.hidden_dim, arglist.num_hidden, 
            arglist.gru_layers, arglist.activation,
            obs_cov=arglist.obs_cov, ctl_cov=arglist.ctl_cov, 
            use_tanh=arglist.use_tanh, ctl_lim=ctl_lim
        )
        agent.obs_model.init_batch_norm(obs_mean, obs_var)
        if not arglist.use_tanh:
            agent.ctl_model.init_batch_norm(ctl_mean, ctl_var)

    elif arglist.agent == "idm":
        agent = IDM(std=ctl_var)

    elif arglist.agent == "mlp":
        agent = MLPAgent(
            obs_dim, ctl_dim, arglist.hidden_dim, arglist.num_hidden, 
            activation=arglist.activation, use_tanh=arglist.use_tanh, ctl_limits=ctl_lim
        )
    
    # init trainer
    if arglist.algo == "bc":
        model = BehaviorCloning(
            agent, arglist.bptt_steps, arglist.obs_penalty, lr=arglist.lr, 
            decay=arglist.decay, grad_clip=arglist.grad_clip
        )
    elif arglist.algo == "hbc":
        model = HyperBehaviorCloning(
            agent, arglist.bptt_steps, arglist.obs_penalty, lr=arglist.lr, 
            decay=arglist.decay, grad_clip=arglist.grad_clip
        )
    elif arglist.algo == "rbc":
        model = ReverseBehaviorCloning(
            agent, arglist.hidden_dim, arglist.num_hidden, arglist.activation,
            norm_obs=arglist.norm_obs, use_state=arglist.use_state, 
            d_batch_size=arglist.d_batch_size, bptt_steps=arglist.bptt_steps, 
            d_steps=arglist.d_steps, grad_target=0., grad_penalty=arglist.grad_penalty,
            bc_penalty=arglist.bc_penalty, obs_penalty=arglist.obs_penalty,
            lr_d=arglist.lr_d, lr_a=arglist.lr, decay=arglist.decay, grad_clip=arglist.grad_clip
        )
        if arglist.norm_obs:
            model.obs_mean.data = obs_mean
            model.obs_variance.data = obs_var
        model.fill_buffer(train_loader.dataset)
        
    print(f"num parameters: {count_parameters(model)}")
    print(model)

    # load from check point
    if arglist.checkpoint_path != "none":
        cp_path = os.path.join(
            arglist.exp_path, "agents", 
            arglist.agent, arglist.checkpoint_path
        )
        # load state dict
        state_dict = torch.load(os.path.join(cp_path, "model.pt"))

        # load history
        df_history_cp = pd.read_csv(os.path.join(cp_path, "history.csv"))

        model.load_state_dict(state_dict)
        print(f"loaded checkpoint from {cp_path}")
    
    model, df_history = train(
        model, train_loader, test_loader, arglist.epochs, verbose=1
    )
    
    if arglist.checkpoint_path != "none":
        df_history["epoch"] += df_history_cp["epoch"].values[-1] + 1
        df_history["time"] += df_history_cp["time"].values[-1]
        df_history = pd.concat([df_history_cp, df_history], axis=0)

    # save results
    if arglist.save:
        date_time = datetime.datetime.now().strftime("%m-%d-%Y %H-%M-%S")
        exp_path = os.path.join(arglist.exp_path, "agents")
        agent_path = os.path.join(exp_path, arglist.agent)
        save_path = os.path.join(agent_path, date_time)
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        if not os.path.exists(agent_path):
            os.mkdir(agent_path)
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
        
        print(f"\nmodel saved at: {save_path}")
    
if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)