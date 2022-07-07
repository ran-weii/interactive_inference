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
from src.data.train_utils import load_data, count_parameters
from src.data.data_filter import filter_segment_by_length
from src.data.ego_dataset import EgoDataset, RelativeDataset
from src.map_api.lanelet import MapReader
from src.simulation.simulator import InteractionSimulator
from src.simulation.observers import Observer

# model imports
from src.distributions.hmm import ContinuousGaussianHMM, EmbeddedContinuousGaussianHMM
from src.agents.vin_agents import VINAgent

# training imports
from src.algo.airl import AIRL, train
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
    parser.add_argument("--state_dim", type=int, default=30, help="agent state dimension, default=30")
    parser.add_argument("--act_dim", type=int, default=45, help="agent action dimension, default=45")
    parser.add_argument("--horizon", type=int, default=30, help="agent planning horizon, default=30")
    parser.add_argument("--obs_cov", type=str, default="full", help="agent observation covariance, default=full")
    parser.add_argument("--ctl_cov", type=str, default="full", help="agent control covariance, default=full")
    parser.add_argument("--hmm_rank", type=int, default=0, help="agent hmm rank, 0 for full rank, default=0")
    parser.add_argument("--state_embed_dim", type=int, default=30, help="agent hmm state embedding dimension, default=30")
    parser.add_argument("--act_embed_dim", type=int, default=30, help="agent hmm action embedding dimension, default=30")
    parser.add_argument("--dynamics_model", type=str, choices=["cghmm", "ecghmm"], help="agent dynamics model, default=cghmm")
    parser.add_argument("--agent", type=str, choices=["vin"], default="vin", help="agent type, default=vin")
    parser.add_argument("--action_set", type=str, choices=["ego", "frenet"], default="frenet", help="agent action set, default=frenet")
    parser.add_argument("--dynamics_path", type=str, default="none", help="pretrained dynamics path, default=none")
    parser.add_argument("--train_dynamics", type=bool_, default=True, help="whether to train dynamics, default=True")
    # trainer model args
    parser.add_argument("--hidden_dim", type=int, default=32, help="trainer network hidden dims, default=32")
    parser.add_argument("--gru_layers", type=int, default=2, help="trainer gru layers, default=2")
    parser.add_argument("--mlp_layers", type=int, default=2, help="trainer mlp layers, default=2")
    parser.add_argument("--gamma", type=float, default=0.9, help="trainer discount factor, default=0.9")
    # training args
    parser.add_argument("--algo", type=str, choices=["airl"], default="airl", help="training algorithm, default=airl")
    parser.add_argument("--min_eps_len", type=int, default=50, help="min track length, default=50")
    parser.add_argument("--max_eps_len", type=int, default=200, help="max track length, default=200")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="ratio of training dataset, default=0.7")
    parser.add_argument("--batch_size", type=int, default=64, help="training batch size, default=64")
    parser.add_argument("--num_eps", type=int, default=3, help="number of episodes per epochs, default=10")
    parser.add_argument("--epochs", type=int, default=3, help="number of training epochs, default=10")
    parser.add_argument("--obs_penalty", type=float, default=0., help="observation penalty, default=0.")
    parser.add_argument("--buffer_size", type=int, default=1e5, help="agent replay buffer size, default=1e5")
    parser.add_argument("--rnn_steps", type=int, default=10, help="rnn steps to store, default=10")
    parser.add_argument("--d_steps", type=int, default=10, help="discriminator steps, default=10")
    parser.add_argument("--ac_steps", type=int, default=10, help="actor critic steps, default=10")
    parser.add_argument("--lr", type=float, default=0.01, help="agent learning rate, default=0.01")
    parser.add_argument("--lr_q", type=float, default=0.005, help="value function learning rate, default=0.005")
    parser.add_argument("--lr_d", type=float, default=0.005, help="discriminator learning rate, default=0.005")
    parser.add_argument("--decay", type=float, default=1e-5, help="weight decay, default=0")
    parser.add_argument("--grad_clip", type=float, default=None, help="gradient clipping, default=None")
    parser.add_argument("--polyak", type=float, default=0.995, help="polyak averaging factor, default=0.995")
    parser.add_argument("--plot_history", type=bool_, default=True, help="plot learning curve, default=True")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def main(arglist):
    np.random.seed(arglist.seed)
    torch.manual_seed(arglist.seed)
    
    df_track = load_data(arglist.data_path, arglist.scenario, arglist.filename)

    # filter episode length
    df_track["eps_id"], df_track["eps_len"] = filter_segment_by_length(
        df_track["eps_id"].values, arglist.min_eps_len
    )
    
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
    else:
        action_set = ["ax_ego", "ay_ego"]
    assert set(action_set).issubset(set(ACTION_SET))
    
    # compute obs and ctl mean and variance stats
    obs_mean = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][feature_set].mean().values).to(torch.float32)
    obs_var = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][feature_set].var().values).to(torch.float32)
    ctl_mean = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].mean().values).to(torch.float32)
    ctl_var = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].var().values).to(torch.float32)
    
    ego_dataset = EgoDataset(df_track, train_labels_col="is_train")
    rel_dataset = RelativeDataset(
        df_track, feature_set, action_set, train_labels_col="is_train"
    )
    obs_dim, ctl_dim = len(feature_set), 2 

    print(f"feature set: {feature_set}")
    print(f"action set: {action_set}")
    print(f"data size: {len(ego_dataset)}")
    
    # init dynamics model
    if arglist.dynamics_model == "cghmm":
        dynamics_model = ContinuousGaussianHMM(
            arglist.state_dim, arglist.act_dim, obs_dim, ctl_dim, 
            arglist.hmm_rank, arglist.obs_cov, arglist.ctl_cov
        )
    elif arglist.dynamics_model == "ecghmm":
        dynamics_model = EmbeddedContinuousGaussianHMM(
            arglist.state_dim, arglist.act_dim, obs_dim, ctl_dim, 
            arglist.state_embed_dim, arglist.act_embed_dim,
            arglist.hmm_rank, arglist.obs_cov, arglist.ctl_cov
        )
    
    # load dynamics
    if arglist.dynamics_path != "none":
        state_dict = torch.load(os.path.join(
            arglist.exp_path, "dynamics", arglist.dynamics_path, "model.pt"
        ))
        dynamics_model.load_state_dict(state_dict)
        for n, p in dynamics_model.named_parameters():
            p.requires_grad = arglist.train_dynamics

    # init agent
    if arglist.agent == "vin":
        agent = VINAgent(dynamics_model, arglist.horizon)
        agent.hmm.obs_model.init_batch_norm(obs_mean, obs_var)
        agent.hmm.ctl_model.init_batch_norm(ctl_mean, ctl_var)

    # init trainer
    model = AIRL(
        agent, obs_mean, obs_var, ctl_mean, ctl_var,
        hidden_dim=arglist.hidden_dim, gru_layers=arglist.gru_layers,
        mlp_layers=arglist.mlp_layers, gamma=arglist.gamma,
        buffer_size=arglist.buffer_size, rnn_steps=arglist.rnn_steps, 
        batch_size=arglist.batch_size, d_steps=arglist.d_steps, ac_steps=arglist.ac_steps,
        lr_agent=arglist.lr, lr_d=arglist.lr_d, lr_q=arglist.lr_q, 
        decay=arglist.decay, grad_clip=arglist.grad_clip, polyak=arglist.polyak
    )
    print(f"num parameters: {count_parameters(model)}")
    print(model)

    # load from check point
    if arglist.checkpoint_path != "none":
        cp_path = os.path.join(
            arglist.exp_path, "agents", 
            arglist.agent, arglist.checkpoint_path
        )
        # load state dict
        state_dict = torch.load(os.path.join(cp_path, "model.pt"), map_location=torch.device("cpu"))

        # load history
        df_history_cp = pd.read_csv(os.path.join(cp_path, "history.csv"))

        model.agent.load_state_dict(state_dict, strict=False)
        print(f"loaded checkpoint from {cp_path}")

    # load map
    map_data = MapReader()
    map_data.parse(os.path.join(arglist.data_path, "maps", arglist.scenario + ".osm"))
    
    # init simulator
    env = InteractionSimulator(ego_dataset, map_data)
    observer = Observer(
        map_data, ego_features=ego_features, relative_features=relative_features
    )

    model, df_history = train(
        env, observer, model, rel_dataset, action_set, 
        arglist.num_eps, arglist.epochs, arglist.max_eps_len
    )

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)