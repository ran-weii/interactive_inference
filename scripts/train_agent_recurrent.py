import argparse
import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 

# setup imports
from src.data.train_utils import load_data, update_record_id
from src.data.train_utils import train_test_split, count_parameters
from src.data.data_filter import filter_segment_by_length
from src.data.ego_dataset import RelativeDataset, collate_fn

# model imports
from src.agents.rule_based import IDM
from src.agents.nn_agents import MLPAgent, RNNAgent
from src.agents.vin_agent import VINAgent
from src.agents.hyper_vin_agent import HyperVINAgent
from src.algo.bc import RecurrentBehaviorCloning
from src.algo.hyper_bc import HyperBehaviorCloning

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
    parser.add_argument("--filenames", type=str_list_, default=["vehicle_tracks_000.csv"])
    parser.add_argument("--checkpoint_path", type=str, default="none", 
        help="if entered train agent from check point")
    parser.add_argument("--valid_lanes", type=str_list_, default=["3", "4"])
    parser.add_argument("--feature_set", type=str_list_, default=["ego_ds", "lv_s_rel", "lv_ds_rel"], help="agent feature set")
    parser.add_argument("--action_set", type=str_list_, default="dds_smooth", help="agent action set, default=dds_smooth")
    # agent args
    parser.add_argument("--agent", type=str, choices=["vin", "hvin", "rnn", "mlp", "idm"], default="vin", help="agent type, default=vin")
    parser.add_argument("--state_dim", type=int, default=10, help="agent state dimension, default=10")
    parser.add_argument("--act_dim", type=int, default=15, help="agent action dimension, default=15")
    parser.add_argument("--horizon", type=int, default=30, help="agent planning horizon, default=30")
    parser.add_argument("--obs_model", type=str, choices=["flow", "gmm"], default="flow", help="agent observation model, default=flow")
    parser.add_argument("--obs_cov", type=str, default="full", help="agent observation covariance, default=full")
    parser.add_argument("--ctl_cov", type=str, default="full", help="agent control covariance, default=full")
    parser.add_argument("--hmm_rank", type=int, default=0, help="agent hmm rank, 0 is full rank, default=0")
    parser.add_argument("--alpha", type=float, default=1., help="agent entropy reward coefficient, default=1.")
    parser.add_argument("--beta", type=float, default=0., help="agent policy prior coefficient, default=0.")
    parser.add_argument("--rwd", type=str, choices=["efe", "ece"], default="efe", help="agent reward function. default=efe")
    parser.add_argument("--detach", type=bool_, default=False, help="whether to detach dynamics model, default=False")
    parser.add_argument("--pred_steps", type=int, default=5, help="number of forward prediction steps, default=5")
    parser.add_argument("--hyper_dim", type=int, default=4, help="number of latent factor, default=4")
    parser.add_argument("--hyper_cov", type=bool_, default=False, help="whether to use hyper variable for covariance, default=False")
    parser.add_argument("--train_prior", type=bool_, default=False, help="whether to train hvin prior, default=False")
    # nn args
    parser.add_argument("--hidden_dim", type=int, default=64, help="nn hidden dimension, default=64")
    parser.add_argument("--num_hidden", type=int, default=2, help="number of hidden layers, default=2")
    parser.add_argument("--gru_layers", type=int, default=1, help="number of gru layers, default=1")
    parser.add_argument("--activation", type=str, default="relu", help="nn activation, default=relu")
    parser.add_argument("--norm_obs", type=bool_, default=False, help="whether to normalize observation for discriminator, default=False")
    # algo args
    parser.add_argument("--train_mode", type=str, choices=["prior", "post", "marginal"], help="training mode for hbc")
    parser.add_argument("--bptt_steps", type=int, default=500, help="bptt truncation steps, default=500")
    parser.add_argument("--bc_penalty", type=float, default=1., help="behavior cloning penalty, default=1.")
    parser.add_argument("--obs_penalty", type=float, default=0., help="observation penalty, default=0.")
    parser.add_argument("--pred_penalty", type=float, default=0., help="multi step prediction penalty, default=0.")
    parser.add_argument("--reg_penalty", type=float, default=0., help="regularization penalty, default=0.")
    parser.add_argument("--post_obs_penalty", type=float, default=0., help="posterior observation penalty, default=0.")
    parser.add_argument("--kl_penalty", type=float, default=1., help="kl penalty, default=1.")
    # training args
    parser.add_argument("--min_eps_len", type=int, default=50, help="min track length, default=50")
    parser.add_argument("--max_eps_len", type=int, default=500, help="max track length, default=500")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="ratio of training dataset, default=0.7")
    parser.add_argument("--batch_size", type=int, default=64, help="training batch size, default=64")
    parser.add_argument("--epochs", type=int, default=3, help="number of training epochs, default=10")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate, default=0.01")
    parser.add_argument("--lr_flow", type=float, default=0.001, help="normalizing flow learning rate, default=0.001")
    parser.add_argument("--lr_post", type=float, default=0.005, help="hvin posterior learning rate, default=0.005")
    parser.add_argument("--decay", type=float, default=0., help="weight decay, default=0")
    parser.add_argument("--grad_clip", type=float, default=None, help="gradient clipping, default=None")
    parser.add_argument("--decay_steps", type=int, default=200, help="learning rate decay steps, default=100")
    parser.add_argument("--decay_rate", type=float, default=0.8, help="learning rate decay rate, default=0.8")
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
    ctl_mean = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].mean().values).to(torch.float32)
    ctl_var = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].var().values).to(torch.float32)
    
    # init dataset
    dataset = RelativeDataset(
        df_track, feature_set, action_set, train_labels_col="is_train",
        max_eps_len=arglist.max_eps_len, max_eps=10000, state_action=False, seed=arglist.seed
    )
    train_loader, test_loader = train_test_split(
        dataset, arglist.train_ratio, arglist.batch_size, 
        collate_fn=collate_fn, seed=arglist.seed
    )
    obs_dim, ctl_dim = len(feature_set), len(action_set)

    print(f"feature set: {feature_set}")
    print(f"action set: {action_set}")
    print(f"train size: {len(train_loader.dataset)}, test size: {len(test_loader.dataset)}")
    
    # init agent
    if arglist.agent == "vin":
        agent = VINAgent(
            arglist.state_dim, arglist.act_dim, obs_dim, ctl_dim, arglist.hmm_rank,
            arglist.horizon, alpha=arglist.alpha, beta=arglist.beta, obs_model=arglist.obs_model,
            obs_cov=arglist.obs_cov, ctl_cov=arglist.ctl_cov, rwd=arglist.rwd, detach=arglist.detach
        )
        agent.obs_model.init_batch_norm(obs_mean, obs_var)
        agent.ctl_model.init_batch_norm(ctl_mean, ctl_var)

    elif arglist.agent == "hvin":
        agent = HyperVINAgent(
            arglist.state_dim, arglist.act_dim, obs_dim, ctl_dim, arglist.hmm_rank,
            arglist.horizon, arglist.hyper_dim, arglist.hidden_dim, arglist.num_hidden, 
            arglist.gru_layers, arglist.activation, alpha=arglist.alpha, beta=arglist.beta, 
            obs_model=arglist.obs_model, obs_cov=arglist.obs_cov, ctl_cov=arglist.ctl_cov, rwd=arglist.rwd,
            hyper_cov=arglist.hyper_cov, train_prior=arglist.train_prior
        )
        agent.obs_model.init_batch_norm(obs_mean, obs_var)
        agent.ctl_model.init_batch_norm(ctl_mean, ctl_var)
    
    elif arglist.agent == "rnn":
        agent = RNNAgent(
            obs_dim, ctl_dim, arglist.act_dim, arglist.hidden_dim, 
            arglist.num_hidden, arglist.gru_layers, arglist.activation
        )

    elif arglist.agent == "mlp":
        agent = MLPAgent(
            obs_dim, ctl_dim, arglist.act_dim, arglist.hidden_dim,
            arglist.num_hidden, arglist.activation
        )

    elif arglist.agent == "idm":
        agent = IDM(feature_set)
    
    # preload stats
    if hasattr(agent, "obs_mean") and arglist.norm_obs:
        agent.obs_mean.data = obs_mean
        agent.obs_variance.data = obs_var

    if hasattr(agent, "ctl_model") and (arglist.action_set == ["dds"] or arglist.action_set == ["dds_smooth"]):
        # load ctl gmm parameters
        with open(os.path.join(arglist.exp_path, "agents", "ctl_model", "model.p"), "rb") as f:
            [ctl_means, ctl_covs, weights] = pickle.load(f)

        agent.ctl_model.init_params(ctl_means, ctl_covs)
        print("action model loaded")

    # init trainer
    if arglist.agent == "hvin":
        model = HyperBehaviorCloning(
            agent, arglist.train_mode, arglist.detach, arglist.bptt_steps, arglist.pred_steps,
            arglist.bc_penalty, arglist.obs_penalty, arglist.pred_penalty, arglist.reg_penalty, 
            arglist.post_obs_penalty, arglist.kl_penalty,
            lr=arglist.lr, lr_flow=arglist.lr_flow, lr_post=arglist.lr_post, 
            decay=arglist.decay, grad_clip=arglist.grad_clip, decay_steps=arglist.decay_steps, 
            decay_rate=arglist.decay_rate
        )
    else:
        model = RecurrentBehaviorCloning(
            agent, arglist.bptt_steps, arglist.pred_steps, arglist.bc_penalty, 
            arglist.obs_penalty, arglist.pred_penalty, arglist.reg_penalty, 
            lr=arglist.lr, lr_flow=arglist.lr_flow, decay=arglist.decay, 
            grad_clip=arglist.grad_clip, decay_steps=arglist.decay_steps, 
            decay_rate=arglist.decay_rate
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
        cp_model_path = glob.glob(os.path.join(cp_path, "model/*.pt"))
        cp_model_path.sort(key=lambda x: int(os.path.basename(x).replace(".pt", "").split("_")[-1]))
        
        state_dict = torch.load(cp_model_path[-1])
        model.load_state_dict(state_dict["model_state_dict"], strict=False)
        model.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        model.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        
        # load history
        cp_history = pd.read_csv(os.path.join(cp_path, "history.csv"))
        print(f"loaded checkpoint from {cp_path}")
    
    callback = None
    if arglist.save:
        callback = SaveCallback(arglist, model, cp_history)
    
    model, df_history = train(
        model, train_loader, test_loader, arglist.epochs, callback=callback, verbose=1
    )
    if arglist.save:
        callback.save_checkpoint(model)
        callback.save_history(df_history)
    
if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)