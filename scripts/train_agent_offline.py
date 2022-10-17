import argparse
import os
import json
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 

# setup imports
from src.data.train_utils import load_data, train_test_split, count_parameters
from src.data.data_filter import filter_segment_by_length
from src.data.ego_dataset import RelativeDataset, aug_flip_lr, collate_fn

# model imports
from src.agents.vin_agent import VINAgent
from src.agents.hyper_vin_agent import HyperVINAgent
from src.agents.rule_based import IDM
from src.agents.mlp_agents import MLPAgent
from src.algo.irl import BehaviorCloning
from src.algo.hyper_bc import HyperBehaviorCloning

# training imports
from src.algo.utils import train
from src.visualization.utils import plot_history

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    str_list_ = lambda x: x.split(",")
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../interaction-dataset-master"
    )
    parser.add_argument("--exp_path", type=str, default="../exp")
    parser.add_argument("--scenario", type=str, default="DR_CHN_Merging_ZS")
    parser.add_argument("--filenames", type=str_list_, default=["vehicle_tracks_007.csv"])
    parser.add_argument("--checkpoint_path", type=str, default="none", 
        help="if entered train agent from check point")
    # agent args
    parser.add_argument("--agent", type=str, choices=["vin", "hvin", "vin", "idm", "mlp"], default="vin", help="agent type, default=vin")
    parser.add_argument("--state_dim", type=int, default=30, help="agent state dimension, default=30")
    parser.add_argument("--act_dim", type=int, default=45, help="agent action dimension, default=45")
    parser.add_argument("--horizon", type=int, default=30, help="agent planning horizon, default=30")
    parser.add_argument("--obs_cov", type=str, default="full", help="agent observation covariance, default=full")
    parser.add_argument("--ctl_cov", type=str, default="full", help="agent control covariance, default=full")
    parser.add_argument("--hmm_rank", type=int, default=32, help="agent hmm rank, default=32")
    parser.add_argument("--alpha", type=float, default=1., help="agent entropy reward coefficient, default=1.")
    parser.add_argument("--beta", type=float, default=1., help="agent policy prior coefficient, default=1.")
    parser.add_argument("--rwd", type=str, choices=["efe", "ece"], default="efe", help="agent reward function. default=efe")
    parser.add_argument("--feature_set", type=str_list_, default=["ego_ds", "lv_s_rel", "lv_ds_rel"], help="agent feature set")
    parser.add_argument("--action_set", type=str_list_, default="dds", help="agent action set, default=dds")
    parser.add_argument("--use_tanh", type=bool_, default=True, help="whether to use tanh transform, default=True")
    parser.add_argument("--detach", type=bool_, default=True, help="whether to detach dynamics model, default=True")
    parser.add_argument("--discretize_ctl", type=bool_, default=True, help="whether to discretize ctl using gmm, default=True")
    parser.add_argument("--hyper_dim", type=int, default=4, help="number of latent factor, default=4")
    parser.add_argument("--train_prior", type=bool_, default=False, help="whether to train hvin prior, default=False")
    parser.add_argument("--sample_z", type=bool_, default=False, help="whether to compute obs likelihood with sampled z, default=False")
    # nn args
    parser.add_argument("--hidden_dim", type=int, default=64, help="nn hidden dimension, default=64")
    parser.add_argument("--num_hidden", type=int, default=2, help="number of hidden layers, default=2")
    parser.add_argument("--gru_layers", type=int, default=1, help="number of gru layers, default=1")
    parser.add_argument("--activation", type=str, default="relu", help="nn activation, default=relu")
    parser.add_argument("--norm_obs", type=bool_, default=False, help="whether to normalize observation for discriminator, default=False")
    # algo args
    parser.add_argument("--algo", type=str, choices=["bc", "hbc", "dag"], default="bc", help="training algorithm, default=bc")
    parser.add_argument("--train_mode", type=str, choices=["prior", "post", "marginal"], help="training mode for hbc")
    parser.add_argument("--bptt_steps", type=int, default=30, help="bptt truncation steps, default=30")
    parser.add_argument("--bc_penalty", type=float, default=1., help="behavior cloning penalty, default=1.")
    parser.add_argument("--obs_penalty", type=float, default=0., help="observation penalty, default=0.")
    parser.add_argument("--reg_penalty", type=float, default=0., help="regularization penalty, default=0.")
    parser.add_argument("--post_obs_penalty", type=float, default=0., help="posterior observation penalty, default=0.")
    parser.add_argument("--kl_penalty", type=float, default=1., help="kl penalty, default=1.")
    # training args
    parser.add_argument("--min_eps_len", type=int, default=50, help="min track length, default=50")
    parser.add_argument("--max_eps_len", type=int, default=200, help="max track length, default=200")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="ratio of training dataset, default=0.7")
    parser.add_argument("--batch_size", type=int, default=64, help="training batch size, default=64")
    parser.add_argument("--epochs", type=int, default=3, help="number of training epochs, default=10")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate, default=0.01")
    parser.add_argument("--lr_post", type=float, default=0.005, help="hvin posterior learning rate, default=0.01")
    parser.add_argument("--decay", type=float, default=1e-5, help="weight decay, default=0")
    parser.add_argument("--grad_clip", type=float, default=None, help="gradient clipping, default=None")
    parser.add_argument("--cp_every", type=int, default=1000, help="checkpoint interval, default=1000")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

class SaveCallback:
    def __init__(self, arglist, cp_history=None):
        date_time = datetime.datetime.now().strftime("%m-%d-%Y %H-%M-%S")
        exp_path = os.path.join(arglist.exp_path, "agents")
        agent_path = os.path.join(exp_path, arglist.agent)
        save_path = os.path.join(agent_path, date_time)
        model_path = os.path.join(save_path, "model")
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        if not os.path.exists(agent_path):
            os.mkdir(agent_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        
        # save args
        with open(os.path.join(save_path, "args.json"), "w") as f:
            json.dump(vars(arglist), f)

        self.save_path = save_path
        self.model_path = model_path
        self.cp_history = cp_history
        self.cp_every = arglist.cp_every

        self.num_test_eps = 0
        self.iter = 0

    def __call__(self, model, history):
        self.iter += 1
        if self.iter % self.cp_every != 0:
            return
        
        # save history
        df_history = pd.DataFrame(history)
        self.save_history(model, df_history)
        
        # save model
        cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state_dict, os.path.join(self.model_path, f"model_{self.iter}.pt"))
        print(f"\ncheckpoint saved at: {self.save_path}\n")
    
    def save_history(self, model, df_history):
        if self.cp_history is not None:
            df_history["epoch"] += self.cp_history["epoch"].values[-1] + 1
            df_history["time"] += self.cp_history["time"].values[-1]
            df_history = pd.concat([self.cp_history, df_history], axis=0)
        df_history.to_csv(os.path.join(self.save_path, "history.csv"), index=False)
        
        # save history plot
        fig_history, _ = plot_history(df_history, model.loss_keys)
        fig_history.savefig(os.path.join(self.save_path, "history.png"), dpi=100)

        plt.clf()
        plt.close()

    def save_model(self, model):
        cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state_dict, os.path.join(self.save_path, "model.pt"))

def get_new_eps_id(df_track):
    """ Combine record_id and eps_id """
    df_track["record_id"] = df_track["record_id"].astype(int)
    num_digits = len(str(np.nanmax(df_track["eps_id"]).astype(int)))
    new_eps_id = df_track["record_id"] * 10**num_digits + df_track["eps_id"]
    return new_eps_id

def main(arglist):
    torch.manual_seed(arglist.seed)
    
    # load data files
    df_track = []
    for filename in arglist.filenames:
        df_track.append(load_data(arglist.data_path, arglist.scenario, filename, load_raw=False))
    df_track = pd.concat(df_track, axis=0)
    df_track = df_track.assign(is_supervised=1)
    
    # load unlabeled random dataset
    df_track_random = load_data(arglist.data_path, arglist.scenario, "vehicle_tracks_015.csv", load_raw=False)
    df_track_random = df_track_random.assign(is_supervised=0)
    df_track = df_track.merge(df_track_random, how="outer")

    df_track = df_track.assign(eps_id=get_new_eps_id(df_track))

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
    
    ctl_max = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].max().values).to(torch.float32)
    ctl_min = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].min().values).to(torch.float32)
    ctl_lim = torch.max(torch.abs(ctl_max), torch.abs(ctl_min)) * 1.2
    
    dataset = RelativeDataset(
        df_track, feature_set, action_set, train_labels_col="is_train",
        max_eps_len=arglist.max_eps_len, augmentation=[], max_eps=10000, seed=arglist.seed
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
            arglist.horizon, alpha=arglist.alpha, beta=arglist.beta, 
            obs_cov=arglist.obs_cov, ctl_cov=arglist.ctl_cov, rwd=arglist.rwd, 
            use_tanh=arglist.use_tanh, ctl_lim=ctl_lim, detach=arglist.detach
        )
        
        agent.obs_model.init_batch_norm(obs_mean, obs_var)
        if not arglist.use_tanh:
            agent.ctl_model.init_batch_norm(ctl_mean, ctl_var)
        
        if arglist.action_set == ["dds"] and arglist.discretize_ctl:
            # load ctl gmm parameters
            with open(os.path.join(arglist.exp_path, "agents", "ctl_model", "model.p"), "rb") as f:
                [ctl_means, ctl_covs, weights] = pickle.load(f)

            agent.ctl_model.init_params(ctl_means, ctl_covs)
            print("action model loaded")

    elif arglist.agent == "hvin":
        agent = HyperVINAgent(
            arglist.state_dim, arglist.act_dim, obs_dim, ctl_dim, arglist.hmm_rank,
            arglist.horizon, arglist.hyper_dim, arglist.hidden_dim, arglist.num_hidden, 
            arglist.gru_layers, arglist.activation, alpha=arglist.alpha, beta=arglist.beta, 
            obs_cov=arglist.obs_cov, ctl_cov=arglist.ctl_cov, rwd=arglist.rwd,
            use_tanh=arglist.use_tanh, ctl_lim=ctl_lim, train_prior=arglist.train_prior
        )
        agent.obs_mean.data = obs_mean
        agent.obs_variance.data = obs_var
        agent.obs_model.init_batch_norm(obs_mean, obs_var)
        if not arglist.use_tanh:
            agent.ctl_model.init_batch_norm(ctl_mean, ctl_var)
        
        if arglist.action_set == ["dds"]:
            # load ctl gmm parameters
            with open(os.path.join(arglist.exp_path, "agents", "ctl_model", "model.p"), "rb") as f:
                [ctl_means, ctl_covs, weights] = pickle.load(f)

            agent.ctl_model.init_params(ctl_means, ctl_covs)
            print("action model loaded")

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
            agent, arglist.bptt_steps, arglist.bc_penalty, arglist.obs_penalty, arglist.reg_penalty, 
            lr=arglist.lr, decay=arglist.decay, grad_clip=arglist.grad_clip
        )
    elif arglist.algo == "hbc":
        model = HyperBehaviorCloning(
            agent, arglist.train_mode, arglist.detach, arglist.bptt_steps, 
            arglist.bc_penalty, arglist.obs_penalty, arglist.reg_penalty, 
            arglist.post_obs_penalty, arglist.kl_penalty,
            lr=arglist.lr, lr_post=arglist.lr_post, decay=arglist.decay, grad_clip=arglist.grad_clip
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