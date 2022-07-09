import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# setup imports
from src.simulation.observers import FEATURE_SET, ACTION_SET
from src.data.train_utils import load_data
from src.data.ego_dataset import RelativeDataset, aug_flip_lr, collate_fn

# model imports
from src.distributions.hmm import ContinuousGaussianHMM
from src.agents.vin_agents import VINAgent
from src.algo.irl import BehaviorCloning

# eval imports
from src.evaluation.offline import eval_dynamics_episode, sample_action_components
from src.visualization.utils import set_plotting_style, plot_time_series, plot_scatter

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
    parser.add_argument("--dynamics_model", type=str, choices=["cghmm", "ecghmm"],
        default="cghmm", help="dynamics model type, default=cghmm")
    parser.add_argument("--agent", type=str, choices=["vin"], default="vin", 
        help="agent type, default=vin")
    parser.add_argument("--load_from_agent", type=bool_, default=False,
        help="whether to load dynamics model from agent, default=False")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--max_eps_len", type=int, default=200, 
        help="max episode length, default=200")
    parser.add_argument("--num_eps", type=int, default=5, 
        help="number of episodes to evaluate, default=5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def main(arglist):
    np.random.seed(arglist.seed)
    torch.manual_seed(arglist.seed)
    
    df_track = load_data(arglist.data_path, arglist.scenario, arglist.filename)
    df_track["is_train"] = 1 - df_track["is_train"]
    df_track = df_track.loc[df_track["is_train"] == 1]
    
    # get experiment path
    path = "agents" if arglist.load_from_agent else "dynamics"
    model_name = arglist.agent if arglist.load_from_agent else arglist.dynamics_model
    exp_path = os.path.join(arglist.exp_path, path, model_name, arglist.exp_name)
    print(f"evalusting offline exp: {exp_path}")

    # load config 
    with open(os.path.join(exp_path, "args.json"), "r") as f:
        config = json.load(f)

    """ TODO: add code to adapt input feature set """
    # define feature set
    ego_features = ["d", "ds", "dd", "kappa_r", "psi_error_r", ]
    relative_features = ["s_rel", "d_rel", "ds_rel", "dd_rel", "loom_s"]
    feature_set = ego_features + relative_features
    assert set(ego_features).issubset(set(FEATURE_SET["ego"]))
    assert set(relative_features).issubset(set(FEATURE_SET["relative"]))
    
    # define action set
    if config["action_set"] == "frenet":
        action_set = ["dds", "ddd"]
    else:
        action_set = ["ax_ego", "ay_ego"]
    assert set(action_set).issubset(set(ACTION_SET))

    dataset = RelativeDataset(
        df_track, feature_set, action_set, train_labels_col="is_train",
        max_eps_len=arglist.max_eps_len, augmentation=[aug_flip_lr]
    )
    loader = DataLoader(dataset, len(dataset), shuffle=False, collate_fn=collate_fn)
    obs_dim, ctl_dim = len(feature_set), 2 

    print(f"feature set: {feature_set}")
    print(f"test size: {len(loader.dataset)}")
    
    # load state dict
    state_dict = torch.load(os.path.join(exp_path, "model.pt"), map_location=torch.device("cpu"))

    # init dynamics model 
    dynamics_model_name = model_name if not arglist.load_from_agent else config["dynamics_model"]
    if dynamics_model_name == "cghmm":
        dynamics_model = ContinuousGaussianHMM(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, 
            config["hmm_rank"], config["obs_cov"], config["ctl_cov"]
        )
    
    # init agent
    if arglist.load_from_agent:
        if arglist.agent == "vin":
            agent = VINAgent(dynamics_model, config["horizon"])
    
        # init model
        if config["algo"] == "bc":
            model = BehaviorCloning(agent)

        model.load_state_dict(state_dict)
        dynamics_model = model.agent
    else:
        dynamics_model.load_state_dict(state_dict, strict=False)

    print(dynamics_model)

    # sample eval episodes
    test_eps_id = np.random.choice(np.arange(len(dataset)), arglist.num_eps)
    
    figs_o = [] # dynamics prediction figures
    for i, eps_id in enumerate(test_eps_id):
        obs = dataset[eps_id]["ego"]
        ctl = dataset[eps_id]["act"]
        meta = dataset[eps_id]["meta"]
        title = f"track {meta[0]} eps {meta[1]}"

        o_sample = eval_dynamics_episode(dynamics_model, obs, ctl)
        fig_o, ax = plot_time_series(
            obs, feature_set, x_sample=o_sample, 
            num_cols=5, figsize=(12, 4), title=title
        )
        
        figs_o.append(fig_o)
    
    # plot action components
    u_sample_components = sample_action_components(agent.hmm.ctl_model, num_samples=50)
    fig_cmp, ax = plot_scatter(u_sample_components, action_set[0], action_set[1])

    """ TODO: plot marginal samples """
    
    # save results
    if arglist.save:
        save_path = os.path.join(exp_path, "eval_dynamics")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        for i, f in enumerate(figs_o):
            f.savefig(os.path.join(save_path, f"test_scene_{i}_dynamics.png"), dpi=100)
        
        fig_cmp.savefig(os.path.join(save_path, f"action_components.png"), dpi=100)
        
        print("\nonline evaluation results saved at {}".format(save_path))
    
if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)