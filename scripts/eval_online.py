import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# set up imports
from src.simulation.observers import FEATURE_SET, ACTION_SET
from src.data.train_utils import load_data
from src.data.data_filter import filter_segment_by_length
from src.map_api.lanelet import MapReader
from src.data.ego_dataset import EgoDataset, aug_flip_lr, collate_fn
from src.simulation.simulator import InteractionSimulator
from src.simulation.observers import Observer
from src.simulation.controllers import AgentWrapper
from src.evaluation.online import eval_episode

# model imports
from src.distributions.hmm import ContinuousGaussianHMM
from src.agents.vin_agents import VINAgent
from src.algo.irl import BehaviorCloning

# plotting imports
from src.visualization.utils import set_plotting_style
from src.visualization.animation import animate, save_animation

import warnings
warnings.filterwarnings("ignore")

# set pandas display option
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

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
    parser.add_argument("--min_eps_len", type=int, default=100,
        help="min episode length, default=100")
    parser.add_argument("--max_eps_len", type=int, default=200, 
        help="max episode length, default=200")
    parser.add_argument("--num_eps", type=int, default=5, 
        help="number of episodes to evaluate, default=5")
    parser.add_argument("--sample_method", type=str, 
        choices=["bma", "ace", "acm"], default="ace", 
        help="action sampling method, default=ace")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def main(arglist):
    np.random.seed(arglist.seed)
    torch.manual_seed(arglist.seed)

    # get experiment path
    exp_path = os.path.join(arglist.exp_path, "agents", arglist.agent, arglist.exp_name)
    print(f"evalusting offline exp: {arglist.agent}/{arglist.exp_name}")
    
    df_track = load_data(arglist.data_path, arglist.scenario, arglist.filename)
    df_track["is_train"] = 1 - df_track["is_train"]

    # filter episode length
    df_track["eps_id"], df_track["eps_len"] = filter_segment_by_length(
        df_track["eps_id"].values, arglist.min_eps_len
    )
    
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
    
    dataset = EgoDataset(df_track, train_labels_col="is_train")
    obs_dim, ctl_dim = len(feature_set), 2 

    print(f"feature set: {feature_set}")
    print(f"action set: {action_set}")
    print(f"test size: {len(dataset)}")
    
    # init dynamics model
    if config["dynamics_model"] == "cghmm":
        dynamics_model = ContinuousGaussianHMM(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, 
            config["hmm_rank"], config["obs_cov"], config["ctl_cov"]
        )

    # init agent
    if arglist.agent == "vin":
        agent = VINAgent(dynamics_model, config["horizon"])

    # init model
    if config["algo"] == "bc":
        model = BehaviorCloning(agent)
    
    # load state dict
    state_dict = torch.load(os.path.join(exp_path, "model.pt"), map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=False)
    agent = model.agent
    print(agent)
    
    # load map
    map_data = MapReader()
    map_data.parse(os.path.join(arglist.data_path, "maps", arglist.scenario + ".osm"))

    # init simulation
    env = InteractionSimulator(dataset, map_data)
    observer = Observer(
        map_data, ego_features=ego_features, relative_features=relative_features
    )
    controller = AgentWrapper(observer, agent, action_set, sample_method="ace")
    
    # sample eval episodes
    test_eps_id = np.random.choice(np.arange(len(dataset)), arglist.num_eps)
    
    animations = []
    for i, eps_id in enumerate(test_eps_id):
        title = f"eps {eps_id}"

        sim_states, sim_acts, track_data = eval_episode(env, controller, eps_id)
        
        ani = animate(map_data, sim_states, track_data, title=title)

        animations.append(ani)
    
    # save results
    if arglist.save:
        eval_path = os.path.join(exp_path, "eval_online")
        save_path = os.path.join(eval_path, f"{arglist.seed}_{arglist.sample_method}")
        if not os.path.exists(eval_path):
            os.mkdir(eval_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        for i, ani in enumerate(animations):
            save_animation(ani, os.path.join(save_path, f"test_scene_{i}_ani.mp4"))
        
        print("\nonline evaluation results saved at {}".format(save_path))

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)