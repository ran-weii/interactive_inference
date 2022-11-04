import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# setup imports
from src.data.train_utils import load_data, update_record_id, count_parameters
from src.data.data_filter import filter_segment_by_length
from src.data.ego_dataset import RelativeDataset, collate_fn

# model imports
from src.agents.rule_based import IDM
from src.agents.nn_agents import MLPAgent, RNNAgent
from src.agents.vin_agent import VINAgent
from src.agents.hyper_vin_agent import HyperVINAgent

# eval imports
from src.evaluation.offline import eval_actions_episode, eval_actions_batch
from src.evaluation.metrics import mean_absolute_error
from src.visualization.utils import set_plotting_style, plot_time_series, plot_scatter

import warnings
warnings.filterwarnings("ignore")

set_plotting_style()

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    str_list_ = lambda x: x.replace(" ", "").split(",")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../interaction-dataset-master"
    )
    parser.add_argument("--exp_path", type=str, default="../exp")
    parser.add_argument("--scenario", type=str, default="DR_CHN_Merging_ZS")
    parser.add_argument("--filenames", type=str_list_, default=["vehicle_tracks_000.csv"])
    parser.add_argument("--test_lanes", type=str_list_, default=["3", "4"])
    # testing args
    parser.add_argument("--agent", type=str, choices=["vin", "hvin", "rnn", "mlp", "idm"], 
        default="vin", help="agent type, default=vin")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--min_eps_len", type=int, default=100, 
        help="min episode length, default=100")
    parser.add_argument("--max_eps_len", type=int, default=200, 
        help="max episode length, default=200")
    parser.add_argument("--batch_size", type=int, default=64, 
        help="evaluation batch size, default=64")
    parser.add_argument("--num_eps", type=int, default=5, 
        help="number of episodes to evaluate, default=5")
    parser.add_argument("--num_samples", type=int, default=30, 
        help="number of sample to draw, default=30")
    parser.add_argument("--sample_method", type=str, choices=["ace", "acm"], default="acm", 
        help="action sampling method, default=acm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def main(arglist):
    np.random.seed(arglist.seed)
    torch.manual_seed(arglist.seed)
    
    # load data files
    df_track = []
    for filename in arglist.filenames:
        df_track.append(load_data(arglist.data_path, arglist.scenario, filename, load_raw=False))
    df_track = pd.concat(df_track, axis=0)
    df_track = update_record_id(df_track)
    
    # reverse train test labels
    df_track["is_train"] = 1 - df_track["is_train"]
    
    # filter invalid lanes
    test_lanes = [int(l) for l in arglist.test_lanes]
    is_train = df_track["is_train"].values
    is_train[df_track["ego_lane_id"].isin(test_lanes) == False] = np.nan
    df_track = df_track.assign(is_train=is_train)

    # filter episode length
    eps_id, df_track["eps_len"] = filter_segment_by_length(
        df_track["eps_id"].values, arglist.min_eps_len
    )
    df_track = df_track.assign(eps_id=eps_id.astype(float))    
    df_track = df_track.loc[df_track["is_train"] == 1]
    
    # get experiment path
    exp_path = os.path.join(arglist.exp_path, "agents", arglist.agent, arglist.exp_name)
    print(f"evalusting offline exp: {arglist.agent}/{arglist.exp_name}")
    print(f"on records: {arglist.filenames}")

    # load config 
    with open(os.path.join(exp_path, "args.json"), "r") as f:
        config = json.load(f)
    
    # init dataset
    feature_set = config["feature_set"]
    action_set = config["action_set"]
    dataset = RelativeDataset(
        df_track, feature_set, action_set, train_labels_col="is_train",
        max_eps_len=arglist.max_eps_len, max_eps=10000, state_action=False, seed=arglist.seed
    )
    test_loader = DataLoader(
        dataset, batch_size=arglist.batch_size, shuffle=False, collate_fn=collate_fn
    )
    obs_dim, ctl_dim = len(feature_set), len(action_set)

    print(f"feature set: {feature_set}")
    print(f"action set: {action_set}")
    print(f"test size: {len(test_loader.dataset)}")

    # init agent
    if config["agent"] == "vin":
        agent = VINAgent(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["hmm_rank"],
            config["horizon"], alpha=config["alpha"], beta=config["beta"], obs_model=config["obs_model"],
            obs_cov=config["obs_cov"], ctl_cov=config["ctl_cov"], rwd=config["rwd"], detach=config["detach"]
        )

    elif config["agent"] == "hvin":
        agent = HyperVINAgent(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["hmm_rank"],
            config["horizon"], config["hyper_dim"], config["hidden_dim"], config["num_hidden"], 
            config["gru_layers"], config["activation"], alpha=config["alpha"], beta=config["beta"], 
            obs_model=config["obs_model"], obs_cov=config["obs_cov"], ctl_cov=config["ctl_cov"], rwd=config["rwd"],
            hyper_cov=config["hyper_cov"], train_prior=config["train_prior"]
        )
    
    elif config["agent"] == "rnn":
        agent = RNNAgent(
            obs_dim, ctl_dim, config["act_dim"], config["hidden_dim"], 
            config["num_hidden"], config["gru_layers"], config["activation"]
        )

    elif config["agent"] == "mlp":
        agent = MLPAgent(
            obs_dim, ctl_dim, config["act_dim"], config["hidden_dim"],
            config["num_hidden"], config["activation"]
        )

    elif config["agent"] == "idm":
        agent = IDM(feature_set)

    # load state dict
    state_dict = torch.load(os.path.join(exp_path, "model.pt"), map_location=torch.device("cpu"))
    state_dict = {k.replace("agent.", ""): v for (k, v) in state_dict.items() if "agent." in k}
    agent.load_state_dict(state_dict, strict=False)
    agent.eval()
    print(agent)
    print(f"num parameters: {count_parameters(agent)}")
    
    u_true, u_sample = eval_actions_batch(
        agent, test_loader, sample_method=arglist.sample_method
    )
    mae = mean_absolute_error(u_true, u_sample)
    
    print(f"mae: {mae}")
    
    # save results
    if arglist.save:
        save_path = os.path.join(exp_path, "eval_offline")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        with open(os.path.join(save_path, "lanes_{}.json".format(",".join(arglist.test_lanes))), "w") as f:
            json.dump({"mae": float(mae[0]), "test_lanes": test_lanes}, f)

        print("\nonline evaluation results saved at {}".format(save_path))

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)