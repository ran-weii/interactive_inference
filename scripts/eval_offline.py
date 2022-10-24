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
from src.simulation.observers import FEATURE_SET, ACTION_SET
from src.simulation.observers import Observer, CarfollowObserver
from src.data.train_utils import load_data
from src.data.ego_dataset import RelativeDataset, aug_flip_lr, collate_fn
from src.simulation.sensors import EgoSensor, LeadVehicleSensor, LidarSensor

# model imports
from src.agents.vin_agent import VINAgent, VINAgent2
from src.agents.hyper_vin_agent import HyperVINAgent
from src.agents.mlp_agents import MLPAgent

# eval imports
from src.evaluation.offline import eval_actions_episode, sample_action_components, transform_action
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
    parser.add_argument("--agent", type=str, choices=["vin", "hvin", "mlp", "tvin"], default="vin", 
        help="agent type, default=vin")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--max_eps_len", type=int, default=200, 
        help="max episode length, default=200")
    parser.add_argument("--num_eps", type=int, default=5, 
        help="number of episodes to evaluate, default=5")
    parser.add_argument("--num_sample", type=int, default=30, 
        help="number of sample to draw, default=30")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

# def transform_action(u, discretizers):
#     u_split = torch.chunk(u, 2, dim=-1)
#     u_shape = list(u_split[0].shape)
#     ux = discretizers[0].inverse_transform(u_split[0].view(-1, 1)).reshape(u_shape)
#     uy = discretizers[1].inverse_transform(u_split[1].view(-1, 1)).reshape(u_shape)
#     ux = torch.from_numpy(ux).to(torch.float32)
#     uy = torch.from_numpy(uy).to(torch.float32)
#     u = torch.cat([ux, uy], dim=-1)
#     return u

def main(arglist):
    np.random.seed(arglist.seed)
    torch.manual_seed(arglist.seed)

    # get experiment path
    exp_path = os.path.join(arglist.exp_path, "agents", arglist.agent, arglist.exp_name)
    print(f"evalusting offline exp: {arglist.agent}/{arglist.exp_name}")
    
    df_track = load_data(arglist.data_path, arglist.scenario, arglist.filename)
    df_track["is_train"] = 1 - df_track["is_train"]
    df_track = df_track.loc[df_track["is_train"] == 1]
    
    # load config 
    with open(os.path.join(exp_path, "args.json"), "r") as f:
        config = json.load(f)

    # define action set
    if config["action_set"] == "frenet":
        action_set = ["dds", "ddd"]
    elif config["action_set"] == "ego":
        action_set = ["ax_ego", "ay_ego"]
    
    # define feature set
    ego_sensor = EgoSensor(None)
    lv_sensor = LeadVehicleSensor(None)
    lidar_sensor = LidarSensor()
    sensors = [ego_sensor, lv_sensor]
    if config["use_lidar"]:
        sensors.append(lidar_sensor)
    
    if config["observer"] == "full":
        observer = Observer(None, sensors, action_set)
    elif config["observer"] == "car_follow":
        observer = CarfollowObserver(None, sensors)
    feature_set = observer.feature_names
    action_set = observer.action_set

    # compute empirical control limits
    ctl_max = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].max().values).to(torch.float32)
    ctl_min = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].min().values).to(torch.float32)
    ctl_lim = torch.max(torch.abs(ctl_max), torch.abs(ctl_min)) * 1.5

    dataset = RelativeDataset(
        df_track, feature_set, action_set, train_labels_col="is_train",
        max_eps_len=arglist.max_eps_len, augmentation=[]
    )
    loader = DataLoader(dataset, len(dataset), shuffle=False, collate_fn=collate_fn)
    obs_dim, ctl_dim = len(feature_set), len(action_set)
    
    print(f"feature set: {feature_set}")
    print(f"test size: {len(loader.dataset)}")

    # init agent
    if arglist.agent == "vin":
        alpha = config["alpha"] if "alpha" in config else 0.
        beta = config["beta"] if "beta" in config else 1.
        agent = VINAgent(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["hmm_rank"],
            config["horizon"], alpha=alpha, beta=beta, obs_cov=config["obs_cov"], ctl_cov=config["ctl_cov"], 
            use_tanh=config["use_tanh"], ctl_lim=ctl_lim
        )
        if "causal" in config.keys():
            if config["causal"] == False:
                agent = VINAgent2(
                    config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["hmm_rank"],
                    config["horizon"], alpha=alpha, obs_cov=config["obs_cov"], ctl_cov=config["ctl_cov"],            
                    use_tanh=config["use_tanh"], ctl_lim=ctl_lim
                )
    elif arglist.agent == "hvin":
        agent = HyperVINAgent(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["hmm_rank"],
            config["horizon"], config["hyper_dim"], config["hidden_dim"], config["num_hidden"], 
            config["gru_layers"], config["activation"],
            obs_cov=config["obs_cov"], ctl_cov=config["ctl_cov"], 
            use_tanh=config["use_tanh"], ctl_lim=ctl_lim
        )
    elif arglist.agent == "mlp":
        agent = MLPAgent(
            obs_dim, ctl_dim, config["hidden_dim"], config["num_hidden"],
            use_tanh=True, ctl_limits=torch.tensor([5.5124, 0.0833])
        )

    # load state dict
    state_dict = torch.load(os.path.join(exp_path, "model.pt"), map_location=torch.device("cpu"))
    state_dict = {k.replace("agent.", ""): v for (k, v) in state_dict.items() if "agent." in k}
    agent.load_state_dict(state_dict, strict=True)
    print(agent)

    # sample eval episodes
    test_eps_id = np.random.choice(np.arange(len(dataset)), arglist.num_eps)
    
    figs_u = [] # action prediction figures
    for i, eps_id in enumerate(test_eps_id):
        obs = dataset[eps_id]["ego"]
        ctl = dataset[eps_id]["act"]
        meta = dataset[eps_id]["meta"]
        title = f"track {meta[0]} eps {meta[1]}"

        u_sample, loss = eval_actions_episode(agent, obs, ctl, num_samples=arglist.num_sample)
        
        print("loss", loss)

        # if config["discrete_action"]:
        #     u_sample = transform_action(u_sample, discretizers)
        #     ctl = transform_action(ctl, discretizers)
        
        fig_u, ax = plot_time_series(
            ctl, action_set, x_sample=u_sample, 
            num_cols=2, figsize=(4, 2.5), title=None
            # num_cols=2, figsize=(6, 4), title=title
        )
        plt.show()
        # exit()
        figs_u.append(fig_u)
    
    # save results
    if arglist.save:
        save_path = os.path.join(exp_path, "eval_offline")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        for i, f in enumerate(figs_u):
            f.savefig(os.path.join(save_path, f"test_scene_{i}_action.png"), dpi=100)
        
        # plot action components
        if arglist.agent == "vin":
            u_sample_components = sample_action_components(agent.ctl_model, num_samples=500)
            fig_cmp, ax = plot_scatter(u_sample_components, action_set[0], action_set[1])
            fig_cmp.savefig(os.path.join(save_path, f"action_components.png"), dpi=100)

        print("\nonline evaluation results saved at {}".format(save_path))

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)