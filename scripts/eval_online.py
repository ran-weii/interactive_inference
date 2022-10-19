import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# set up imports
from src.simulation.observers import FEATURE_SET, ACTION_SET
from src.data.train_utils import load_data
from src.data.data_filter import filter_segment_by_length
from src.map_api.lanelet import MapReader
from src.data.ego_dataset import RelativeDataset
from src.simulation.simulator import InteractionSimulator
from src.simulation.sensors import EgoSensor, LeadVehicleSensor, LidarSensor
from src.simulation.observers import Observer, CarfollowObserver
from src.simulation.utils import create_svt_from_df
from src.evaluation.online import eval_episode

# model imports
from src.agents.vin_agent import VINAgent
from src.agents.hyper_vin_agent import HyperVINAgent
from src.agents.rule_based import IDM
from src.agents.mlp_agents import MLPAgent

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
    parser.add_argument("--agent", type=str, choices=["vin", "idm", "mlp", "hvin"], default="vin", 
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
    parser.add_argument("--rollout_idm", type=bool_, default=False)
    parser.add_argument("--playback", type=bool_, default=False)
    parser.add_argument("--test_posterior", type=bool_, default=False, help="whether to test hvin posterior, default=False")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    parser.add_argument("--save_data", type=bool_, default=False)
    parser.add_argument("--save_video", type=bool_, default=False)
    arglist = parser.parse_args()
    return arglist

def compute_latent(data, agent):
    o = data["ego"].unsqueeze(-2)
    u = data["act"].unsqueeze(-2)
    mask = torch.ones(len(o), 1)
    
    with torch.no_grad():
        # z = agent.encode(o, u, mask)
        z_dist = agent.get_posterior_dist(o, u, mask)
        z = z_dist.mean
    return z

def main(arglist):
    np.random.seed(arglist.seed)
    torch.manual_seed(arglist.seed)

    # get experiment path
    exp_path = os.path.join(arglist.exp_path, "agents", arglist.agent, arglist.exp_name)
    print(f"evalusting online exp: {arglist.agent}/{arglist.exp_name}")
    
    df_track = load_data(arglist.data_path, arglist.scenario, arglist.filename, load_raw=True)
    df_track["is_train"] = 1 - df_track["is_train"]

    # filter episode length
    df_track["eps_id"], df_track["eps_len"] = filter_segment_by_length(
        df_track["eps_id"].values, arglist.min_eps_len
    )

    # build svt
    svt = create_svt_from_df(df_track, eps_id_col="eps_id")

    # load map
    map_data = MapReader()
    map_data.parse(os.path.join(arglist.data_path, "maps", arglist.scenario + ".osm"))
    
    # load config 
    with open(os.path.join(exp_path, "args.json"), "r") as f:
        config = json.load(f)
    
    if config["action_set"] == "frenet":
        config["action_set"] = ["dds"]
    if "feature_set" not in list(config.keys()):
        config["feature_set"] = ["ego_ds", "lv_s_rel", "lv_ds_rel", "lv_inv_tau"]
    
    # define feature set
    ego_sensor = EgoSensor(map_data)
    lv_sensor = LeadVehicleSensor(map_data, track_lv=True)
    lidar_sensor = LidarSensor()
    sensors = [ego_sensor, lv_sensor, lidar_sensor]

    if config["action_set"] == ["dds"]:
        observer = CarfollowObserver(map_data, sensors, feature_set=config["feature_set"])
    else:
        observer = Observer(map_data, sensors, feature_set=config["feature_set"], action_set=config["action_set"])
    
    env = InteractionSimulator(map_data, sensors, observer, svt)

    feature_set = env.observer.feature_set
    action_set = env.observer.action_set

    obs_dim = len(feature_set)
    ctl_dim = len(action_set)
    print(f"feature set: {feature_set}")
    print(f"action set: {action_set}")

    # compute ctl limits
    ctl_max = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].max().values).to(torch.float32)
    ctl_min = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].min().values).to(torch.float32)
    ctl_lim = torch.max(torch.abs(ctl_max), torch.abs(ctl_min)) * 1.2

    # init agent
    if arglist.agent == "vin":
        agent = VINAgent(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["hmm_rank"],
            config["horizon"], alpha=config["alpha"], beta=config["beta"], obs_cov=config["obs_cov"], ctl_cov=config["ctl_cov"],            
            use_tanh=config["use_tanh"], ctl_lim=ctl_lim
        )
    elif arglist.agent == "hvin":
        agent = HyperVINAgent(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["hmm_rank"],
            config["horizon"], config["hyper_dim"], config["hidden_dim"], config["num_hidden"],
            config["gru_layers"], config["activation"], alpha=config["alpha"], beta=config["beta"],
            obs_cov=config["obs_cov"], ctl_cov=config["ctl_cov"], rwd=config["rwd"],
            use_tanh=config["use_tanh"], ctl_lim=ctl_lim
        )
    elif arglist.agent == "idm":
        agent = IDM()
    elif arglist.agent == "mlp":
        agent = MLPAgent(
            obs_dim, ctl_dim, config["hidden_dim"], config["num_hidden"],
            activation=config["activation"], use_tanh=config["use_tanh"], ctl_limits=ctl_lim
        )

    # load state dict
    state_dict = torch.load(os.path.join(exp_path, "model.pt"), map_location=torch.device("cpu"))
    state_dict = {k.replace("agent.", ""): v for (k, v) in state_dict.items() if "agent." in k}
    agent.load_state_dict(state_dict, strict=False)
    
    if arglist.rollout_idm:
        env.observer = Observer(map_data, env.sensors, action_set=action_set)
        agent = IDM(env.observer.feature_set)
    
    agent.eval()
    print(agent)
    
    # sample eval episodes
    test_eps_id = np.random.choice(np.arange(env.num_eps), arglist.num_eps)
    print(test_eps_id)
    
    """ debug add rel dataset for hvin to scan """
    dataset = RelativeDataset(df_track, feature_set, action_set, max_eps=1000, seed=arglist.seed)
    
    data = []
    animations = []
    for i, eps_id in enumerate(test_eps_id):
        title = f"eps {eps_id}"
        if arglist.agent == "hvin" and arglist.test_posterior:
            z = compute_latent(dataset[eps_id], agent)
        else:
            z = None
        sim_data, rewards = eval_episode(
            env, agent, eps_id, z=z, sample_method=arglist.sample_method, playback=arglist.playback
        )
        print(f"test eps: {i}, mean reward: {np.mean(rewards)}")
        
        # """ debug plotting """
        # act = np.stack([d["sim_state"]["sim_act"]["act"] for d in sim_data[1:]])
        # true_act = np.stack([d["sim_state"]["sim_act"]["true_act"] for d in sim_data[1:]])
        
        # fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        # ax[0].plot(act[:, 0], label="pred")
        # ax[0].plot(true_act[:, 0], label="true")
        # ax[0].legend()

        # ax[1].plot(act[:, 1], label="pred")
        # ax[1].plot(true_act[:, 1], label="true")
        # ax[1].legend()

        # plt.show()

        data.append(sim_data)
        if arglist.save_video:
            plot_lidar = False if "lidar" not in (",").join(feature_set) else True
            ani = animate(map_data, sim_data, title=title, plot_lidar=plot_lidar)
            animations.append(ani)
    
    # save results
    if arglist.save_data or arglist.save_video:
        post_fix = "_post" if arglist.test_posterior else ""
        eval_path = os.path.join(exp_path, "eval_online")
        save_path = os.path.join(eval_path, f"{arglist.seed}_{arglist.sample_method}{post_fix}")
        if not os.path.exists(eval_path):
            os.mkdir(eval_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        if arglist.save_data:
            filename = "sim_data.p"
            if arglist.rollout_idm:
                filename = "sim_data_idm.p"
            if arglist.playback:
                filename = "sim_data_playback.p"
            with open(os.path.join(save_path, filename), "wb") as f:
                pickle.dump(data, f)
        
        if arglist.save_video:
            for i, ani in enumerate(animations):
                save_animation(ani, os.path.join(save_path, f"test_scene_{i}_ani.mp4"))
                # save_animation(ani, f"/Users/hfml/Documents/temp/test_car_follow_{i}.mp4")
        
        print("\nonline evaluation results saved at {}".format(save_path))

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)