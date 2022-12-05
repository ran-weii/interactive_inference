import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# set up imports
from src.data.train_utils import load_data, update_record_id, count_parameters
from src.data.data_filter import filter_segment_by_length
from src.map_api.lanelet import MapReader
from src.data.ego_dataset import RelativeDataset
from src.simulation.simulator import InteractionSimulator
from src.simulation.sensors import EgoSensor, LeadVehicleSensor, LidarSensor
from src.simulation.observers import Observer, CarfollowObserver
from src.simulation.utils import create_svt_from_df
from src.evaluation.online import eval_episode
from src.evaluation.metrics import compute_interquartile_mean

# agent imports
from src.agents.rule_based import IDM
from src.agents.nn_agents import MLPAgent, RNNAgent
from src.agents.vin_agent import VINAgent
from src.agents.hyper_vin_agent import HyperVINAgent

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
    parser.add_argument("--agent", type=str, choices=["vin", "hvin", "rnn", "mlp", "idm"], default="vin", 
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
    parser.add_argument("--playback", type=bool_, default=False)
    parser.add_argument("--test_on_train", type=bool_, default=False, 
        help="whether to test on train episode, default=False")
    parser.add_argument("--test_posterior", type=bool_, default=False, 
        help="whether to test hvin posterior, default=False")
    parser.add_argument("--eps_ids", type=str_list_, default=[], 
        help="test episode ids supplied, default=[]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_summary", type=bool_, default=True, help="whether to save test summary")
    parser.add_argument("--save_data", type=bool_, default=False)
    parser.add_argument("--save_video", type=bool_, default=False)
    arglist = parser.parse_args()
    return arglist

def compute_latent(data, agent):
    o = data["obs"].unsqueeze(-2)
    u = data["act"].unsqueeze(-2)
    mask = torch.ones(len(o), 1)
    
    with torch.no_grad():
        z_dist = agent.get_posterior_dist(o, u, mask)
        z = z_dist.mean
    return z

def main(arglist):
    np.random.seed(arglist.seed)
    torch.manual_seed(arglist.seed)

    # get experiment path
    exp_path = os.path.join(arglist.exp_path, "agents", arglist.agent, arglist.exp_name)
    print(f"evalusting online exp: {arglist.agent}/{arglist.exp_name}")
    
    # load data
    df_track = []
    for filename in arglist.filenames:
        df_track.append(load_data(arglist.data_path, arglist.scenario, filename, load_raw=True))
    df_track = pd.concat(df_track, axis=0)
    df_track = update_record_id(df_track)
    if not arglist.test_on_train:
        df_track["is_train"] = 1 - df_track["is_train"]

    # filter invalid lanes
    test_lanes = [int(l) for l in arglist.test_lanes]
    is_train = df_track["is_train"].values
    is_train[df_track["ego_lane_id"].isin(test_lanes) == False] = np.nan
    df_track = df_track.assign(is_train=is_train)
    
    # reassign test eps id
    eps_id = df_track["eps_id"]
    eps_id[df_track["is_train"] != 1] = np.nan
    df_track = df_track.assign(eps_id=eps_id)

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
    
    # init simulator
    feature_set = config["feature_set"]
    action_set = config["action_set"]

    ego_sensor = EgoSensor(map_data)
    lv_sensor = LeadVehicleSensor(map_data, track_lv=True)
    lidar_sensor = LidarSensor()
    sensors = [ego_sensor, lv_sensor, lidar_sensor]

    if (action_set == ["dds"] or action_set == ["dds_smooth"]):
        observer = CarfollowObserver(map_data, sensors, feature_set=feature_set)
    else:
        observer = Observer(map_data, sensors, feature_set=feature_set, action_set=action_set)
    
    env = InteractionSimulator(map_data, sensors, observer, svt)
    
    # init dataset for inference model
    dataset = RelativeDataset(
        df_track, feature_set, action_set, train_labels_col="is_train", max_eps=1000, seed=arglist.seed
    )
    print(f"total test episodes: sim {env.num_eps}, dataset {len(dataset)}")

    # init agent
    obs_dim = len(feature_set)
    ctl_dim = len(action_set)
    print(f"feature set: {feature_set}")
    print(f"action set: {action_set}")

    if arglist.agent == "vin":
        agent = VINAgent(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["hmm_rank"], 
            config["horizon"], alpha=config["alpha"], beta=config["beta"], obs_model=config["obs_model"], 
            obs_cov=config["obs_cov"], ctl_cov=config["ctl_cov"], rwd=config["rwd"], detach=config["detach"]
        )
    
    elif arglist.agent == "hvin":
        agent = HyperVINAgent(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["hmm_rank"],
            config["horizon"], config["hyper_dim"], config["hidden_dim"], config["num_hidden"],
            config["gru_layers"], config["activation"], alpha=config["alpha"], beta=config["beta"],
            obs_model=config["obs_model"], obs_cov=config["obs_cov"], ctl_cov=config["ctl_cov"], 
            rwd=config["rwd"], hyper_cov=config["hyper_cov"]
        )
    
    elif arglist.agent == "rnn":
        agent = RNNAgent(
            obs_dim, ctl_dim, config["act_dim"], config["hidden_dim"], 
            config["num_hidden"], config["gru_layers"], config["activation"]
        )

    elif arglist.agent == "mlp":
        agent = MLPAgent(
            obs_dim, ctl_dim, config["act_dim"], config["hidden_dim"],
            config["num_hidden"], config["activation"]
        )

    elif arglist.agent == "idm":
        agent = IDM(feature_set)

    # load state dict
    state_dict = torch.load(os.path.join(exp_path, "model.pt"), map_location=torch.device("cpu"))
    state_dict = state_dict if "model_state_dict" not in state_dict.keys() else state_dict["model_state_dict"]
    state_dict = {k.replace("agent.", ""): v for (k, v) in state_dict.items() if "agent." in k}
    agent.load_state_dict(state_dict, strict=False)
    agent.eval()
    print(agent)
    print(f"num parameters: {count_parameters(agent)}")
    
    # sample eval episodes
    if len(arglist.eps_ids) > 0:
        test_eps_id = [int(i) for i in arglist.eps_ids]
    else:
        test_eps_id = np.random.choice(
            np.arange(env.num_eps), min(arglist.num_eps, env.num_eps), replace=False
        )
    print("test episodes:", test_eps_id)
    
    # eval loop
    data = []
    maes = []
    animations = []
    for i, eps_id in enumerate(test_eps_id):
        if arglist.agent == "hvin":
            if arglist.test_posterior:
                z = compute_latent(dataset[eps_id], agent)
            else:
                z = agent.get_prior_dist().mean
        else:
            z = None
        
        sim_data, rewards = eval_episode(
            env, agent, eps_id, z=z, sample_method=arglist.sample_method, playback=arglist.playback
        )
        
        data.append(sim_data)
        maes.append(np.mean(rewards))
        if arglist.save_video:
            ani = animate(map_data, sim_data, title=f"eps {eps_id}", plot_lidar=False)
            animations.append(ani)

        print(f"test eps: {i}, mean reward: {np.mean(rewards)}")
    
    print(f"iqm: {compute_interquartile_mean(np.array(maes)):.4f}")

    # save results
    if arglist.save_summary or arglist.save_data or arglist.save_video:
        post_fix = "_post" if arglist.test_posterior else ""
        lane_fix = "_lanes_{}".format(",".join(arglist.test_lanes))
        eval_path = os.path.join(exp_path, "eval_online")
        save_path = os.path.join(eval_path, f"{arglist.seed}_{arglist.sample_method}{post_fix}{lane_fix}")
        if not os.path.exists(eval_path):
            os.mkdir(eval_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        if arglist.save_summary:
            result_dict = {
                "maes": maes,
                "test_lanes": test_lanes,
                "test_posterior": arglist.test_posterior
            }
            with open(os.path.join(save_path, "summary.json"), "w") as f:
                json.dump(result_dict, f)
            
        if arglist.save_data:
            filename = "sim_data.p"
            if arglist.playback:
                filename = "sim_data_playback.p"
            with open(os.path.join(save_path, filename), "wb") as f:
                pickle.dump(data, f)
        
        if arglist.save_video:
            for i, ani in enumerate(animations):
                save_animation(ani, os.path.join(save_path, f"test_scene_{i}_ani.mp4"))
        
        print("\nonline evaluation results saved at {}".format(save_path))

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)