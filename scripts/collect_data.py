import argparse
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# set up imports
from src.data.train_utils import load_data
from src.data.data_filter import filter_segment_by_length
from src.map_api.lanelet import MapReader
from src.simulation.simulator import InteractionSimulator
from src.simulation.sensors import EgoSensor, LeadVehicleSensor, LidarSensor
from src.simulation.observers import CarfollowObserver
from src.simulation.utils import create_svt_from_df
from src.evaluation.online import eval_episode

# model imports
from src.agents.vin_agent import VINAgent

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    str_list_ = lambda x: x.replace(" ", "").split(",")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../interaction-dataset-master"
    )
    parser.add_argument("--scenario", type=str, default="DR_CHN_Merging_ZS")
    parser.add_argument("--filenames", type=str_list_, 
        default=["vehicle_tracks_003.csv"])
    parser.add_argument("--min_eps_len", type=int, default=100,help="min episode length, default=100")
    parser.add_argument("--state_dim", type=int, default=10, help="random vin agent state dimension, default=10")
    parser.add_argument("--act_dim", type=int, default=15, help="random vin agent action dimension, default=15")
    parser.add_argument("--feature_set", type=int, default=["ego_ds", "lv_s_rel", "lv_ds_rel", "lv_inv_tau"])
    parser.add_argument("--action_set", type=int, default=["dds"])
    parser.add_argument("--num_eps", type=int, default=1, help="number of episodes to collect, default=3")
    parser.add_argument("--max_steps", type=int, default=300, help="max episode steps, default=300")
    parser.add_argument("--out_filename", type=str, default="vehicle_tracks_015.csv")
    parser.add_argument("--seed", type=int, default=0)
    arglist = parser.parse_args()
    return arglist

def get_new_eps_id(df_track):
    """ Combine record_id and eps_id """
    df_track["record_id"] = df_track["record_id"].astype(int)
    num_digits = len(str(np.nanmax(df_track["eps_id"]).astype(int)))
    new_eps_id = df_track["record_id"] * 10**num_digits + df_track["eps_id"]
    return new_eps_id

def episode(env, agent, eps_id, max_steps=1000, sample_method="ace"):
    """ Evaluate episode
    
    Args:
        env (Env): gym style simulator
        agent (Agent): agent class
        eps_id (int): episode id
        max_steps (int, optional): maximum number of steps. Default=1000
        sample_method (str, optional): 

    Returns:
        sim_states (np.array): simulated states [T, num_agents, 5]
        sim_acts (np.array): simulated actions [T, 2]
        track_data (dict): recorded track data
    """
    agent.eval()
    agent.reset()
    obs = env.reset(eps_id)
    env.observer.push(env._state, agent._state)

    rewards = []
    for t in range(max_steps):
        with torch.no_grad():
            ctl, _ = agent.choose_action(obs.to(agent.device), sample_method=sample_method)
            ctl = ctl.cpu().data.view(1, -1)
            
        obs, r, done, info = env.step(ctl)
        
        if done or t >= max_steps or info["terminate"]:
            break
        rewards.append(r)
        env.observer.push(env._state, agent._state)
    
    return env.observer._data, rewards

def main(arglist):
    np.random.seed(arglist.seed)
    torch.manual_seed(arglist.seed)

    # load data files
    df_track = []
    for filename in arglist.filenames:
        df_track.append(load_data(arglist.data_path, arglist.scenario, filename))
    df_track = pd.concat(df_track, axis=0)
    df_track = df_track.assign(eps_id=get_new_eps_id(df_track))

    # filter episode length
    df_track["eps_id"], df_track["eps_len"] = filter_segment_by_length(
        df_track["eps_id"].values, arglist.min_eps_len
    )
    
    # build svt
    svt = create_svt_from_df(df_track, eps_id_col="eps_id")
    print(f"num trajectories loaded: {len(svt.eps_ids)}")

    # load map
    map_data = MapReader()
    map_data.parse(os.path.join(arglist.data_path, "maps", arglist.scenario + ".osm"))

    # define feature set
    ego_sensor = EgoSensor(map_data)
    lv_sensor = LeadVehicleSensor(map_data, track_lv=True)
    lidar_sensor = LidarSensor()
    sensors = [ego_sensor, lv_sensor, lidar_sensor]

    feature_set = arglist.feature_set
    action_set = arglist.action_set
    full_feature_set = ego_sensor.feature_names + lv_sensor.feature_names + lidar_sensor.feature_names
    
    observer = CarfollowObserver(map_data, sensors, feature_set=feature_set, max_s_rel=-2.)
    print(observer.feature_set)

    env = InteractionSimulator(map_data, sensors, observer, svt)
    
    # compute obs and ctl mean and variance stats
    obs_mean = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][feature_set].mean().values).to(torch.float32)
    obs_var = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][feature_set].var().values).to(torch.float32)
    ctl_mean = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].mean().values).to(torch.float32)
    ctl_var = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].var().values).to(torch.float32)

    # init agent
    obs_dim = len(observer.feature_set)
    ctl_dim = 1
    agent = VINAgent(
        arglist.state_dim, arglist.act_dim, obs_dim, ctl_dim, 10,
        5, alpha=1, beta=1, obs_cov="full", ctl_cov="diag", use_tanh=False
    )
    agent.obs_model.init_batch_norm(obs_mean, obs_var)
    agent.ctl_model.init_batch_norm(ctl_mean, ctl_var)
    
    # # add more weights to large actions in pi0
    # c = 4
    # agent._pi0.data += torch.linspace(-c, c, agent.act_dim).abs().view(1, agent.act_dim, 1)

    # load ctl gmm parameters
    with open(os.path.join("../exp", "agents", "ctl_model", "model.p"), "rb") as f:
        [ctl_means, ctl_covs, weights] = pickle.load(f)

    agent.ctl_model.init_params(ctl_means, ctl_covs)
    print("action model loaded")
    
    agent.eval()
    print(agent)
    
    # # sample eval episodes
    # test_eps_id = np.random.choice(np.arange(env.num_eps), arglist.num_eps)
    # print(test_eps_id)
    
    def pack_obs(episode):
        o = []
        for e in episode:
            sensor_obs = e["sim_state"]["sensor_obs"]
            o.append(np.hstack([o_.flatten() for o_ in sensor_obs.values()]))
        return np.stack(o)

    def pack_act(episode):
        a = []
        for e in episode[1:]:
            sim_act = e["sim_state"]["sim_act"]["act_local"]
            a.append(sim_act)
        return np.stack(a)
    
    # accept sample based on max lv_inv_tau value 
    min_lv_inv_tau = 1.

    df_features = []
    df_labels = []
    # for i, eps_id in enumerate(test_eps_id):
    while len(df_features) < arglist.num_eps:
        # sim_data, rewards = eval_episode(
        #     env, agent, eps_id, sample_method="ace", playback=False
        # )
        test_id = np.random.choice(np.arange(env.num_eps))
        sim_data, rewards = episode(
            env, agent, test_id, sample_method="ace"
        )
        print(f"test eps: {test_id}, mean reward: {np.mean(rewards)}, len: {len(rewards)}")
        
        # pack dataframe
        if len(rewards) > 50:
            track_id = len(df_features)
            obs = pack_obs(sim_data)
            act = pack_act(sim_data)
            df_eps_features = pd.DataFrame(np.hstack([obs[:-1], act]), columns=full_feature_set + action_set)
            df_eps_features.insert(0, "track_id", track_id * np.ones((len(act),)))
            df_eps_features.insert(1, "frame_id", np.arange(len(act)))
            
            labels = np.stack([
                track_id * np.ones(len(act)),
                np.arange(len(act)),
                track_id * np.ones(len(act)),
                len(act) * np.ones(len(act)),
                np.ones(len(act))
            ]).T
            df_eps_labels = pd.DataFrame(
                labels, columns=["track_id", "frame_id", "eps_id", "eps_len", "is_train"]
            )

            if df_eps_features["lv_inv_tau"].abs().max() >= min_lv_inv_tau:
                df_features.append(df_eps_features)
                df_labels.append(df_eps_labels)
                print(f"accepted trajectories: {len(df_features)}")
    
    print(f"num trajectories accepted: {len(df_features)}")

    df_features = pd.concat(df_features)
    df_labels = pd.concat(df_labels)

    # save 
    feature_save_path = os.path.join(
        arglist.data_path, 
        "processed_trackfiles", 
        "features", 
        arglist.scenario,
        arglist.out_filename
    )
    label_save_path = os.path.join(
        arglist.data_path, 
        "processed_trackfiles", 
        "train_labels", 
        arglist.scenario,
        arglist.out_filename
    )
    df_features.to_csv(feature_save_path, index=False)
    df_labels.to_csv(label_save_path, index=False)

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)