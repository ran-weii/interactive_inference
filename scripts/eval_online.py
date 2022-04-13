import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch 
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from src.map_api.lanelet import MapReader
from src.data.ego_dataset import EgoDataset, RelativeDataset
from src.agents.active_inference import ActiveInference
from src.irl.algorithms import MLEIRL
from src.simulation.simulator import InteractionSimulator
from src.simulation.observers import RelativeObserver
from src.visualization.animation import animate, save_animation

import warnings
warnings.filterwarnings("ignore")

# set plotting style
strip_size = 12
label_size = 12
mpl.rcParams["axes.labelsize"] = label_size
mpl.rcParams["xtick.labelsize"] = strip_size
mpl.rcParams["ytick.labelsize"] = strip_size
mpl.rcParams["legend.title_fontsize"] = strip_size
mpl.rcParams["legend.fontsize"] = strip_size
mpl.rcParams["axes.titlesize"] = label_size
mpl.rcParams["figure.titlesize"] = label_size

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../interaction-dataset-master"
    )
    parser.add_argument(
        "--lanelet_path", type=str, default="../exp/lanelet"
    )
    parser.add_argument(
        "--exp_path", type=str, default="../exp/"
    )
    parser.add_argument("--scenario", type=str, default="DR_CHN_Merging_ZS")
    parser.add_argument("--filename", type=str, default="vehicle_tracks_007.csv")
    parser.add_argument("--method", type=str, choices=["mleirl", "il", "birl"], 
        default="active_inference", help="algorithm, default=mleirl")
    parser.add_argument("--exp_name", type=str, default="03-10-2022 10-52-38")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def collate_fn(batch):
    pad_obs = pad_sequence([b["ego"] for b in batch])
    pad_act = pad_sequence([b["act"] for b in batch])
    mask = torch.all(pad_obs != 0, dim=-1).to(torch.float32)
    return pad_obs, pad_act, mask

def eval_offline(track_data, agent):
    agent.eval()
    
    o = track_data["ego"].unsqueeze(1)
    u = track_data["act"].unsqueeze(1)
    
    u_batch = agent.choose_action(o, u, batch=True)

    # sequential eval
    agent.reset()
    b = [torch.empty(0)] * len(o)
    a = [torch.empty(0)] * len(o)
    u_seq = torch.zeros(len(o), 2)
    for t in range(len(o)):
        o_t = o[t]
        u_t = u[t] if t > 0 else torch.zeros_like(u)[0]
        u_seq[t] = agent.choose_action(o_t, u_t).data
        b[t] = agent._b
        a[t] = agent._a
    b = torch.stack(b, dim=0).squeeze(1)
    a = torch.stack(a, dim=0).squeeze(1)
    return u.data.squeeze(-2), u_batch.data.squeeze(-2), u_seq

def eval_episode(env, observer, agent, eps_id, data, control_direction, max_steps=1000):
    agent.eval()
    agent.reset()
    
    ctl_data = data["act"]
    a = []
    b = []

    obs_env = env.reset(eps_id)
    obs = observer.observe(obs_env)
    ctl_agent = torch.zeros(1, 2) if control_direction == "both" else torch.zeros(1, 1)
    for t in range(max_steps):
        ctl_agent = agent.choose_action(obs, ctl_agent).view(-1)
        
        if control_direction == "both":
            ctl = ctl_agent
        elif control_direction == "lon":
            ctl = torch.tensor([ctl_agent[0], ctl_data[t][1]])
        elif control_direction == "lat":
            ctl = torch.tensor([ctl_data[t][0], ctl_agent[0]])
        
        ctl_env = observer.control(ctl, obs_env)
        obs_env, r, done, info = env.step(ctl_env)
        if done:
            break
        obs = observer.observe(obs_env)

        # collect agent states
        b.append(agent._b)
        a.append(agent._a)
    
    b = torch.cat(b, dim=0).data.numpy()
    a = torch.cat(a, dim=0).data.numpy()
    states = env._states[:t+1]
    acts = env._acts[:t+1]
    track = {k:v[:t+1] for (k, v) in env._track.items()}
    return states, acts, track, [b, a]

def plot_map_trajectories(map_data, states, track, title):
    fig, ax = map_data.plot(option="ways", annot=False, figsize=(8, 4))
    ax.plot(states[:, 0], states[:, 1], "o", label="agent")
    ax.plot(track["ego"][:, 0], track["ego"][:, 1], "o", label="data")
    ax.legend()
    ax.set_title(title)
    return fig, ax

def plot_actions(acts, track, title):
    fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    ax[0].plot(acts[:, 0], label="agent")
    ax[0].plot(track["act"][:, 0], label="data")
    ax[0].set_ylabel("lon (m/s^2)")
    ax[0].legend()
    ax[0].set_title(title)

    ax[1].plot(acts[:, 1], label="agent")
    ax[1].plot(track["act"][:, 1], label="data")
    ax[1].set_xlabel("time (0.1 s)")
    ax[1].set_ylabel("lat (m/s^2)")
    ax[1].legend()
    plt.tight_layout()
    return fig, ax

def main(arglist):
    exp_path = os.path.join(arglist.exp_path, arglist.method, arglist.exp_name)
    print(f"evalusting exp: {arglist.exp_name}")

    # load args
    with open(os.path.join(exp_path, "args.json"), "rb") as f:
        config = json.load(f)

    # load raw data
    df_processed = pd.read_csv(
        os.path.join(arglist.data_path, "processed_trackfiles", arglist.scenario, arglist.filename)
    )
    df_track = pd.read_csv(
        os.path.join(arglist.data_path, "recorded_trackfiles", arglist.scenario, arglist.filename)
    )
    df_track = df_track.merge(df_processed, on=["track_id", "frame_id"])
    df_track["psi_rad"] = np.clip(df_track["psi_rad"], -np.pi, np.pi)
    
    # load test labels
    df_train_labels = pd.read_csv(
        os.path.join(arglist.data_path, "test_labels", arglist.scenario, arglist.filename)
    )
    df_train_labels["is_train"] = ~df_train_labels["is_train"]
    
    # load map
    map_data = MapReader()
    map_data.parse(os.path.join(arglist.data_path, "maps", arglist.scenario + ".osm"))
    
    # load state dict
    state_dict = torch.load(os.path.join(exp_path, "model.pt"))
    torch.manual_seed(config["seed"])

    dataset = EgoDataset(
        df_track, map_data, df_train_labels, 
        min_eps_len=config["min_eps_len"], max_eps_len=1000,
    )
    env = InteractionSimulator(dataset, map_data)
    observer = RelativeObserver(map_data)
    
    obs_dim = 10
    ctl_dim = 2 if config["control_direction"] == "both" else 1
    # load model
    if arglist.method == "mleirl":
        agent = ActiveInference(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["horizon"],
            obs_model=config["obs_model"], obs_dist=config["obs_dist"], obs_cov=config["obs_cov"], 
            ctl_model=config["ctl_model"], ctl_dist=config["ctl_dist"], ctl_cov=config["ctl_cov"]
        )
        model = MLEIRL(agent)
    else:
        raise NotImplementedError
    
    try:
        model.load_state_dict(state_dict)
    except:
        raise RuntimeError("Error loading state dict")

    agent = model.agent
    
    rel_dataset = RelativeDataset(
        df_track, map_data, df_train_labels, 
        min_eps_len=config["min_eps_len"], max_eps_len=1000,
    )

    # simulate
    test_eps_id = [121, 109, 119, 93]
    sim_data = []
    map_figs = []
    ctl_figs = []
    animations = []
    for i in test_eps_id:
        rel_track = rel_dataset[i]

        states, ctl, track, out = eval_episode(env, observer, agent, i, rel_track, config["control_direction"])
        
        title = f"track_{track['meta'][2]}_id_{i} agent control: {config['control_direction']}"
        ani = animate(map_data, states, track, title=title)
        fig_map, _ = plot_map_trajectories(map_data, states, track, title)
        fig_act, _ = plot_actions(ctl, track, title)
        
        sim_data.append({"states": states, "ctl": ctl, "track": track, "b": out[0], "a": out[1]})
        map_figs.append(fig_map)
        ctl_figs.append(fig_act)
        animations.append(ani)
        # break
    
    # save results
    if arglist.save:
        save_path = os.path.join(exp_path, "eval_offline")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        for i, (dat, fig_m, fig_a, ani) in enumerate(zip(sim_data, map_figs, ctl_figs, animations)):
            with open(os.path.join(save_path, f"test_scene_{i}_online_dat.p"), "wb") as f:
                pickle.dump(dat, f)

            fig_m.savefig(os.path.join(save_path, f"test_scene_{i}_online_map.png"), dpi=100)
            fig_a.savefig(os.path.join(save_path, f"test_scene_{i}_online_ctl.png"), dpi=100)
            save_animation(ani, os.path.join(save_path, f"test_scene_{i}_online_ani.mp4"))
        
        print("\noffline evaluation results saved at "
              "./exp/mleirl/{}/eval_offline".format(
            arglist.exp_name
        ))

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)
