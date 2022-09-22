import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch 

from src.map_api.lanelet import MapReader
from src.data.data_filter import filter_lane, filter_tail_merging
from src.data.ego_dataset import EgoDataset, RelativeDataset
from src.agents.active_inference import ActiveInference
from src.agents.embedded_agent import EmbeddedActiveInference
from src.agents.baseline import (
    StructuredRecurrentAgent, FullyRecurrentAgent, 
    HMMRecurrentAgent, StructuredHMMRecurrentAgent)
from src.agents.recurrent_agent import RecurrentActiveInference
from src.irl.algorithms import (
    MLEIRL, ImitationLearning, ReverseKL)
from src.irl.implicit_irl import ImplicitIRL
from src.simulation.simulator import InteractionSimulator
from src.simulation.observers import RelativeObserver
from src.simulation.controllers import AgentWrapper
from src.evaluation.online import Evaluator, eval_episode
from src.visualization.animation import animate, save_animation
from src.visualization.inspection import ModelExplainer

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
    parser.add_argument("--method", type=str, choices=["mleirl", "il", "birl", "srnn", "frnn", "hrnn", "shrnn", "eai", "rkl", "cirl"], 
        default="active_inference", help="algorithm, default=mleirl")
    parser.add_argument("--exp_name", type=str, default="03-10-2022 10-52-38")
    parser.add_argument("--explain_trajectory", type=bool_, default=True)
    parser.add_argument("--control_method", type=str, choices=["bma", "ace", "acm", "data"], default="ace", help="agent control sampling method, default=ace")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def plot_map_trajectories(map_data, states, track, title):
    fig, ax = map_data.plot(option="ways", annot=False, figsize=(8, 4))
    ax.plot(track["ego"][:, 0], track["ego"][:, 1], "o", alpha=0.6, label="data")
    ax.plot(states[:, 0], states[:, 1], ".", label="agent")
    ax.legend()
    ax.set_title(title)
    return fig, ax

def plot_action_trajectory(acts, track, title):
    fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    ax[0].plot(track["act"][:, 0], label="data")
    ax[0].plot(acts[:, 0], label="agent")    
    ax[0].set_ylabel("lon (m/s^2)")
    ax[0].legend()
    ax[0].set_title(title)
    
    ax[1].plot(track["act"][:, 1], label="data")
    ax[1].plot(acts[:, 1], label="agent")
    ax[1].set_xlabel("time (0.1 s)")
    ax[1].set_ylabel("lat (m/s^2)")
    ax[1].legend()
    plt.tight_layout()
    return fig, ax

def plot_agent_model(explainer):
    fig_cd, ax = plt.subplots(1, 4, figsize=(12, 3))
    explainer.plot_C(ax[0])
    explainer.plot_D(ax[1])
    explainer.plot_S(ax[2])
    explainer.plot_tau(ax[3])
    plt.tight_layout()
    
    fig_ab, ax = plt.subplots(1, 3, figsize=(15, 8))
    explainer.plot_B_pi(ax[0], annot=False)
    explainer.plot_A(ax[1], ax[2], annot=True)
    plt.tight_layout()
    return fig_cd, fig_ab

def plot_belief_trajectory(explainer, sim_data):
    obs_keys = []
    fig, _ = explainer.plot_episode(sim_data, obs_keys, figsize=(10, 10))
    return fig

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
    
    # filter data
    lane_id = [1, 6]
    max_psi_error = 0.05
    df_track = filter_lane(df_track, lane_id)
    df_track = filter_tail_merging(df_track, max_psi_error=max_psi_error, min_bound_dist=1)

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
    
    frame = "carte" if "frame" not in config.keys() else config["frame"]
    obs_fields = "rel" if "obs_fields" not in config.keys() else config["obs_fields"]
    observer = RelativeObserver(map_data, frame=frame, fields=obs_fields)
    dataset = EgoDataset(
        df_track, observer, df_train_labels, 
        min_eps_len=config["min_eps_len"], max_eps_len=1000,
    )
    env = InteractionSimulator(dataset, map_data)
    
    obs_dim = len(observer.ego_fields)
    ctl_dim = 2 if config["control_direction"] == "both" else 1

    # load model
    rwd_model = "efe" if "rwd_model" not in config.keys() else config["rwd_model"]
    hmm_rank = 0 if "hmm_rank" not in config.keys() else config["hmm_rank"]

    if arglist.method == "mleirl":
        agent = ActiveInference(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["horizon"],
            obs_model=config["obs_model"], obs_dist=config["obs_dist"], obs_cov=config["obs_cov"], 
            ctl_model=config["ctl_model"], ctl_dist=config["ctl_dist"], ctl_cov=config["ctl_cov"],
            rwd_model=rwd_model, hmm_rank=hmm_rank, planner=config["planner"], tau=config["tau"], hidden_dim=config["hidden_dim"], 
            num_hidden=config["num_hidden"], activation=config["activation"]
        )
        model = MLEIRL(agent)
    elif arglist.method == "rkl":
        agent = ActiveInference(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["horizon"],
            obs_model=config["obs_model"], obs_dist=config["obs_dist"], obs_cov=config["obs_cov"], 
            ctl_model=config["ctl_model"], ctl_dist=config["ctl_dist"], ctl_cov=config["ctl_cov"],
            rwd_model=rwd_model, hmm_rank=hmm_rank, planner=config["planner"], tau=config["tau"], hidden_dim=config["hidden_dim"], 
            num_hidden=config["num_hidden"], activation=config["activation"]
        )
        model = ReverseKL(agent)
    elif arglist.method == "eai":
        agent = EmbeddedActiveInference(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["horizon"],
            obs_model=config["obs_model"], obs_dist=config["obs_dist"], obs_cov=config["obs_cov"], 
            ctl_model=config["ctl_model"], ctl_dist=config["ctl_dist"], ctl_cov=config["ctl_cov"],
            planner=config["planner"], tau=config["tau"], hidden_dim=config["hidden_dim"], 
            num_hidden=config["num_hidden"], activation=config["activation"]
        )
        model = MLEIRL(agent)
    elif arglist.method == "frnn":
        agent = FullyRecurrentAgent(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["horizon"],
            ctl_dist=config["ctl_dist"], ctl_cov=config["ctl_cov"],
            hidden_dim=config["hidden_dim"], num_hidden=config["num_hidden"]
        )
        model = MLEIRL(agent)
    elif arglist.method == "srnn":
        agent = StructuredRecurrentAgent(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["horizon"],
            ctl_dist=config["ctl_dist"], ctl_cov=config["ctl_cov"],
            hidden_dim=config["hidden_dim"], num_hidden=config["num_hidden"]
        )
        model = MLEIRL(agent)
    elif arglist.method == "hrnn":
        agent = HMMRecurrentAgent(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["horizon"],
            ctl_dist=config["ctl_dist"], ctl_cov=config["ctl_cov"],
            hidden_dim=config["hidden_dim"], num_hidden=config["num_hidden"]
        )
        model = MLEIRL(agent)
    elif arglist.method == "shrnn":
        agent = StructuredHMMRecurrentAgent(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["horizon"],
            ctl_dist=config["ctl_dist"], ctl_cov=config["ctl_cov"],
            hidden_dim=config["hidden_dim"], num_hidden=config["num_hidden"]
        )
        model = ImitationLearning(agent)
    elif arglist.method == "cirl":
        agent = RecurrentActiveInference(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["horizon"],
            obs_model=config["obs_model"], obs_dist=config["obs_dist"], obs_cov=config["obs_cov"], 
            ctl_model=config["ctl_model"], ctl_dist=config["ctl_dist"], ctl_cov=config["ctl_cov"],
            rwd_model=rwd_model, hmm_rank=hmm_rank, planner=config["planner"], tau=config["tau"], hidden_dim=config["hidden_dim"], 
            num_hidden=config["num_hidden"], activation=config["activation"]
        )
        model = ImplicitIRL(agent)
    else:
        raise NotImplementedError
    
    try:
        model.load_state_dict(state_dict, strict=False)
    except:
        raise RuntimeError("Error loading state dict")

    agent = model.agent
    print(agent)
    num_params = 0
    for n, p in model.named_parameters():
        if p.requires_grad and "tl" not in n:
            num_params += torch.prod(torch.tensor(list(p.shape)))
    print("num params", num_params)
    
    rel_dataset = RelativeDataset(
        df_track, observer, df_train_labels, 
        min_eps_len=config["min_eps_len"], max_eps_len=1000,
    )

    # simulate
    controller = AgentWrapper(observer, agent, arglist.control_method)
    # test_eps_id = [121, 109, 119, 93]
    # test_eps_id = [121, 158, 48, 109, 98, 124, 119, 82, 67, 93, 110, 33]
    test_eps_id = [105, 139, 41, 107, 88, 84, 100, 109, 52, 82, 97, 29]
    sim_data = []
    map_figs = []
    ctl_figs = []
    animations = []
    for i in test_eps_id:
        rel_track = rel_dataset[i]
        
        sim_states, sim_acts, track_data = eval_episode(env, controller, i)
        title = f"track_{track_data['meta'][2]}_id_{i} agent control: {config['control_direction']}"
        ani = animate(map_data, sim_states, track_data, title=title)
        fig_map, _ = plot_map_trajectories(map_data, sim_states, track_data, title)
        fig_act, _ = plot_action_trajectory(sim_acts, track_data, title)
        
        sim_data.append({"sim_states": sim_states, "sim_acts": sim_acts, "track_data": track_data})
        map_figs.append(fig_map)
        ctl_figs.append(fig_act)
        animations.append(ani)
        break
    
    # explain active inference agent
    if arglist.method == "mleirl" and arglist.explain_trajectory:
        explainer = ModelExplainer(agent, observer.ego_fields)
        explainer.sort(by="C")
        belief_figs = []
        for data in sim_data:
            belief_figs.append(plot_belief_trajectory(explainer, data))

    # save results
    if arglist.save:
        eval_path = os.path.join(exp_path, "eval_online")
        save_path = os.path.join(eval_path, f"{arglist.seed}_{arglist.control_method}")
        # if arglist.diagnostic:
        #     save_path += "_diagnostic"

        if not os.path.exists(eval_path):
            os.mkdir(eval_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        for i, (dat, fig_m, fig_a, ani) in enumerate(zip(sim_data, map_figs, ctl_figs, animations)):
            with open(os.path.join(save_path, f"test_scene_{i}_online_dat.p"), "wb") as f:
                pickle.dump(dat, f)

            fig_m.savefig(os.path.join(save_path, f"test_scene_{i}_online_map.png"), dpi=100)
            fig_a.savefig(os.path.join(save_path, f"test_scene_{i}_online_ctl.png"), dpi=100)
            save_animation(ani, os.path.join(save_path, f"test_scene_{i}_online_ani.mp4"))
        
        if arglist.method == "mleirl" and arglist.explain_trajectory:
            for i, f in enumerate(belief_figs):
                f.savefig(os.path.join(save_path, f"test_scene_{i}_online_belief.png"), dpi=100)

        print("\nonline evaluation results saved at "
              "./exp/mleirl/{}/eval_offline".format(
            arglist.exp_name
        ))

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)
