import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import bokeh

import torch 
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from src.map_api.lanelet import MapReader
from src.data.data_filter import filter_lane, filter_tail_merging
from src.simulation.observers import RelativeObserver
from src.data.ego_dataset import RelativeDataset
from src.agents.active_inference import ActiveInference
from src.agents.embedded_agent import EmbeddedActiveInference
from src.agents.baseline import (
    StructuredRecurrentAgent, FullyRecurrentAgent, 
    HMMRecurrentAgent, StructuredHMMRecurrentAgent)
from src.agents.recurrent_agent import RecurrentActiveInference
from src.irl.algorithms import (
    MLEIRL, ImitationLearning, ReverseKL)
from src.irl.implicit_irl import ImplicitIRL
from src.evaluation.offline_metrics import (
    mean_absolute_error, threshold_relative_error)
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
    parser.add_argument("--explain_agent", type=bool_, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def collate_fn(batch):
    pad_obs = pad_sequence([b["ego"] for b in batch])
    pad_act = pad_sequence([b["act"] for b in batch])
    mask = torch.any(pad_obs != 0, dim=-1).to(torch.float32)
    return pad_obs, pad_act, mask

def eval_epoch(agent, loader, num_samples=10):
    agent.eval()
    
    # get data and predictions
    u_true, u_pred, u_sample, obs, masks = [], [], [], [], []
    for i, batch in enumerate(loader):
            o, u, mask = batch
            with torch.no_grad():
                ctl = agent.choose_action(o, u, batch=True)
                ctl_sample = agent.choose_action(o, u, batch=True, num_samples=num_samples)
            
            u_true.append(u)
            u_pred.append(ctl)
            u_sample.append(ctl_sample)
            obs.append(o)
            masks.append(mask)
    
    """ TODO: add padding for batch evaluation """
    u_true = torch.cat(u_true, dim=1).data.numpy()
    u_pred = torch.cat(u_pred, dim=1).data.numpy()
    u_sample = torch.cat(u_sample, dim=2).data.numpy()
    obs = torch.cat(obs, dim=1).data.numpy()
    masks = torch.cat(masks, dim=1).data.numpy()
    return u_true, u_pred, u_sample, obs, masks

def sample_trajectory_by_cluster(dataset, num_samples, sample=False, seed=0):
    """
    Args:
        dataset (torch.dataset): relative dataset
        num_samples (int): number of episode samples per cluster
        sample (bool, optional): sample randomly or take head. Defaults to False.
        seed (int, optional): random seed. Defaults to 0.

    Returns:
        df_eps (pd.dataframe): dataframe with fields [cluster, eps_id, idx]
    """
    df_track = dataset.df_track.loc[dataset.df_track["eps_id"] != -1]
    unique_clusters = np.unique(df_track["cluster"])
    unique_clusters = unique_clusters[np.isnan(unique_clusters) == False]
    
    # get episode header
    merge_keys = ["scenario", "record_id", "track_id"]
    df_track_head = df_track.groupby(merge_keys).head(1)
    df_track_head = df_track_head.loc[df_track_head["is_train"] == True]
    
    df_eps = df_track_head.dropna(subset=["cluster"]).groupby("cluster")
    if sample:
        df_eps = df_eps.sample(n=num_samples, random_state=seed)
    else:
        df_eps = df_eps.head(num_samples)
    
    eps_id = df_eps["eps_id"].values
    idx = [np.where(dataset.unique_eps == i)[0][0] for i in eps_id]
    
    df_eps = df_eps[merge_keys + ["cluster", "eps_id"]].reset_index(drop=True)
    df_eps = df_eps.assign(idx=idx)
    return df_eps

""" TODO: temporary plotting solution """
def plot_action_trajectory(u_true, u_pred, u_sample, mask, title="", figsize=(6, 4.5)):
    # get u_sample stats
    u_mu = np.mean(u_sample, axis=0)
    u_std = np.std(u_sample, axis=0)
    
    # mask actions
    nan_mask = np.expand_dims(mask, axis=-1).copy()
    nan_mask[nan_mask == 0] = float("nan")
    u_true *= nan_mask
    u_pred *= nan_mask
    u_mu *= nan_mask
    u_std *= nan_mask
    
    n_rows = u_true.shape[-1]
    fig, ax = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    if n_rows == 1:
        ax = [ax]
    
    time = np.arange(len(u_mu))/10
    for i, x in enumerate(ax):
        x.plot(time, u_true[:, i], label="true")
        x.plot(time, u_pred[:, i], label="pred")
        x.fill_between(
            time,
            u_mu[:, i] + u_std[:, i], 
            u_mu[:, i] - u_std[:, i], 
            alpha=0.4,
            label="1std"
        )
        x.set_xlabel("time (s)")
        x.set_ylabel(f"u_{i} (m/s^2)")
        x.legend()
    
    ax[0].set_title(title)
            
    plt.tight_layout()
    return fig, ax

def plot_action_units(agent, figsize=(6, 4)):
    grid = torch.linspace(-2, 2, 300).view(-1, 1)
    grid_ = torch.stack([grid, grid]).permute(1, 2, 0)

    mean = agent.ctl_model.mean().data
    std = torch.sqrt(agent.ctl_model.variance()).data
    
    if agent.ctl_model.__class__.__name__ == "FactoredConditionalDistribution":
        dist = agent.ctl_model.get_distribution_class()
        pdf = dist.log_prob(grid_).exp().data
    else:
        pdf = torch.distributions.Normal(
            mean, std
        ).log_prob(grid_).exp().data
    
    font_size = 12
    n_rows = agent.ctl_dim
    fig, ax = plt.subplots(n_rows, 1, figsize=figsize)
    if n_rows == 1:
        ax = [ax]
    
    for i in range(n_rows):
        for j in range(pdf.shape[1]):
            ax[i].plot(grid, pdf[:, j, i])
        ax[i].set_xlabel(f"u_{i}", fontsize=font_size)
        ax[i].set_ylabel("pdf", fontsize=font_size)
    plt.tight_layout()
    return fig, ax

def plot_action_units_2d(agent, num_samples=100, figsize=(6, 6)):
    x = agent.ctl_model.sample((num_samples,)).squeeze(1).data.numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(x[:, :, 0], x[:, :, 1], "o", ms=4)
    ax.set_xlabel("u_0")
    ax.set_ylabel("u_1")
    return fig, ax

def plot_agent_model(explainer):
    fig_cd, ax = plt.subplots(1, 5, figsize=(14, 3))
    explainer.plot_C(ax[0])
    explainer.plot_r(ax[1])
    explainer.plot_D(ax[2])
    explainer.plot_S(ax[3])
    explainer.plot_tau(ax[4])
    plt.tight_layout()
    
    fig_ab, ax = plt.subplots(1, 3, figsize=(15, 8))
    explainer.plot_B_pi(ax[0], annot=False)
    explainer.plot_A(ax[1], ax[2], annot=True)
    plt.tight_layout()

    fig_f, ax = plt.subplots(1, 2, figsize=(8, 8))
    explainer.plot_F(ax[0], ax[1], annot=True)

    fig_pi, ax = plt.subplots(1, 1, figsize=(10, 8))
    explainer.plot_pi(ax, annot=False, cbar=True)
    return fig_cd, fig_ab, fig_f, fig_pi

def main(arglist):
    exp_path = os.path.join(arglist.exp_path, arglist.method, arglist.exp_name)
    print(f"evalusting exp: {arglist.exp_name}")
    
    # load args
    with open(os.path.join(exp_path, "args.json"), "rb") as f:
        config = json.load(f)
    
    # load state dict
    state_dict = torch.load(os.path.join(exp_path, "model.pt"))
    torch.manual_seed(config["seed"])
    
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
    
    frame = "carte" if "frame" not in config.keys() else config["frame"]
    obs_fields = "rel" if "obs_fields" not in config.keys() else config["obs_fields"]
    observer = RelativeObserver(map_data, frame=frame, fields=obs_fields)
    dataset = RelativeDataset(
        df_track, observer, df_train_labels=df_train_labels, 
        min_eps_len=config["min_eps_len"], max_eps_len=1000,
        control_direction=config["control_direction"]
    )
    test_loader = DataLoader(
        dataset, len(dataset), shuffle=False, 
        drop_last=False, collate_fn=collate_fn
    )
    
    # get dimensions from data 
    [obs, ctl, mask] = next(iter(test_loader))
    obs_dim = obs.shape[-1]
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
    elif arglist.method == "il":
        model = ImitationLearning(config["act_dim"], obs_dim, ctl_dim)
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
    
    model.load_state_dict(state_dict, strict=False)
    agent = model.agent

    print(dataset.ego_fields)
    print(agent)
    
    u_true, u_pred, u_sample, obs, masks = eval_epoch(agent, test_loader, num_samples=30)
    
    """ TODO: temporary solution for lateral control"""
    # add padding for metrics calculation
    if config["control_direction"] == "lon":
        u_true_pad = np.concatenate([u_true, np.zeros_like(u_true)], axis=-1)
        u_pred_pad = np.concatenate([u_pred, np.zeros_like(u_pred)], axis=-1)
    elif config["control_direction"] == "lat":
        u_true_pad = np.concatenate([np.zeros_like(u_true), u_true], axis=-1)
        u_pred_pad = np.concatenate([np.zeros_like(u_pred), u_pred], axis=-1)
    else:
        u_true_pad = u_true
        u_pred_pad = u_pred
        
    # get speed
    ego_fields = test_loader.dataset.ego_fields
    id_speed = [i for i, f in enumerate(ego_fields) if f in ["vx_ego", "vy_ego"]]
    speed = np.take(obs, id_speed, axis=-1)
    
    # compute offline metrics
    mae = mean_absolute_error(
        u_true_pad, u_pred_pad, mask=masks, speed=None, cumulative=False
    ).tolist()
    mae_s = mean_absolute_error(
        u_true_pad, u_pred_pad, mask=masks, speed=speed, cumulative=False
    ).tolist()
    mae_sc = mean_absolute_error(
        u_true_pad, u_pred_pad, mask=masks, speed=speed, cumulative=True
    ).tolist()
    tre = threshold_relative_error(
        u_true_pad, u_pred_pad, mask=masks, alpha=0.1
    ).tolist()
    metrics_dict = {"mae": mae, "mae_s": mae_s, "mae_sc": mae_sc, "tre": tre}
    
    # plot sample test scenes
    num_samples = 3
    df_eps = sample_trajectory_by_cluster(
        dataset, num_samples, sample=True, seed=arglist.seed
    )
    acc_figs = []
    for i in range(len(df_eps)):
        idx = df_eps.iloc[i]["idx"]
        track_id = df_eps.iloc[i]["track_id"]
        cluster_id = df_eps.iloc[i]["cluster"]
        title = f"track_{track_id}_id_{idx}_cluster_{cluster_id:.0f} agent control: {config['control_direction']}"
        fig_acc, ax = plot_action_trajectory(
            u_true[:, idx], u_pred[:, idx], u_sample[:, :, idx], masks[:, idx], title=title
        )
        acc_figs.append(fig_acc)
        # print(idx, track_id)
    
    # explain active inference agent
    if arglist.method == "mleirl" and arglist.explain_agent:
        explainer = ModelExplainer(agent, dataset.ego_fields)
        explainer.sort(by="C")
        fig_agent = plot_agent_model(explainer)
        fig_action, ax = plot_action_units(agent)

    # save results
    if arglist.save:
        save_path = os.path.join(exp_path, "eval_offline")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        # save metrics
        with open(os.path.join(save_path, "metrics.json"), "w") as f:
            json.dump(metrics_dict, f)
        
        # save figures
        for i, fig in enumerate(acc_figs):
            fig.savefig(os.path.join(save_path, f"test_scene_{i}_offline_ctl.png"), dpi=100)
        
        if arglist.method == "mleirl" and arglist.explain_agent:
            fig_action.savefig(os.path.join(save_path, "action_units.png"), dpi=100)
            for i, fig in enumerate(fig_agent):
                fig.savefig(os.path.join(save_path, f"agent_model_{i}.png"), dpi=100)
            
        print("\noffline evaluation results saved at "
              "./exp/mleirl/{}/eval_offline".format(
            arglist.exp_name
        ))

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)