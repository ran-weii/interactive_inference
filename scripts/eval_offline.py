import argparse
import os
import json
import numpy as np
import pandas as pd
import bokeh

import torch 
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from src.data.lanelet import load_lanelet_df
from src.data.ego_dataset import RelativeDataset, EgoDataset
from src.irl.algorithms import MLEIRL, ImitationLearning
from src.evaluation.offline_metrics import (
    mean_absolute_error, threshold_relative_error)
from src.visualization.inspection import (
    get_active_inference_parameters, plot_active_inference_parameters)
from src.visualization.visualizer import build_bokeh_sources, visualize_scene

import warnings
warnings.filterwarnings("ignore")

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
    parser.add_argument("--method", type=str, choices=["mleirl", "il"], 
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

def eval_epoch(agent, loader):
    agent.eval()
    
    # get data and predictions
    u_true, u_pred, obs, masks = [], [], [], []
    for i, batch in enumerate(loader):
            o, u, mask = batch
            with torch.no_grad():
                ctl = agent.choose_action(o, u)
            
            u_true.append(u)
            u_pred.append(ctl)
            obs.append(o)
            masks.append(mask)
    
    """ TODO: add padding for batch evaluation """
    u_true = torch.cat(u_true, dim=1).data.numpy()
    u_pred = torch.cat(u_pred, dim=1).data.numpy()
    obs = torch.cat(obs, dim=1).data.numpy()
    masks = torch.cat(masks, dim=1).data.numpy()
    
    """ TODO: temporary solution for lateral control"""
    u_true = np.concatenate([u_true, np.zeros_like(u_true)], axis=-1)
    u_pred = np.concatenate([u_pred, np.zeros_like(u_pred)], axis=-1)
    
    # get speed
    ego_fields = loader.dataset.ego_fields
    id_speed = [i for i, f in enumerate(ego_fields) if f in ["vx_ego", "vy_ego"]]
    speed = np.take(obs, id_speed, axis=-1)
    
    # compute offline metrics
    mae = mean_absolute_error(
        u_true, u_pred, mask=masks, speed=None, cumulative=False
    ).tolist()
    mae_s = mean_absolute_error(
        u_true, u_pred, mask=masks, speed=speed, cumulative=False
    ).tolist()
    mae_sc = mean_absolute_error(
        u_true, u_pred, mask=masks, speed=speed, cumulative=True
    ).tolist()
    tre = threshold_relative_error(
        u_true, u_pred, mask=masks, alpha=0.1
    ).tolist()
    out = {"mae": mae, "mae_s": mae_s, "mae_sc": mae_sc, "tre": tre}
    return out, u_true, u_pred

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
    unique_clusters = dataset.df_track["cluster"].unique()
    unique_clusters = unique_clusters[np.isnan(unique_clusters) == False]
    
    # get episode header
    merge_keys = ["scenario", "record_id", "track_id"]
    df_track_head = dataset.df_track.groupby(merge_keys).head(1)
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

def main(arglist):
    exp_path = os.path.join(arglist.exp_path, arglist.method, arglist.exp_name)
    
    # load args
    with open(os.path.join(exp_path, "args.json"), "rb") as f:
        config = json.load(f)
    
    # load state dict
    state_dict = torch.load(os.path.join(exp_path, "model.pt"))
    torch.manual_seed(config["seed"])
    
    # load raw data
    df_lanelet = load_lanelet_df(os.path.join(arglist.lanelet_path, arglist.scenario + ".json"))
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
    
    dataset = RelativeDataset(
        df_track, df_lanelet, df_train_labels=df_train_labels, 
        min_eps_len=config["min_eps_len"], max_eps_len=1000,
        lateral_control=config["lateral_control"]
    )
    test_loader = DataLoader(
        dataset, len(dataset), shuffle=False, 
        drop_last=False, collate_fn=collate_fn
    )
    
    # get dimensions from data 
    [obs, ctl, mask] = next(iter(test_loader))
    obs_dim = obs.shape[-1]
    ctl_dim = ctl.shape[-1]
    
    # load model
    if arglist.method == "mleirl":
        model = MLEIRL(
            config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["horizon"],
            obs_dist=config["obs_dist"], obs_cov=config["obs_cov"], 
            ctl_dist=config["ctl_dist"], ctl_cov=config["ctl_cov"]
        )
    elif arglist.method == "il":
        model = ImitationLearning(config["act_dim"], obs_dim, ctl_dim)
    
    model.load_state_dict(state_dict)
    agent = model.agent
    
    metrics_dict, u_true, u_pred = eval_epoch(agent, test_loader)
    
    # plot sample test scenes
    num_samples = 1
    df_eps = sample_trajectory_by_cluster(
        dataset, num_samples, sample=True, seed=arglist.seed
    )
    ego_dataset = EgoDataset(
        df_track, df_lanelet, df_train_labels=df_train_labels, 
        min_eps_len=config["min_eps_len"], max_eps_len=1000
    )
    scene_figs = []
    for i in range(len(df_eps)):
        idx = df_eps.iloc[i]["idx"]
        track_data = ego_dataset[idx]
        frames, lanelet_source = build_bokeh_sources(
            track_data, df_lanelet, ego_dataset.ego_fields, ego_dataset.agent_fields,
            acc_true=u_true[:, idx], acc_pred=u_pred[:, idx]
        )
        track_id = df_eps.iloc[i]["track_id"]
        cluster_id = df_eps.iloc[i]["cluster"]
        fig = visualize_scene(
            frames, lanelet_source, title=f"track_{track_id}_cluster_{cluster_id:.0f}"
        )
        scene_figs.append(fig)
    
    # plot active inference parameters
    if arglist.method == "mleirl":
        theta_dict = get_active_inference_parameters(agent)
        fig_params, _ = plot_active_inference_parameters(theta_dict, cmap="inferno")

    # save results
    if arglist.save:
        save_path = os.path.join(exp_path, "eval_offline")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        # save metrics
        with open(os.path.join(save_path, "metrics.json"), "w") as f:
            json.dump(metrics_dict, f)
        
        # save figures
        if arglist.method == "mleirl":
            fig_params.savefig(os.path.join(save_path, "params.png"), dpi=100)
        
        for i, fig in enumerate(scene_figs):
            bokeh.plotting.save(fig, os.path.join(save_path, f"test_scene_{i}.html"))
            
        print("\noffline evaluation results saved at "
              "./exp/mleirl/{}/eval_offline".format(
            arglist.exp_name
        ))
    return 

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)