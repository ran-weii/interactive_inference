import argparse
import os
import json
import numpy as np
import pandas as pd

import torch 
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from src.data.lanelet import load_lanelet_df
from src.data.ego_dataset import RelativeDataset
from src.irl.algorithms import MLEIRL
from src.evaluation.offline_metrics import (
    mean_absolute_error, threshold_relative_error)
from src.visualization.inspection import (
    get_active_inference_parameters, plot_active_inference_parameters)

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
        "--exp_path", type=str, default="../exp/mleirl"
    )
    parser.add_argument("--scenario", type=str, default="DR_CHN_Merging_ZS")
    parser.add_argument("--filename", type=str, default="vehicle_tracks_007.csv")
    parser.add_argument("--exp_name", type=str, default="03-10-2022 10-52-38")
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def collate_fn(batch):
    pad_obs = pad_sequence([b["ego"] for b in batch])
    pad_act = pad_sequence([b["act"] for b in batch])
    mask = torch.all(pad_obs != 0, dim=-1).to(torch.float32)
    return pad_obs, pad_act, mask

def train_test_split(dataset, train_ratio, batch_size, seed):
    gen = torch.Generator()
    gen.manual_seed(seed)
    
    train_size = np.ceil(train_ratio * len(dataset)).astype(int)
    test_size = len(dataset) - train_size
    
    train_set, test_set = random_split(
        dataset, [train_size, test_size], generator=gen
    )
    
    train_loader = DataLoader(
        train_set, batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False
    )
    return train_loader, test_loader

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
    
    u_true = torch.cat(u_true, dim=1).data.numpy()
    u_pred = torch.cat(u_pred, dim=1).data.numpy()
    obs = torch.cat(obs, dim=1).data.numpy()
    masks = torch.cat(masks, dim=1).data.numpy()
    
    # get speed
    ego_fields = loader.dataset.dataset.ego_fields
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
    return out

def main(arglist):
    exp_path = os.path.join(arglist.exp_path, arglist.exp_name)
    
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
    
    dataset = RelativeDataset(df_track, df_lanelet, config["min_eps_len"], config["max_eps_len"])
    train_loader, test_loader = train_test_split(
        dataset, config["train_ratio"], config["batch_size"], seed=123
    )
    
    # get dimensions from data 
    [obs, ctl, mask] = next(iter(train_loader))
    obs_dim = obs.shape[-1]
    ctl_dim = ctl.shape[-1]
    
    # load model
    model = MLEIRL(
        config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["horizon"],
        obs_dist=config["obs_dist"], obs_cov=config["obs_cov"], 
        ctl_dist=config["ctl_dist"], ctl_cov=config["ctl_cov"]
    )
    model.load_state_dict(state_dict)
    agent = model.agent
    
    metrics_dict = eval_epoch(agent, test_loader)
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
        fig_params.savefig(os.path.join(save_path, "params.png"), dpi=100)
        
        print("\noffline evaluation results saved at "
              "./exp/mleirl/{}/eval_offline".format(
            arglist.exp_name
        ))
    return 

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)