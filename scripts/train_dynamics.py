import argparse
import os
import json
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 

from src.simulation.observers import FEATURE_SET
from src.data.train_utils import load_data, train_test_split, count_parameters
from src.data.ego_dataset import RelativeDataset, aug_flip_lr, collate_fn
from src.distributions.hmm import ContinuousGaussianHMM
from src.visualization.utils import plot_history

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
        "--save_path", type=str, default="../exp/dynamics"
    )
    parser.add_argument("--scenario", type=str, default="DR_CHN_Merging_ZS")
    parser.add_argument("--filename", type=str, default="vehicle_tracks_007.csv")
    # agent args
    parser.add_argument("--state_dim", type=int, default=30, help="agent state dimension, default=30")
    parser.add_argument("--act_dim", type=int, default=45, help="agent action dimension, default=45")
    parser.add_argument("--obs_cov", type=str, default="full", help="agent observation covariance, default=full")
    parser.add_argument("--ctl_cov", type=str, default="full", help="agent control covariance, default=full")
    parser.add_argument("--hmm_rank", type=int, default=0, help="agent hmm rank, 0 for full rank, default=0")
    parser.add_argument("--model", type=str, choices=["cghmm"], default="cghmm", help="dynamics model, default=cghmm")
    # training args
    parser.add_argument("--min_eps_len", type=int, default=50, help="min track length, default=50")
    parser.add_argument("--max_eps_len", type=int, default=200, help="max track length, default=200")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="ratio of training dataset, default=0.7")
    parser.add_argument("--batch_size", type=int, default=64, help="training batch size, default=64")
    parser.add_argument("--epochs", type=int, default=3, help="number of training epochs, default=10")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate, default=0.01")
    parser.add_argument("--decay", type=float, default=1e-5, help="weight decay, default=0")
    parser.add_argument("--grad_clip", type=float, default=None, help="gradient clipping, default=None")
    parser.add_argument("--plot_history", type=bool_, default=True, help="plot learning curve, default=True")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def run_epoch(model, loader, optimizer, train=True):
    if train:
        model.train()
    else:
        model.eval()
    
    stats = []
    for i, batch in enumerate(loader):
        obs, ctl, mask = batch
        logp_o, logp_u = model.predict(obs, ctl, prior=False, inference=False)

        loss_o = -torch.sum(logp_o * mask, dim=0) / mask.sum(0)
        loss_u = -torch.sum(logp_u * mask[:-1], dim=0) / mask[:-1].sum(0)
        loss = torch.mean(loss_o + loss_u)

        if train:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        stats.append({
            "loss": loss.data.cpu().item(),
            "loss_o": loss_o.data.mean().cpu().item(),
            "loss_u": loss_u.data.mean().cpu().item(),
            "train": train
        })
        
    df_stats = pd.DataFrame(stats)
    return df_stats.mean().to_dict()

def main(arglist):
    torch.manual_seed(arglist.seed)
    
    df_track = load_data(arglist.data_path, arglist.scenario, arglist.filename)
    df_track = df_track.loc[df_track["is_train"] == 1]
    
    feature_set = [
        "d", "ds", "dd", "kappa_r", "psi_error_r", 
        "s_rel", "d_rel", "ds_rel", "dd_rel", "loom_s"
    ]
    assert set(feature_set).issubset(set(FEATURE_SET["ego"] + FEATURE_SET["relative"]))
    dataset = RelativeDataset(
        df_track, feature_set, train_labels_col="is_train",
        min_eps_len=arglist.min_eps_len, max_eps_len=arglist.max_eps_len,
        augmentation=[aug_flip_lr]
    )
    train_loader, test_loader = train_test_split(
        dataset, arglist.train_ratio, arglist.batch_size, 
        collate_fn=collate_fn, seed=arglist.seed
    )
    obs_dim, ctl_dim = len(feature_set), 2 

    print(f"feature set: {feature_set}")
    print(f"train size: {len(train_loader.dataset)}, test size: {len(test_loader.dataset)}")
    
    if arglist.model == "cghmm":
        model = ContinuousGaussianHMM(
            arglist.state_dim, arglist.act_dim, obs_dim, ctl_dim, 
            rank=arglist.hmm_rank, obs_cov=arglist.obs_cov, ctl_cov=arglist.ctl_cov
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=arglist.lr, weight_decay=arglist.decay)
    
    print(f"num parameters: {count_parameters(model)}")
    print(model)

    history = []
    start_time = time.time()
    for e in range(arglist.epochs):
        train_stats = run_epoch(model, train_loader, optimizer, train=True)
        test_stats = run_epoch(model, test_loader, optimizer, train=False)
        
        tnow = time.time() - start_time
        train_stats.update({"epoch": e, "time": tnow})
        test_stats.update({"epoch": e, "time": tnow})
        history.append(train_stats)
        history.append(test_stats)

        if (e + 1) % 1 == 0:
            print("e: {}, train_loss: {:.4f}, test_loss: {:.4f}, t: {:.2f}".format(
                e + 1, train_stats["loss"], test_stats["loss"], tnow
            ))

    df_history = pd.DataFrame(history)

    # save results
    if arglist.save:
        date_time = datetime.datetime.now().strftime("%m-%d-%Y %H-%M-%S")
        exp_path = os.path.join(arglist.save_path, arglist.model)
        save_path = os.path.join(exp_path, date_time)
        if not os.path.exists(arglist.save_path):
            os.mkdir(arglist.save_path)
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        # save args
        with open(os.path.join(save_path, "args.json"), "w") as f:
            json.dump(vars(arglist), f)
        
        # save model
        torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
        
        # save history
        df_history.to_csv(os.path.join(save_path, "history.csv"), index=False)
        
        # save history plot
        fig_history, _ = plot_history(df_history, ["loss_o", "loss_u"])
        fig_history.savefig(os.path.join(save_path, "history.png"), dpi=100)
        
        print(f"\nmodel saved at: {save_path}")
        

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)