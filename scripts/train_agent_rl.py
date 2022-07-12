import argparse
import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 

# setup imports
from src.simulation.observers import ACTION_SET, FEATURE_SET
from src.data.train_utils import load_data, count_parameters
from src.data.data_filter import filter_segment_by_length
from src.data.ego_dataset import EgoDataset
from src.map_api.lanelet import MapReader
from src.simulation.simulator import InteractionSimulator
from src.simulation.observers import Observer

# model imports
from src.agents.mlp_agents import MLPAgent

# training imports
from src.algo.rl import SAC
from src.algo.rl_utils import train
from src.visualization.utils import plot_history

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../interaction-dataset-master"
    )
    parser.add_argument("--exp_path", type=str, default="../exp")
    parser.add_argument("--scenario", type=str, default="DR_CHN_Merging_ZS")
    parser.add_argument("--filename", type=str, default="vehicle_tracks_007.csv")
    parser.add_argument("--checkpoint_path", type=str, default="none", 
        help="if entered train agent from check point")
    # agent args
    parser.add_argument("--agent", type=str, choices=["vin", "mlp"], default="vin", help="agent type, default=vin")
    parser.add_argument("--action_set", type=str, choices=["ego", "frenet"], default="frenet", help="agent action set, default=frenet")
    parser.add_argument("--dynamics_path", type=str, default="none", help="pretrained dynamics path, default=none")
    parser.add_argument("--train_dynamics", type=bool_, default=True, help="whether to train dynamics, default=True")
    parser.add_argument("--use_tanh", type=bool_, default=False, help="whether to use tanh transformation, default=False")
    # trainer model args
    parser.add_argument("--algo", type=str, choices=["sac"], default="sac", help="training algorithm, default=sac")
    parser.add_argument("--hidden_dim", type=int, default=64, help="neural network hidden dims, default=64")
    parser.add_argument("--num_hidden", type=int, default=2, help="number of hidden layers, default=2")
    parser.add_argument("--activation", type=str, default="relu", help="neural network activation, default=relu")
    parser.add_argument("--gamma", type=float, default=0.9, help="trainer discount factor, default=0.9")
    parser.add_argument("--beta", type=float, default=0.2, help="softmax temperature, default=0.2")
    parser.add_argument("--polyak", type=float, default=0.995, help="polyak averaging factor, default=0.995")
    # data args
    parser.add_argument("--max_data_eps", type=int, default=5, help="max number of data episodes, default=10")
    parser.add_argument("--create_svt", type=bool_, default=True, help="create svt to speed up rollout, default=True")
    parser.add_argument("--min_eps_len", type=int, default=50, help="min track length, default=50")
    parser.add_argument("--max_eps_len", type=int, default=500, help="max track length, default=200")
    # training args
    parser.add_argument("--batch_size", type=int, default=100, help="training batch size, default=100")
    parser.add_argument("--buffer_size", type=int, default=1e5, help="agent replay buffer size, default=1e5")
    parser.add_argument("--epochs", type=int, default=10, help="number of training epochs, default=10")
    parser.add_argument("--steps_per_epoch", type=int, default=1000, help="number of env steps per epoch, default=1000")
    parser.add_argument("--update_after", type=int, default=3000, help="burn-in env steps, default=3000")
    parser.add_argument("--update_every", type=int, default=50, help="update every env steps, default=50")
    parser.add_argument("--a_steps", type=int, default=10, help="actor critic steps, default=50")
    parser.add_argument("--lr", type=float, default=0.001, help="model learning rate, default=0.001")
    parser.add_argument("--decay", type=float, default=1e-5, help="weight decay, default=0")
    parser.add_argument("--grad_clip", type=float, default=1000., help="gradient clipping, default=1000.")
    parser.add_argument("--plot_history", type=bool_, default=True, help="plot learning curve, default=True")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def main(arglist):
    np.random.seed(arglist.seed)
    torch.manual_seed(arglist.seed)
    
    df_track = load_data(arglist.data_path, arglist.scenario, arglist.filename, train=True)

    # filter episode length
    df_track["eps_id"], df_track["eps_len"] = filter_segment_by_length(
        df_track["eps_id"].values, arglist.min_eps_len
    )
    
    """ TODO: add code to adapt input feature set """
    # define feature set
    ego_features = ["d", "ds", "dd", "kappa_r", "psi_error_r", ]
    relative_features = ["s_rel", "d_rel", "ds_rel", "dd_rel", "loom_s"]
    feature_set = ego_features + relative_features
    assert set(ego_features).issubset(set(FEATURE_SET["ego"]))
    assert set(relative_features).issubset(set(FEATURE_SET["relative"]))

    # define action set
    if arglist.action_set == "frenet":
        action_set = ["dds", "ddd"]
    else:
        action_set = ["ax_ego", "ay_ego"]
    assert set(action_set).issubset(set(ACTION_SET))
    
    # compute ctl limits
    ctl_max = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].max().values).to(torch.float32)
    ctl_min = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].min().values).to(torch.float32)
    ctl_lim = torch.max(torch.abs(ctl_max), torch.abs(ctl_min)) * 1.2

    ego_dataset = EgoDataset(
        df_track, train_labels_col="is_train", 
        max_eps=arglist.max_data_eps, create_svt=True, seed=arglist.seed
    )
    obs_dim, ctl_dim = len(feature_set), 2 

    print(f"feature set: {feature_set}")
    print(f"action set: {action_set}")
    print(f"data size: {len(ego_dataset)}")

    agent = MLPAgent(
        obs_dim, ctl_dim, arglist.hidden_dim, arglist.num_hidden, 
        activation=arglist.activation, use_tanh=arglist.use_tanh, ctl_limits=ctl_lim
    )

    # init trainer
    if arglist.algo == "sac":
        model = SAC(
            agent, arglist.hidden_dim, arglist.num_hidden, 
            gamma=arglist.gamma, beta=arglist.beta, buffer_size=arglist.buffer_size,
            batch_size=arglist.batch_size, a_steps=arglist.a_steps, lr=arglist.lr, 
            decay=arglist.decay, polyak=arglist.polyak, grad_clip=arglist.grad_clip
        )
    print(f"num parameters: {count_parameters(model)}")
    print(model)

    # load map
    map_data = MapReader()
    map_data.parse(os.path.join(arglist.data_path, "maps", arglist.scenario + ".osm"))
    
    # init simulator
    observer = Observer(
        map_data, ego_features=ego_features, relative_features=relative_features,
        action_set=action_set
    )
    env = InteractionSimulator(ego_dataset, map_data, observer, max_eps_steps=arglist.max_eps_len)
    
    model, logger = train(
        env, model, arglist.epochs, arglist.steps_per_epoch, 
        arglist.update_after, arglist.update_every
    )
    
    df_history = pd.DataFrame(logger.history)
    df_history = df_history.assign(train=1)
    print(df_history)
    
    # save results
    if arglist.save:
        date_time = datetime.datetime.now().strftime("%m-%d-%Y %H-%M-%S")
        exp_path = os.path.join(arglist.exp_path, "agents")
        agent_path = os.path.join(exp_path, arglist.agent)
        save_path = os.path.join(agent_path, date_time)
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        if not os.path.exists(agent_path):
            os.mkdir(agent_path)
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
        fig_history, _ =plot_history(df_history, model.plot_keys)
        fig_history.savefig(os.path.join(save_path, "history.png"), dpi=100)
        
        print(f"\nmodel saved at: {save_path}")

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)