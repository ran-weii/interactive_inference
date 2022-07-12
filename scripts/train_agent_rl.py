import argparse
import os
import json
import time
import datetime
import pprint
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
from src.simulation.controllers import AgentWrapper

# model imports
from src.agents.mlp_agents import MLPAgent

# training imports
from src.algo.rl import SAC
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
    parser.add_argument("--state_dim", type=int, default=30, help="agent state dimension, default=30")
    parser.add_argument("--act_dim", type=int, default=45, help="agent action dimension, default=45")
    parser.add_argument("--horizon", type=int, default=30, help="agent planning horizon, default=30")
    parser.add_argument("--obs_cov", type=str, default="full", help="agent observation covariance, default=full")
    parser.add_argument("--ctl_cov", type=str, default="full", help="agent control covariance, default=full")
    parser.add_argument("--hmm_rank", type=int, default=0, help="agent hmm rank, 0 for full rank, default=0")
    parser.add_argument("--state_embed_dim", type=int, default=30, help="agent hmm state embedding dimension, default=30")
    parser.add_argument("--act_embed_dim", type=int, default=30, help="agent hmm action embedding dimension, default=30")
    parser.add_argument("--dynamics_model", type=str, choices=["cghmm", "ecghmm"], help="agent dynamics model, default=cghmm")
    parser.add_argument("--agent", type=str, choices=["vin", "mlp"], default="vin", help="agent type, default=vin")
    parser.add_argument("--action_set", type=str, choices=["ego", "frenet"], default="frenet", help="agent action set, default=frenet")
    parser.add_argument("--dynamics_path", type=str, default="none", help="pretrained dynamics path, default=none")
    parser.add_argument("--train_dynamics", type=bool_, default=True, help="whether to train dynamics, default=True")
    # trainer model args
    parser.add_argument("--hidden_dim", type=int, default=64, help="trainer network hidden dims, default=32")
    parser.add_argument("--gru_layers", type=int, default=2, help="trainer gru layers, default=2")
    parser.add_argument("--mlp_layers", type=int, default=2, help="trainer mlp layers, default=2")
    parser.add_argument("--gamma", type=float, default=0.9, help="trainer discount factor, default=0.9")
    parser.add_argument("--beta", type=float, default=0.2, help="softmax temperature, default=0.2")
    # training args
    parser.add_argument("--algo", type=str, choices=["airl"], default="airl", help="training algorithm, default=airl")
    parser.add_argument("--max_data_eps", type=int, default=5, help="max number of data episodes, default=10")
    parser.add_argument("--create_svt", type=bool_, default=True, help="create svt to speed up rollout, default=True")
    parser.add_argument("--min_eps_len", type=int, default=50, help="min track length, default=50")
    parser.add_argument("--max_eps_len", type=int, default=500, help="max track length, default=200")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="ratio of training dataset, default=0.7")
    parser.add_argument("--batch_size", type=int, default=64, help="training batch size, default=64")
    parser.add_argument("--num_eps", type=int, default=3, help="number of episodes per epochs, default=10")
    parser.add_argument("--epochs", type=int, default=3, help="number of training epochs, default=10")
    parser.add_argument("--obs_penalty", type=float, default=0., help="observation penalty, default=0.")
    parser.add_argument("--buffer_size", type=int, default=1e5, help="agent replay buffer size, default=1e5")
    parser.add_argument("--rnn_steps", type=int, default=10, help="rnn steps to store, default=10")
    parser.add_argument("--d_steps", type=int, default=10, help="discriminator steps, default=10")
    parser.add_argument("--ac_steps", type=int, default=10, help="actor critic steps, default=10")
    parser.add_argument("--lr", type=float, default=0.01, help="agent learning rate, default=0.01")
    parser.add_argument("--lr_q", type=float, default=0.005, help="value function learning rate, default=0.005")
    parser.add_argument("--lr_d", type=float, default=0.005, help="discriminator learning rate, default=0.005")
    parser.add_argument("--decay", type=float, default=1e-5, help="weight decay, default=0")
    parser.add_argument("--grad_clip", type=float, default=None, help="gradient clipping, default=None")
    parser.add_argument("--polyak", type=float, default=0.995, help="polyak averaging factor, default=0.995")
    parser.add_argument("--plot_history", type=bool_, default=True, help="plot learning curve, default=True")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

class Logger():
    def __init__(self):
        self.epoch_dict = dict()
        self.history = []
        self.test_episodes = []
    
    def push(self, stats_dict, with_min_and_max=False, with_std=False):
        for key, val in stats_dict.items():
            if not (key in self.epoch_dict.keys()):
                self.epoch_dict[key] = []
            self.epoch_dict[key].append(val)

    def log(self, with_min_and_max=False, with_std=False):
        stats = dict()
        for key, val in self.epoch_dict.items():
            if isinstance(val[0], np.ndarray) or len(val) > 1:
                vals = np.stack(val)
                stats[key + "Avg"] = np.mean(vals)
                stats[key + "Std"] = np.std(vals)
                stats[key + "Min"] = np.min(vals)
                stats[key + "Max"] = np.max(vals)
            else:
                stats[key] = val[-1]

        pprint.pprint(stats)
        self.history.append(stats)

        # erase epoch stats
        self.epoch_dict = dict()

def train(
    env, model, epochs, steps_per_epoch, max_eps_len, update_after, update_every
    ):
    logger = Logger()

    total_steps = epochs * steps_per_epoch
    start_time = time.time()
    
    model.agent.reset()
    eps_id = np.random.choice(np.arange(len(env.dataset)))
    obs, eps_return, eps_len = env.reset(eps_id), 0, 0
    for t in range(total_steps):
        ctl = model.choose_action(obs)
        next_obs, reward, done, info = env.step(ctl)
        eps_return += reward
        eps_len += 1

        # env done handeling
        done = True if eps_len >= max_eps_len else done
        done = True if info["terminated"] == True else done
        
        model.replay_buffer(obs, ctl, reward, done)
        obs = next_obs

        # end of trajectory handeling
        if done or eps_len >= max_eps_len:
            model.replay_buffer.push()
            logger.push({"eps_return": eps_return/eps_len})
            logger.push({"eps_len": eps_len})
            
            model.agent.reset()
            eps_id = np.random.choice(np.arange(len(env.dataset)))
            obs, eps_return, eps_len = env.reset(eps_id), 0, 0

        # train model
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                train_stats = model.take_gradient_step()
                logger.push(train_stats)

        # end of epoch handeling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            logger.push({"epoch": epoch})
            logger.push({"time": time.time() - start_time})
            logger.log()
            print()

    return model, logger

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
        obs_dim, ctl_dim, arglist.hidden_dim, arglist.mlp_layers, 
        activation="relu", use_tanh=False, ctl_limits=ctl_lim
    )

    # init trainer
    gamma = 0.9
    beta = 0.2
    buffer_size = 1000000
    batch_size = 200
    lr = 1e-3
    decay = 1e-5
    polyak = 0.995
    grad_clip = None

    model = SAC(
        agent, arglist.hidden_dim, arglist.mlp_layers, gamma, beta,
        buffer_size, batch_size, lr, decay, polyak, grad_clip
    )
    print(f"num parameters: {count_parameters(model)}")
    print(model)
    print(model.gamma, model.beta)

    # load map
    map_data = MapReader()
    map_data.parse(os.path.join(arglist.data_path, "maps", arglist.scenario + ".osm"))
    
    # init simulator
    observer = Observer(
        map_data, ego_features=ego_features, relative_features=relative_features,
        action_set=action_set
    )
    env = InteractionSimulator(ego_dataset, map_data, observer, max_eps_steps=arglist.max_eps_len)
    
    # torch.autograd.set_detect_anomaly(True)
    epochs = 50
    steps_per_epoch = 1000
    max_eps_len = 500
    update_after = 3000
    update_every = 50
    model, logger = train(
        env, model, 
        epochs, steps_per_epoch, max_eps_len, update_after, update_every
    )

    # save results
    # if arglist.save:
    #     date_time = datetime.datetime.now().strftime("%m-%d-%Y %H-%M-%S")
    #     exp_path = os.path.join(arglist.exp_path, "agents")
    #     agent_path = os.path.join(exp_path, arglist.agent)
    #     save_path = os.path.join(agent_path, date_time)
    #     if not os.path.exists(exp_path):
    #         os.mkdir(exp_path)
    #     if not os.path.exists(agent_path):
    #         os.mkdir(agent_path)
    #     if not os.path.exists(save_path):
    #         os.mkdir(save_path)
        
    #     # save args
    #     with open(os.path.join(save_path, "args.json"), "w") as f:
    #         json.dump(vars(arglist), f)
        
    #     # save model
    #     torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
        
    #     # save history
    #     df_history.to_csv(os.path.join(save_path, "history.csv"), index=False)
        
    #     # save history plot
    #     fig_history, _ = plot_history(df_history, model.loss_keys)
    #     fig_history.savefig(os.path.join(save_path, "history.png"), dpi=100)
        
    #     print(f"\nmodel saved at: {save_path}")

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)