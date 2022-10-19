import argparse
import os
import json
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 

# setup imports
from src.data.train_utils import load_data, count_parameters
from src.data.data_filter import filter_segment_by_length
from src.map_api.lanelet import MapReader
from src.data.ego_dataset import RelativeDataset
from src.simulation.sensors import EgoSensor, LeadVehicleSensor, LidarSensor
from src.simulation.simulator import InteractionSimulator
from src.simulation.observers import Observer, CarfollowObserver
from src.simulation.utils import create_svt_from_df

# model imports
from src.agents.vin_agent import VINAgent
from src.agents.hyper_vin_agent import HyperVINAgent
from src.agents.mlp_agents import MLPAgent

# training imports
from src.algo.airl import DAC
from src.algo.recurrent_airl import RecurrentDAC
from src.algo.rl_utils import train
from src.visualization.utils import plot_history
from src.visualization.animation import animate, save_animation

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    str_list_ = lambda x: x.split(",")
    
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
    parser.add_argument("--agent", type=str, choices=["vin", "hvin", "mlp"], default="vin", help="agent type, default=vin")
    parser.add_argument("--state_dim", type=int, default=30, help="vin agent hidden state dim, default=30")
    parser.add_argument("--act_dim", type=int, default=60, help="vin agent action dim, default=60")
    parser.add_argument("--hmm_rank", type=int, default=32, help="vin agent hmm rank, default=32")
    parser.add_argument("--horizon", type=int, default=30, help="vin agent planning horizon, default=30")
    parser.add_argument("--obs_cov", type=str, default="full", help="vin agent observation covariance, default=full")
    parser.add_argument("--ctl_cov", type=str, default="full", help="vin agent control covariance, default=full")
    parser.add_argument("--use_tanh", type=bool_, default=False, help="whether to use tanh transformation, default=False")
    parser.add_argument("--alpha", type=float, default=1., help="agent entropy reward coefficient, default=1.")
    parser.add_argument("--epsilon", type=float, default=1., help="agent policy prior coefficient, default=1.")
    parser.add_argument("--rwd", type=str, choices=["efe", "ece"], default="efe", help="agent reward function. default=efe")
    parser.add_argument("--feature_set", type=str_list_, default=["ego_ds", "lv_s_rel", "lv_ds_rel"], help="agent feature set")
    parser.add_argument("--action_set", type=str_list_, default=["dds"], help="agent action set, default=dds")
    parser.add_argument("--discretize_ctl", type=bool_, default=True, help="whether to discretize ctl using gmm, default=True")
    parser.add_argument("--hyper_dim", type=int, default=4, help="number of hyper factor, default=4")
    # trainer model args
    parser.add_argument("--algo", type=str, choices=["dac", "rdac"], default="dac", help="training algorithm, default=dac")
    parser.add_argument("--hidden_dim", type=int, default=64, help="neural network hidden dims, default=64")
    parser.add_argument("--num_hidden", type=int, default=2, help="number of hidden layers, default=2")
    parser.add_argument("--gru_layers", type=int, default=1, help="number of gru layers, defualt=1")
    parser.add_argument("--activation", type=str, default="relu", help="neural network activation, default=relu")
    parser.add_argument("--gamma", type=float, default=0.9, help="trainer discount factor, default=0.9")
    parser.add_argument("--beta", type=float, default=0.2, help="softmax temperature, default=0.2")
    parser.add_argument("--polyak", type=float, default=0.995, help="polyak averaging factor, default=0.995")
    parser.add_argument("--norm_obs", type=bool_, default=False, help="whether to normalize observations for agent and algo, default=False")
    parser.add_argument("--use_state", type=bool_, default=False, help="whether to use state in discriminator and critic, default=False")
    # data args
    parser.add_argument("--max_data_eps", type=int, default=5, help="max number of data episodes, default=10")
    parser.add_argument("--create_svt", type=bool_, default=True, help="create svt to speed up rollout, default=True")
    parser.add_argument("--min_eps_len", type=int, default=50, help="min track length, default=50")
    parser.add_argument("--max_eps_len", type=int, default=500, help="max track length, default=200")
    # training args
    parser.add_argument("--buffer_size", type=int, default=1e5, help="agent replay buffer size, default=1e5")
    parser.add_argument("--d_batch_size", type=int, default=100, help="discriminator batch size, default=100")
    parser.add_argument("--a_batch_size", type=int, default=32, help="actor critic batch size, default=32")
    parser.add_argument("--rnn_len", type=int, default=10, help="number of recurrent steps to sample, default=10")
    parser.add_argument("--d_steps", type=int, default=30, help="discriminator steps, default=30")
    parser.add_argument("--a_steps", type=int, default=30, help="actor critic steps, default=30")
    parser.add_argument("--lr_d", type=float, default=0.001, help="discriminator learning rate, default=0.001")
    parser.add_argument("--lr_a", type=float, default=0.001, help="actor learning rate, default=0.001")
    parser.add_argument("--lr_c", type=float, default=0.001, help="critic learning rate, default=0.001")
    parser.add_argument("--decay", type=float, default=1e-5, help="weight decay, default=0")
    parser.add_argument("--grad_clip", type=float, default=1000., help="gradient clipping, default=1000.")
    parser.add_argument("--grad_penalty", type=float, default=1., help="discriminator gradient norm penalty, default=1.")
    parser.add_argument("--grad_target", type=float, default=1., help="discriminator gradient norm target, default=0.")
    parser.add_argument("--bc_penalty", type=float, default=1., help="behavior cloning penalty, default=1.")
    parser.add_argument("--obs_penalty", type=float, default=1., help="observation penalty, default=1.")
    parser.add_argument("--reg_penalty", type=float, default=1., help="agent regularization penalty, default=1.")
    # rollout args
    parser.add_argument("--epochs", type=int, default=10, help="number of training epochs, default=10")
    parser.add_argument("--max_steps", type=int, default=400, help="max rollout steps, default=400")
    parser.add_argument("--steps_per_epoch", type=int, default=1000, help="number of env steps per epoch, default=1000")
    parser.add_argument("--update_after", type=int, default=1000, help="burn-in env steps, default=1000")
    parser.add_argument("--update_every", type=int, default=50, help="update every env steps, default=50")
    parser.add_argument("--log_test_every", type=int, default=10, help="steps between logging test episodes, default=10")
    parser.add_argument("--verbose", type=bool_, default=False, help="whether to verbose during training, default=False")
    parser.add_argument("--cp_every", type=int, default=1000, help="checkpoint interval, default=1000")
    parser.add_argument("--save_buffer", type=bool_, default=False, help="whether to save buffer, default=False")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

class SaveCallback:
    def __init__(self, arglist, map_data, cp_history=None):
        date_time = datetime.datetime.now().strftime("%m-%d-%Y %H-%M-%S")
        exp_path = os.path.join(arglist.exp_path, "agents")
        agent_path = os.path.join(exp_path, arglist.agent)
        save_path = os.path.join(agent_path, date_time)
        test_eps_path = os.path.join(save_path, "test_episodes")
        model_path = os.path.join(save_path, "models") # used to save model checkpoint
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        if not os.path.exists(agent_path):
            os.mkdir(agent_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(test_eps_path):
            os.mkdir(test_eps_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        
        # save args
        with open(os.path.join(save_path, "args.json"), "w") as f:
            json.dump(vars(arglist), f)

        self.save_path = save_path
        self.test_eps_path = test_eps_path
        self.model_path = model_path
        self.cp_history = cp_history
        self.cp_every = arglist.cp_every
        self.map_data = map_data

        self.num_test_eps = 0
        self.iter = 0

    def __call__(self, model, logger):        
        self.iter += 1

        # save history
        df_history = pd.DataFrame(logger.history)
        df_history = df_history.assign(train=1)
        if self.cp_history is not None:
            df_history["epoch"] += self.cp_history["epoch"].values[-1] + 1
            df_history["time"] += self.cp_history["time"].values[-1]
            df_history = pd.concat([self.cp_history, df_history], axis=0)
        df_history.to_csv(os.path.join(self.save_path, "history.csv"), index=False)
        
        # save history plot
        fig_history, _ = plot_history(df_history, model.plot_keys, plot_std=True)
        fig_history.savefig(os.path.join(self.save_path, "history.png"), dpi=100)
        
        # save test episode
        if len(logger.test_episodes) > self.num_test_eps:
            ani = animate(self.map_data, logger.test_episodes[-1], plot_lidar=False)
            save_animation(ani, os.path.join(self.test_eps_path, f"epoch_{self.iter}_ani.mp4"))
            self.num_test_eps += 1
        
        plt.clf()
        plt.close()
        
        if (self.iter + 1) % self.cp_every == 0:
            # save model
            cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state_dict, os.path.join(self.model_path, f"model_{self.iter}.pt"))
            print(f"\ncheckpoint saved at: {self.save_path}\n")
    
    def save_model(self, model):
        cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state_dict, os.path.join(self.save_path, "model.pt"))

    def save_buffer(self, model):
        with open(os.path.join(self.save_path, "buffer.p"), "wb") as f:
            pickle.dump(model.replay_buffer, f)
        
def main(arglist):
    np.random.seed(arglist.seed)
    torch.manual_seed(arglist.seed)
    
    df_track = load_data(arglist.data_path, arglist.scenario, arglist.filename, train=True, load_raw=True)
    df_track.loc[df_track["is_train"] != 1]["is_train"] = np.nan
    
    # filter episode length
    df_track["eps_id"], df_track["eps_len"] = filter_segment_by_length(
        df_track["eps_id"].values, arglist.min_eps_len
    )

    # build svt
    svt = create_svt_from_df(df_track, eps_id_col="eps_id")

    # load map
    map_data = MapReader()
    map_data.parse(os.path.join(arglist.data_path, "maps", arglist.scenario + ".osm"))
    
    # init sensors
    ego_sensor = EgoSensor(map_data)
    lv_sensor = LeadVehicleSensor(map_data, track_lv=True)
    lidar_sensor = LidarSensor()
    sensors = [ego_sensor, lv_sensor, lidar_sensor]

    feature_set = arglist.feature_set
    action_set = arglist.action_set

    if arglist.action_set == ["dds"]:
        observer = CarfollowObserver(map_data, sensors, feature_set=feature_set)
    else:
        observer = Observer(map_data, sensors, feature_set=feature_set, action_set=action_set)

    env = InteractionSimulator(map_data, sensors, observer, svt)
    
    # compute obs and ctl mean and variance stats
    obs_mean = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][feature_set].mean().values).to(torch.float32)
    obs_var = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][feature_set].var().values).to(torch.float32)
    ctl_mean = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].mean().values).to(torch.float32)
    ctl_var = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].var().values).to(torch.float32)

    # compute ctl limits
    ctl_max = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].max().values).to(torch.float32)
    ctl_min = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].min().values).to(torch.float32)
    ctl_lim = torch.max(torch.abs(ctl_max), torch.abs(ctl_min)) * 1.2
    
    obs_dim = len(feature_set)
    ctl_dim = len(action_set)
    print(f"obs_dim: {obs_dim}, ctl_dim: {ctl_dim}")
    
    rel_dataset = RelativeDataset(
        df_track, feature_set, action_set, train_labels_col="is_train", 
        max_eps=arglist.max_data_eps, seed=arglist.seed
    )
    
    if arglist.agent == "mlp":
        agent = MLPAgent(
            obs_dim, ctl_dim, arglist.hidden_dim, arglist.num_hidden, 
            activation=arglist.activation, use_tanh=arglist.use_tanh, 
            ctl_limits=ctl_lim, norm_obs=arglist.norm_obs
        )
    elif arglist.agent == "vin":
        agent = VINAgent(
            arglist.state_dim, arglist.act_dim, obs_dim, ctl_dim, arglist.hmm_rank,
            arglist.horizon, alpha=arglist.alpha, beta=arglist.epsilon, 
            obs_cov=arglist.obs_cov, ctl_cov=arglist.ctl_cov, rwd=arglist.rwd, 
            use_tanh=arglist.use_tanh, ctl_lim=ctl_lim, detach=False
        )
        agent.obs_model.init_batch_norm(obs_mean, obs_var)
        if not arglist.use_tanh:
            agent.ctl_model.init_batch_norm(ctl_mean, ctl_var)
        
        if arglist.action_set == ["dds"] and arglist.discretize_ctl:
            # load ctl gmm parameters
            with open(os.path.join(arglist.exp_path, "agents", "ctl_model", "model.p"), "rb") as f:
                [ctl_means, ctl_covs, weights] = pickle.load(f)

            agent.ctl_model.init_params(ctl_means, ctl_covs)
            print("action model loaded")

    elif arglist.agent == "hvin":
        agent = HyperVINAgent(
            arglist.state_dim, arglist.act_dim, obs_dim, ctl_dim, arglist.hmm_rank,
            arglist.horizon, arglist.hyper_dim, arglist.hidden_dim, arglist.num_hidden, 
            arglist.gru_layers, arglist.activation,
            obs_cov=arglist.obs_cov, ctl_cov=arglist.ctl_cov, 
            use_tanh=arglist.use_tanh, ctl_lim=ctl_lim
        )
        agent.obs_model.init_batch_norm(obs_mean, obs_var)
        if not arglist.use_tanh:
            agent.ctl_model.init_batch_norm(ctl_mean, ctl_var)

    # init trainer
    if arglist.algo == "dac":
        model = DAC(
            agent, arglist.hidden_dim, arglist.num_hidden, 
            gamma=arglist.gamma, beta=arglist.beta, polyak=arglist.polyak, norm_obs=arglist.norm_obs,
            buffer_size=arglist.buffer_size, batch_size=arglist.d_batch_size, 
            d_steps=arglist.d_steps, a_steps=arglist.a_steps, 
            lr=arglist.lr_d, decay=arglist.decay, grad_clip=arglist.grad_clip, 
            grad_penalty=arglist.grad_penalty, grad_target=arglist.grad_target
        )
        model.fill_real_buffer(rel_dataset)
    elif arglist.algo == "rdac":
        assert "vin" in arglist.agent
        model = RecurrentDAC(
            agent, arglist.hidden_dim, arglist.num_hidden, arglist.activation,
            gamma=arglist.gamma, beta=arglist.beta, polyak=arglist.polyak, 
            norm_obs=arglist.norm_obs, use_state=arglist.use_state,
            buffer_size=arglist.buffer_size, d_batch_size=arglist.d_batch_size, a_batch_size=arglist.a_batch_size,
            rnn_len=arglist.rnn_len, d_steps=arglist.d_steps, a_steps=arglist.a_steps, 
            lr_d=arglist.lr_d, lr_a=arglist.lr_a, lr_c=arglist.lr_c, decay=arglist.decay, grad_clip=arglist.grad_clip, 
            grad_penalty=arglist.grad_penalty, bc_penalty=arglist.bc_penalty, 
            obs_penalty=arglist.obs_penalty, reg_penalty=arglist.reg_penalty
        )
        model.fill_real_buffer(rel_dataset)
    
    # load model from checkpoint
    cp_history = None
    if arglist.checkpoint_path != "none":
        cp_path = os.path.join(
            arglist.exp_path, "agents", 
            arglist.agent, arglist.checkpoint_path
        )
        # load state dict
        state_dict = torch.load(os.path.join(cp_path, "model.pt"))
        
        # load history
        cp_history = pd.read_csv(os.path.join(cp_path, "history.csv"))
        
        model.load_state_dict(state_dict, strict=True)

        print(f"loaded checkpoint from {cp_path}")
    
    print(f"num parameters: {count_parameters(model)}")
    print(model)
    
    callback = None
    if arglist.save:
        callback = SaveCallback(arglist, map_data, cp_history=cp_history)

    model, logger = train(
        env, model, arglist.epochs, arglist.max_steps, arglist.steps_per_epoch, 
        arglist.update_after, arglist.update_every, 
        log_test_every=arglist.log_test_every, verbose=arglist.verbose,
        callback=callback
    )
    if callback is not None:
        callback.save_model(model)
        if arglist.save_buffer:
            callback.save_buffer(model)

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)