import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.kalman_filter import BatchKalmanFilter

# set pandas display option
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../interaction-dataset-master"
    )
    parser.add_argument("--scenario", type=str, default="DR_CHN_Merging_ZS")
    parser.add_argument("--num_files", type=int, default=3, help="number of files used to sample trajectories, default=3")
    parser.add_argument("--min_len", type=int, default=100, help="min track length, default=100")
    parser.add_argument("--num_train", type=int, default=50, help="number of tracks used to train kalman filter, default=50")
    parser.add_argument("--num_test", type=int, default=10, help="number of tracks used to test kalman filter, default=10")
    parser.add_argument("--epochs", type=int, default=10, help="number of em epochs, default=25")
    parser.add_argument("--plot_history", type=bool_, default=True, help="plot em learning curve, default=True")
    parser.add_argument("--seed", type=int, default=0)
    arglist = parser.parse_args()
    return arglist

def derive_acc(df, dt):
    """ differentiate vel """
    f_grad = lambda x: pd.DataFrame(np.gradient(x, dt), index=x.index)
    df["vx_grad"] = df.groupby("track_id")["vx"].apply(f_grad)
    df["vy_grad"] = df.groupby("track_id")["vy"].apply(f_grad)
    return df

def normalize_pos(df):
    """ subtrack all pos by initial pos """
    f_norm = lambda x: x - x.iloc[0]
    df["x"] = df.groupby("track_id")["x"].apply(f_norm)
    df["y"] = df.groupby("track_id")["y"].apply(f_norm)
    return df

def prepare_data(df_all_tracks, dt, num_train, num_test, max_len):
    # random sample train and test sequences
    train_id = np.random.choice(
        len(df_all_tracks["track_id"].unique()),
        size=(num_train + num_test,), replace=False
    )
    df_train = df_all_tracks.loc[
        df_all_tracks["track_id"].isin(train_id[:num_train])
    ].reset_index(drop=True)
    df_test = df_all_tracks.loc[
        df_all_tracks["track_id"].isin(train_id[num_train:])
    ].reset_index(drop=True)
    
    df_train = df_train.groupby("track_id").head(max_len)
    df_test = df_test.groupby("track_id").head(max_len)
        
    # derive acceleration
    df_train = derive_acc(df_train, dt)
    df_test = derive_acc(df_test, dt)
    
    # normalize data
    df_train = normalize_pos(df_train)
    df_test = normalize_pos(df_test)
    return df_train, df_test

def plot_kf_params(kf):
    fig, ax = plt.subplots(3, 2, figsize=(10, 6))
    sns.heatmap(kf.transition_matrices, annot=True, cbar=False, ax=ax[0,0])
    sns.heatmap(kf.transition_covariance, annot=True, cbar=False, ax=ax[0,1])
    sns.heatmap(kf.observation_matrices, annot=True, cbar=False, ax=ax[1,0])
    sns.heatmap(kf.observation_covariance, annot=True, cbar=False, ax=ax[1,1])
    sns.heatmap(kf.initial_state_mean.reshape(-1, 1), annot=True, cbar=False, ax=ax[2,0])
    sns.heatmap(kf.initial_state_covariance, annot=True, cbar=False, ax=ax[2,1])
    
    ax[0,0].set_title("transition_matrix")
    ax[0,1].set_title("transition_cov")
    ax[1,0].set_title("observation_matrix")
    ax[1,1].set_title("observation_cov")
    ax[2,0].set_title("init_state_matrix")
    ax[2,1].set_title("init_state_cov")
    
    plt.tight_layout()
    return fig
    
def plot_history(df_history):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(df_history["epoch"], df_history["train_loss"], "-o", label="train_loss")
    ax.plot(df_history["epoch"], df_history["test_loss"], "-o", label="test_loss")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid()
    return fig

def main(arglist):
    np.random.seed(arglist.seed)
    
    track_paths = glob.glob(
        os.path.join(
            arglist.data_path, 
            "recorded_trackfiles", 
            arglist.scenario,
            "*.csv"
        )
    ) 
    track_id = np.random.randint(
        0, len(track_paths), size=(arglist.num_files)
    )
    track_paths = [track_paths[i] for i in track_id]
    
    # load data
    df_all_tracks = []
    num_cars = 0
    for i, track_path in enumerate(track_paths):
        df_track = pd.read_csv(track_path)
        
        # remove short tracks
        df_track = df_track.groupby("track_id").filter(
            lambda x: x.shape[0] >= arglist.min_len
        )
    
        # reset track_id
        df_track["track_id"] = pd.Categorical(df_track["track_id"]).codes
        df_track["track_id"] += num_cars
        
        df_all_tracks.append(df_track)
        num_cars += len(np.unique(df_track["track_id"]))
        
    df_all_tracks = pd.concat(df_all_tracks, axis=0)
    dt = df_all_tracks.groupby("track_id")["timestamp_ms"].diff().mean() / 1000
    
    df_train, df_test = prepare_data(
        df_all_tracks, 
        dt, 
        arglist.num_train, 
        arglist.num_test, 
        arglist.min_len
    )
    
    # check interpolation quality
    # fig, ax = plt.subplots(3, 2, figsize=(6, 6))
    # ax[0, 0].plot(df_train["x"])
    # ax[0, 1].plot(df_train["y"])
    # ax[1, 0].plot(df_train["vx"])
    # ax[1, 1].plot(df_train["vy"])
    # ax[2, 0].plot(df_train["vx_grad"])
    # ax[2, 1].plot(df_train["vy_grad"])
    # plt.tight_layout()
    # plt.show()
    
    # pack observations
    obs_cols = ["x", "y", "vx", "vy", "vx_grad", "vy_grad"]
    f_df_to_batch = lambda df: df.groupby("track_id")[obs_cols].\
        apply(lambda x: x.values.tolist()).tolist()
    obs_train = np.array(f_df_to_batch(df_train))
    obs_test = np.array(f_df_to_batch(df_test))
    
    # init parameters with const acc model
    eps = 1e-3
    transition_matrix = np.array(
        [[1, 0, dt,  0,  0.5 * dt**2,  0],
         [0, 1,  0, dt,  0,  0.5 * dt**2],
         [0, 0,  1,  0, dt,  0],
         [0, 0,  0,  1,  0, dt],
         [0, 0,  0,  0,  1,  0],
         [0, 0,  0,  0,  0,  1]]
    ) 
    observation_matrix = np.eye(6) 
    transition_covariance = 0.01 * np.eye(6) + eps
    observation_covariance = 0.01 * np.eye(6) + eps
    transition_offset = np.zeros((6))
    observation_offset = np.zeros((6))
    initial_state_mean = np.zeros((6,))
    initial_state_covariance = np.diag([1, 0.5, 1, 0.5, 1, 1])
    
    # train
    kf = BatchKalmanFilter(
        transition_matrix,
        observation_matrix,
        transition_covariance,
        observation_covariance,
        transition_offset,
        observation_offset,
        initial_state_mean,
        initial_state_covariance,
        random_state=arglist.seed,
        em_vars=[
        # "transition_matrices", 
        # "observation_matrices",
        "transition_covariance", 
        "observation_covariance",
        # "transition_offsets", 
        # "observation_offsets", 
        # "initial_state_mean", 
        # "initial_state_covariance"
        ]
    )
    
    history = []
    for e in range(arglist.epochs):
        kf = kf.batch_em(X=obs_train, n_iter=1)
        train_loss = kf.batch_loglikelihood(obs_train)
        test_loss = kf.batch_loglikelihood(obs_test)
        history.append(
            {"epoch": e + 1, "train_loss": train_loss, "test_loss": test_loss}   
        )
        print("epoch: {}, train_loss: {}, test_loss: {}".format(
            e + 1,
            np.round(train_loss, 2),
            np.round(test_loss, 2),
        ))
        
    df_history = pd.DataFrame(history)
    
    fig_params = plot_kf_params(kf)
    fig_history = plot_history(df_history)
    
    plt.show()
    return 

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)