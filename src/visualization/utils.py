import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def set_plotting_style(strip_size=12, label_size=12):
    mpl.rcParams["axes.labelsize"] = label_size
    mpl.rcParams["xtick.labelsize"] = strip_size
    mpl.rcParams["ytick.labelsize"] = strip_size
    mpl.rcParams["legend.title_fontsize"] = strip_size
    mpl.rcParams["legend.fontsize"] = strip_size
    mpl.rcParams["axes.titlesize"] = label_size
    mpl.rcParams["figure.titlesize"] = label_size

def plot_history(df_history, plot_keys):
    """ Plot learning history
    
    Args:
        df_history (pd.dataframe): learning history dataframe with a binary train column.
        plot_keys (list): list of colnames to be plotted.

    Returns:
        fig (plt.figure)
        ax (plt.axes)
    """
    train_unique = list(df_history["train"].unique())
    df_train = df_history.loc[df_history["train"] == 1]
    if 0 in train_unique:
        df_test = df_history.loc[df_history["train"] == 0]
    
    num_cols = len(plot_keys)
    width = min(4 * num_cols, 15)
    fig, ax = plt.subplots(1, num_cols, figsize=(width, 4))
    for i in range(num_cols):
        ax[i].plot(df_train["epoch"], df_train[plot_keys[i]], label="train")
        if 0 in train_unique:
            ax[i].plot(df_test["epoch"], df_test[plot_keys[i]], label="test")
        ax[i].legend()
        ax[i].set_xlabel("epoch")
        ax[i].set_ylabel(plot_keys[i])
        ax[i].grid()
    
    plt.tight_layout()
    return fig, ax

def plot_time_series(x, feature_names, x_sample=None, num_cols=5, figsize=(6, 2), title=""):
    """ Plot time series features for a single episode
    
    Args:
        x (torch.tensor): features. size=[T, f_dim]
        feature_names (list): list of feature names.
        x_sample (torch.tensor, optional): samples of x to plot error shade. 
            size=[num_samples, T, f_dim]. Default=None
        num_cols (int, optional): number of columns. Default=5
        figsize (tuple, optional): figure size. Default=(6, 2)
        title (str, optional): title to be set on ax[0]
    
    Returns:
        fig (plt.figure): figure object
        ax (plt.axes): flattened axes object
    """
    assert len(x.shape) == 2
    assert x.shape[-1] == len(feature_names)
    
    f_dim = x.shape[-1]
    num_cols = min(f_dim, num_cols)
    num_rows = np.ceil(f_dim / num_cols).astype(int)
    fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
    if len(ax) == 1:
        ax = [ax]
    else:
        ax = ax.flat
    
    t = np.arange(len(x))
    for i in range(f_dim):
        ax[i].plot(t, x[:, i])
        if x_sample is not None:
            mu = x_sample[:, :, i].mean(0)
            std = x_sample[:, :, i].std(0)
            ax[i].fill_between(t, mu + std, mu - std, alpha=0.4)
        ax[i].set_xlabel("time")
        ax[i].set_title(feature_names[i])
    ax[0].set_title(title)
    plt.tight_layout()
    return fig, ax

def plot_scatter(x, x_label=None, y_label=None, figsize=(6, 6), title=""):
    """ Create a single scatter plot

    Args:
        x (torch.tensor): data tensor. 
            size=[num_samples, f_dim] or [num_samples, num_groups, f_dim]
        x_label (str, optional): x label. Default=None
        y_label (str, optional): y label. Default=None
        figsize (tuple, optional): figure size. Default=(6, 6)
    """
    x = x.transpose(0, -1)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(x[0].T, x[1].T, "o", ms=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax