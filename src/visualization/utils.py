import pandas as pd
import matplotlib.pyplot as plt

def plot_history(df_history, plot_keys):
    """ Plot learning history
    
    Args:
        df_history (pd.dataframe): learning history dataframe with a binary train column.
        plot_keys (list): list of colnames to be plotted.

    Returns:
        fig (plt.figure)
        ax (plt.axes)
    """
    df_train = df_history.loc[df_history["train"] == 1]
    df_test = df_history.loc[df_history["train"] == 0]
    
    num_cols = len(plot_keys)
    width = min(4 * num_cols, 15)
    fig, ax = plt.subplots(1, num_cols, figsize=(width, 4))
    for i in range(num_cols):
        ax[i].plot(df_train["epoch"], df_train[plot_keys[i]], label="train")
        ax[i].plot(df_test["epoch"], df_test[plot_keys[i]], label="test")
        ax[i].legend()
        ax[i].set_xlabel("epoch")
        ax[i].set_ylabel(plot_keys[i])
        ax[i].grid()
    
    plt.tight_layout()
    return fig, ax