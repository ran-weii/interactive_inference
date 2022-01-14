import numpy as np
import pandas as pd

def derive_acc(df, dt):
    """ differentiate vel and add to df """
    df_ = df.copy()
    f_grad = lambda x: pd.DataFrame(np.gradient(x, dt), index=x.index)
    df_["vx_grad"] = df_.groupby("track_id")["vx"].apply(f_grad)
    df_["vy_grad"] = df_.groupby("track_id")["vy"].apply(f_grad)
    return df_

def normalize_pos(df):
    """ subtrack all pos by initial pos """
    df_ = df.copy()
    
    f_norm = lambda x: x - x.iloc[0]
    df_["x"] = df_.groupby("track_id")["x"].apply(f_norm)
    df_["y"] = df_.groupby("track_id")["y"].apply(f_norm)
    return df_