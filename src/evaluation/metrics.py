import numpy as np

def compute_interquartile_mean(metrics, q1=25., q2=75.):
    """ Compute interquantile mean
    
    Args:
        metrics (np.array): evaluation metrics
        q1 (float): percentile lower bound
        q2 (float): percentile upper bound

    Return:
        iqm (float): interquartile mean
    """
    metrics_ = metrics[
        (metrics > np.percentile(metrics, q1)) & \
        (metrics < np.percentile(metrics, q2))
    ]
    iqm = np.mean(metrics_)
    return iqm

def mean_absolute_error(true, pred, dims):
    """ Compute nan masked mean absolute error 
    
    Args:
        true (np.array): true sequence. size=[..., batch_size, out_dim]
        pred (np.array): predicted sequence. size=[..., batch_size, out_dim]
        dims (tuple): dimensions to perform average

    Returns:
        mae (np.array): mean absolute errors. size=[batch_size, out_dim]
    """
    assert true.shape == pred.shape
    return np.nanmean(np.abs(pred - true), axis=dims)

def threshold_relative_error(true, pred, mask=None, alpha=0.1):
    """ Proposed by "On Offline Evaluation of Vision-based Driving Models, 
        Codevilla et al, 2018"
    Args:
        true (np.array): true control [T, batch_size, ctl_dim]
        pred (np.array): true control [T, batch_size, ctl_dim]
        mask (np.array, optional): sequence mask [T, batch_size]. Defaults to None.
        alpha (float, optional): _description_. Defaults to 0.1.

    Returns:
        tre (np.array): threshold relative error [ctl_dim]
    """
    mae = np.abs(pred - true)
    tre = np.heaviside(mae - alpha * np.abs(true), 0.5)
    
    if mask is not None:
        nan_mask = np.expand_dims(mask, axis=-1).copy()
        nan_mask[nan_mask == 0] = float("nan")
        tre *= nan_mask
    
    tre = np.nanmean(tre, axis=tuple(range(tre.ndim - 1)))
    return tre
