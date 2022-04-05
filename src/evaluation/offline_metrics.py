import numpy as np

def mean_absolute_error(true, pred, mask=None, speed=None, cumulative=False):
    """
    Args:
        true (np.array): true control [T, batch_size, ctl_dim]
        pred (np.array): true control [T, batch_size, ctl_dim]
        mask (np.array, optional): sequence mask [T, batch_size]. Defaults to None.
        speed (np.array, optional): speed vecotr [T, batch_size, speed_dim]. 
            Defaults to None.
        cumulative (bool, optional): whether to accumulate along trajectory. 
            Defaults to False.

    Returns:
        mae (np.array): mean absolute error [ctl_dim]
    """
    mae = np.abs(pred - true)
    
    if mask is not None:
        nan_mask = np.expand_dims(mask, axis=-1).copy()
        nan_mask[nan_mask == 0] = float("nan")
        mae *= nan_mask
    
    if speed is not None:
        mae *= np.abs(speed)
        
    if cumulative:
        mae = np.nansum(mae, 0)
    
    mae = np.nanmean(mae, axis=tuple(range(mae.ndim - 1)))
    return mae

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
