import numpy as np

def mean_absolute_error(true, pred, mask=None, speed=None, cumulative=False):
    """
    Args:
        true (np.array): true control [T, batch_size, ctl_dim]
        pred (np.array): true control [T, batch_size, ctl_dim]
        mask (np.array): sequence mask [T, batch_size]
        speed (np.array, optional): speed vecotr [T, batch_size]. Defaults to None.
        cumulative (bool, optional): whether to accumulate along trajectory. 
            Defaults to False.

    Returns:
        mae (float): mean absolute error
    """
    mae = np.abs(pred - true)
    
    if mask is not None:
        nan_mask = np.expand_dims(mask, axis=-1).copy()
        nan_mask[nan_mask == 0] = float("nan")
        mae *= nan_mask
    
    if speed is not None:
        mae *= np.expand_dims(np.abs(speed), axis=-1)
        
    if cumulative:
        mae = np.nansum(mae, 0)
    
    mae = np.nanmean(mae, axis=tuple(range(mae.ndim - 1)))
    return mae

def threshold_relative_error(true, pred, mask=None):
    return 
