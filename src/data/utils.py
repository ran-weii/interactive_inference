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

def dist_two_points(x1, y1, x2, y2):
    """ Two point distance formula """
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def closest_point_on_line(x, y, x_line, y_line):
    """ Find the closest point (a, b) to an external point (x, y) on a line segement
    
    Args:
        x (float): x coor of target point
        y (float): y coor of target point
        x_line (list): x coors of target line
        y_line (list): y coors of target line
        
    Returns:
        a (float): x coor of closest point
        b (float): y coor of closest point
    """
    [x1, x2] = x_line
    [y1, y2] = y_line
    
    px = x2 - x1
    py = y2 - y1
    norm = px * px + py * py
    
    # fraction of tangent point on line
    u = ((x - x1) * px + (y - y1) * py) / norm
    
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    
    # closest point (a, b)
    a = x1 + u * px
    b = y1 + u * py
    return a, b

def get_cardinal_direction(x, y, heading, a, b):
    """ Determine the cardianl direction of point (a, b) 
    with respect to point (x, y) with heading 
    
    Args:
        x (float): x coor of anchor point
        y (float): y coor of anchor point
        heading (float): heading of point (x, y)
        a (float): x coor of target point
        b (float): y coor of target point
        
    Returns:
        diff_heading (float): heading difference in (-pi, pi)
    """
    assert heading >= -np.pi and heading <= np.pi
    
    # find vector direction from (x, y) to (a, b)
    vec = (a - x, b - y)
    heading_vec = np.arctan2(vec[1], vec[0])
    diff_heading = heading_vec - heading
    
    # normalize to range (-pi, pi)
    if diff_heading > np.pi:
        diff_heading = diff_heading - 2 * np.pi
    elif diff_heading < -np.pi:
        diff_heading = diff_heading + 2 * np.pi
    return diff_heading
      
def is_above_line(x, y, heading, a, b):
    """ Determine if point (a, b) is above (1) or below (-1) the line 
    defined by point (x, y) and heading 
    """
    if x == a and y == b:
        return 0
    
    # extend x, y along heading
    heading_slope = np.tan(heading)
    intercept = y - heading_slope * x
    b_pred = heading_slope * a + intercept
    
    if b_pred > b: 
        return -1
    else: 
        return 1