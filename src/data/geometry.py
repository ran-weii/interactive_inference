import numpy as np

def wrap_angles(x):
    """ Round angles to interval [-pi, pi] 
    Args:
        x (np.array): input angles in range [-inf, inf]
        return_revs(bool, optional): return wrap revolutions

    Returns:
        y (np.array): output angels in range [-pi, pi]
    """
    y = (x + np.pi) % (2 * np.pi) - np.pi
    return y

def dist_two_points(x1, y1, x2, y2):
    """ Two point distance formula """
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def get_heading(x1, y1, x2, y2):
    """ Heading of vector (x1, y1) -> (x2, y2) """
    delta_y = y2 - y1
    delta_x = x2 - x1
    return np.arctan2(delta_y, delta_x)

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
        diff_heading (float): heading difference in range (-pi, pi)
            left is positive, right is negative
    """
    assert heading >= -np.pi and heading <= np.pi
    
    # handle same point
    if x == a and y == b:
        return 0
    
    # find vector direction from (x, y) to (a, b)
    vec = (a - x, b - y)
    heading_vec = np.arctan2(vec[1], vec[0])
    diff_heading = heading_vec - heading

    # normalize to range (-pi, pi)
    # if diff_heading > np.pi:
    #     diff_heading = diff_heading - 2 * np.pi
    # elif diff_heading < -np.pi:
    #     diff_heading = diff_heading + 2 * np.pi
    diff_heading = wrap_angles(diff_heading)
    return diff_heading
      
def is_above_line(x, y, heading, a, b):
    """ Determine if point (a, b) is above (1) or below (-1) the line 
    defined by point (x, y) and heading 
    """
    # handle same point
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
    
def vector_projection(x1, y1, x2, y2, return_vec=False):
    """ Project vector (x1, y1) onto vector (x2, y2)
    
    Args:
        x1 (np.array): vector 1 x coor
        y1 (np.array): vector 1 y coor
        x2 (np.array): vector 2 x coor
        y2 (np.array): vector 3 y coor
        return_vec (bool, optional): return projection vector 
            in world coordinate, default=False
    
    Returns:
        x3 (np.array): projection vector x coor
        y3 (np.array): projection vector y coor
    """
    a = np.stack([x1, y1]).T
    b = np.stack([x2, y2]).T
    if not return_vec:
        b_norm = np.linalg.norm(b, ord=1, axis=1)
        x3 = x1*x2 + y1*y2
        y3 = (y1*x2 - x1*y2) / b_norm
    else:
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        b_unit = b / b_norm
        x3 = (x1*x2 + y1*y2).reshape(-1, 1) * b_unit
        y3 = a - x3
    return x3, y3