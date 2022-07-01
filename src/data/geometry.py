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

def angle_to_vector(theta):
    """ Map angle in radians to unit vector 
    
    Args:
        theta (np.array): angle in radians
    
    Returns:
        out (np.array): unit vector pointing in the direction of theta. 
            size=[batch_size, 2]
    """
    # whether theta is pointing to the right
    is_right = np.all([theta < np.pi/2, theta > -np.pi/2], axis=0)
    x = 1 * is_right + -1 * (is_right == False)
    y = x * np.tan(theta)
    
    out = np.stack([x, y]).T
    out /= np.linalg.norm(out, axis=1).reshape(-1, 1)
    return out

def dist_two_points(x1, y1, x2, y2):
    """ Two point distance formula """
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def mid_point(x1, y1, x2, y2):
    """ Mid point formula """
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    return x, y

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
        y2 (np.array): vector 2 y coor
        return_vec (bool, optional): return projection vector 
            in world coordinate, default=False
    
    Returns:
        x3 (np.array): projection vector x coor
        y3 (np.array): projection vector y coor
    """
    a = np.stack([x1, y1]).T
    b = np.stack([x2, y2]).T
    if not return_vec:
        b_norm = np.linalg.norm(b, ord=1, axis=-1)
        x3 = x1*x2 + y1*y2
        y3 = (y1*x2 - x1*y2) / b_norm
    else:
        b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
        b_unit = b / b_norm
        x3 = (x1*x2 + y1*y2).reshape(-1, 1) * b_unit
        y3 = a - x3
    return x3, y3

def coord_transformation(x, y, e1, e2, theta=None, inverse=False):
    """ 2D vector coordinate transformation 
    (e1, e2) is the x axis of the new coordinate system in the origional coordinate.
    
    Args:
        x (np.array): vector x coord
        y (np.array): vector y coord
        e1 (np.array): target coordinate x axis' x coord
        e2 (np.array): target coordinate x axis' y coord
        theta (np.array, optional): target coord rotation angle, if used overwrite e1 and e2
        inverse (bool ,optional): whether to perform inverse transformation. Defaults to False.
    
    Returns:
        x1 (np.array): x in new coord
        y1 (np.array): y in new coord
    """
    if theta is None:
        theta = np.arctan2(e2, e1)
    
    sin = np.sin(theta)
    cos = np.cos(theta)
    
    if not inverse:
        x1 = cos * x + sin * y
        y1 = -sin * x + cos * y
    else:
        x1 = cos * x - sin * y
        y1 = sin * x + cos * y
    return x1, y1