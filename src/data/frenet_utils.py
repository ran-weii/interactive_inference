import math
from scipy.integrate import quad
from scipy.optimize import fsolve
from src.data.geometry import dist_two_points

def get_arc_length(f_diff, a, b):
    """ Compute the arc length from point a to b on a 2D parameteric curve f 
    
    Args:
        f_diff (func): first order derivative of curve f
        a (float): x coordinate of starting point a
        b (float): b coordinate of ending point b

    Returns:
        (float): arc length integrated using scipy.quad
    """    
    func = lambda x: math.sqrt(1 + f_diff(x)**2)
    return quad(func, a, b)[0]

def where_arc_length(f_diff, length, b):
    """ Find a point at a specific arc length from a point b 
    alone a parameteric curve f with derivative f_diff
    
    Args:
        f_diff (func): first order derivative of curve f
        length (float): target arc length from point b
        b (float): x coordinate of starting point b

    Returns:
        (float): x coordinate of target point a
    """
    return fsolve(lambda x: get_arc_length(f_diff, b, x) - length, 0)[0]

def get_closest_point(f, x0, y0):
    """ Find the closest (tangent) point to (x0, y0) on a parameteric curve f 
    
    Args:
        f (func): function of curve f
        x0 (float): x coordinate of target point
        y0 (float): y coordinate of target point 

    Returns:
        x_tan (float): x coordinate of the tangent point
        y_tan (float): y coordinate of the tangent point
    """
    x_tan = fsolve(lambda x: dist_two_points(x0, y0, x, f(x)), x0)[0]
    y_tan = float(f(x_tan))
    return x_tan, y_tan

def compute_curvature(dx, ddx, dy, ddy):
    """ Compute the curvature of a curve parameterized by arc length x or y = f(s)
    at point with derivatives specified by inputs

    Args:
        dx (float): first order derivative of the curve's x coordinate
        ddx (float): second order derivative of the curve's x coordinate
        dy (float): first order derivative of the curve's y coordinate
        ddy (float): second order derivative of the curve's y coordinate

    Returns:
        (float): curvature kappa at the target point
    """
    a = dx*ddy - dy*ddx
    norm_square = dx*dx+dy*dy
    norm = math.sqrt(norm_square)
    b = norm*norm_square
    return a/b

def compute_curvature_derivative(dx, ddx, dddx, dy, ddy, dddy):
    """ Compute the curvature derivative of a curve parameterized by arc length x or y = f(s)
    at point with derivatives specified by inputs

    Args:
        dx (float): first order derivative of the curve's x coordinate
        ddx (float): second order derivative of the curve's x coordinate
        dddx (float): thrid order derivative of the curve's x coordinate
        dy (float): first order derivative of the curve's y coordinate
        ddy (float): second order derivative of the curve's y coordinate
        dddy (float): third order derivative of the curve's y coordinate

    Returns:
        (float): curvature derivative dkappa at the target point
    """
    a = dx*ddy-dy*ddx
    b = dx*dddy-dy*dddx
    c = dx*ddx+dy*ddy
    d = dx*dx+dy*dy
    return (b*d-3.0*a*c)/(d*d*d)