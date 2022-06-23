import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from src.data.geometry import dist_two_points, wrap_angles

def compute_arc_length(f_diff, a, b):
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
    return fsolve(lambda x: compute_arc_length(f_diff, b, x) - length, 0)[0]

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

def compute_tangent_and_normal_vectors(x, y, dt=0.1):
    """ Compute the tangent and normal vectors along a trajectory 
    using finite difference. Other quantities are computed but do not output
    
    Adapted from: https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy

    Args:
        x (np.array): x coordinates of the trajectory
        y (np.array): y coordinates of the trajectory
        dt (float, optional): time step. Default=0.1
    
    Returns:
        tan_vec (np.array): tangent vector [length, 2]
        norm_vec (np.array): normal vector [length, 2]
    """
    assert len(x) > 1, f"trajectory length={len(x)} is too short"
    dx = np.gradient(x) / dt
    dy = np.gradient(y) / dt
    ds = np.sqrt(dx**2 + dy**2)
    s = np.cumsum(ds)
    
    ddx = np.gradient(dx) / dt
    ddy = np.gradient(dy) / dt
    dds = np.gradient(ds) / dt

    # curvature
    a = dx * ddy - dy * ddx
    norm_square = dx**2 + dy**2
    kappa = a / norm_square ** 1.5
    
    # compute acceleration
    v_vec = np.stack([dx, dy]).T
    tan_vec = np.array([1 / ds] * 2).T * v_vec

    dtan_x = np.gradient(tan_vec[:, 0]) / dt
    dtan_y = np.gradient(tan_vec[:, 1]) / dt
    dtan = np.stack([dtan_x, dtan_y]).T
    dtan_norm = np.sqrt(dtan_x**2 + dtan_y**2)
    norm_vec = np.array([1 / (dtan_norm + 1e-8)] * 2).T * dtan
    
    t_component = np.array([dds] * 2).T
    n_component = np.array([np.abs(kappa) * ds**2] * 2).T
    acc = t_component * tan_vec + n_component * norm_vec
    return tan_vec, norm_vec