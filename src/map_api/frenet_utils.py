import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve, minimize
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
    x_tan = minimize(lambda x: dist_two_points(x0, y0, x, f(x)), x0)["x"][0]
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
    eps = 1e-8
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
    kappa = a / (norm_square + eps) ** 1.5
    
    # compute acceleration
    v_vec = np.stack([dx, dy]).T
    tan_vec = np.array([1 / (ds + eps)] * 2).T * v_vec

    dtan_x = np.gradient(tan_vec[:, 0]) / dt
    dtan_y = np.gradient(tan_vec[:, 1]) / dt
    dtan = np.stack([dtan_x, dtan_y]).T
    dtan_norm = np.sqrt(dtan_x**2 + dtan_y**2)
    norm_vec = np.array([1 / (dtan_norm + eps)] * 2).T * dtan
    
    t_component = np.array([dds] * 2).T
    n_component = np.array([np.abs(kappa) * ds**2] * 2).T
    acc = t_component * tan_vec + n_component * norm_vec
    return tan_vec, norm_vec

def compute_acceleration_vector(dds, ds, kappa, tan_vec, norm_vec):
    """ Compute acceleration vector using tangent and normal components 
    
    Args:
        dds (np.array): signed acceleration alone the ego trajectory. size=[batch_size]
        ds (np.array): signed velocity alone the ego trajectory. size=[batch_size]
        kappa (np.array): curvature of the ego trajectory. size=[batch_size]
        tan_vec (np.array): tangent uint vector. size=[batch_size, 2]
        norm_vec (np.array): normal unit vectory. size=[batch_size, 2]
    
    Returns:
        acc_vec (np.array): acceleration vector. size=[batch_size, 2]
    """
    t_component = dds.reshape(-1, 1)
    n_component = (np.abs(kappa) * ds**2).reshape(-1, 1)
    acc_vec = t_component * tan_vec + n_component * norm_vec
    return acc_vec

def compute_normal_from_kappa(theta, kappa):
    """ Compute normal vector direction from tangent and curvature 
    
    Args:
        theta (float, np.array): tangent vector direction
        kappa (float, np.array): trajectory curvature
    
    Returns:
        norm (np.array): normal vector direction
    """
    kappa_sign = np.sign(kappa + 1e-6)
    norm = theta + np.pi/2 * kappa_sign 
    norm = wrap_angles(norm)
    return norm

def cartesian_to_frenet(rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa, order=3):
    """ Convert from cartesian to frenet coordinate
    Adapted from: https://github.com/ApolloAuto/apollo
    
    Args:
        rs (float): arc length of the tangent point in frenet frame
        rx (float): x coordinate arc of the tangent point in cartesian frame
        ry (float): y coordinate arc of the tangent point in cartesian frame
        rtheta (float): heading of the tangent point in cartesian frame
        rkappa (float): curvature of the tangent point
        rdkappa (float): curvature derivative of the tangent point
        x (float): x coordinate of the target point in cartesian frame
        y (float): y coordinate of the target point in cartesian frame
        v (float): velocity norm of the target point in cartesian frame
        a (float): acceleration norm the target point in cartesian frame
        theta (float): heading of the target point in cartesian frame
        kappa (float): curvature of the target point trajectory in cartesian frame
        order (int, optional): order to derivative to convert. Default=3
    
    Returns:
        s_condition (np.array): longitudinal component of the target point in frenet frame [s, ds, dds]
        d_condition (np.array): lateral component of the target point in frenet frame [d, dd, ddd]
    """
    assert order in [1, 2, 3]
    s_condition = np.zeros(order)
    d_condition = np.zeros(order)
    
    dx = x - rx
    dy = y - ry
    
    cos_theta_r = math.cos(rtheta)
    sin_theta_r = math.sin(rtheta)
    
    cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx

    # first order conversion
    s_condition[0] = rs
    d_condition[0] = math.copysign(math.sqrt(dx * dx + dy * dy), cross_rd_nd)
    
    # second order conversion
    if order >= 2:
        delta_theta = theta - rtheta
        tan_delta_theta = math.tan(delta_theta)
        cos_delta_theta = math.cos(delta_theta)

        one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
        s_condition[1] = v * cos_delta_theta / one_minus_kappa_r_d
        d_condition[1] = one_minus_kappa_r_d * tan_delta_theta
    
    # thir order conversion
    if order >= 3:
        kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]
        d_condition[2] = (-kappa_r_d_prime * tan_delta_theta + 
            one_minus_kappa_r_d / cos_delta_theta / cos_delta_theta *
            (kappa * one_minus_kappa_r_d / cos_delta_theta - rkappa))
        
        delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
        s_condition[2] = ((a * cos_delta_theta -
            s_condition[1] * s_condition[1] *
            (d_condition[1] * delta_theta_prime - kappa_r_d_prime)) /
            one_minus_kappa_r_d)
    return s_condition, d_condition

def frenet_to_cartesian(rs, rx, ry, rtheta, rkappa, rdkappa, s_condition, d_condition, order=3):
    """ Convert from frenet to cartesian coordinate
    Adapted from: https://github.com/ApolloAuto/apollo
    
    Args:
        rs (float): arc length of the tangent point in frenet frame
        rx (float): x coordinate arc of the tangent point in cartesian frame
        ry (float): y coordinate arc of the tangent point in cartesian frame
        rtheta (float): heading of the tangent point in cartesian frame
        rkappa (float): curvature of the tangent point
        rdkappa (float): curvature derivative of the tangent point
        s_condition (np.array): longitudinal component of the target point in frenet frame [s, ds, dds]
        d_condition (np.array): lateral component of the target point in frenet frame [d, dd, ddd]
        order (int, optional): order to derivative to convert. Default=3

    Returns:
        x (float): x coordinate of the target point in cartesian frame
        y (float): y coordinate of the target point in cartesian frame
        v (float): velocity norm of the target point in cartesian frame
        a (float): acceleration norm the target point in cartesian frame
        theta (float): heading of the target point in cartesian frame
        kappa (float): curvature of the target point trajectory in cartesian frame
    """
    assert order in [1, 2, 3]
    if math.fabs(rs - s_condition[0])>= 1.0e-6:
        print("The reference point s and s_condition[0] don't match")

    [x, y, v, a, theta, kappa] = [None] * 6

    cos_theta_r = math.cos(rtheta)
    sin_theta_r = math.sin(rtheta)
    
    # first order conversion
    x = rx - sin_theta_r * d_condition[0]
    y = ry + cos_theta_r * d_condition[0]
    
    # second order conversion
    if order >= 2:
        one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
        d_dot = d_condition[1] * s_condition[1]
        delta_theta = math.atan2(d_condition[1], one_minus_kappa_r_d)
        v = math.sqrt(one_minus_kappa_r_d**2 * s_condition[1]**2 + d_dot**2)
        theta = float(wrap_angles(delta_theta + rtheta))

    # third order conversion
    if order >= 3:
        tan_delta_theta = d_condition[1] / one_minus_kappa_r_d
        cos_delta_theta = math.cos(delta_theta)
        
        kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]
        
        kappa = ((((d_condition[2] + kappa_r_d_prime * tan_delta_theta) *
            cos_delta_theta * cos_delta_theta) / (one_minus_kappa_r_d) 
            + rkappa) * cos_delta_theta / (one_minus_kappa_r_d))
        
        delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * (kappa) - rkappa     
        a = (s_condition[2] * one_minus_kappa_r_d / cos_delta_theta +
            s_condition[1] * s_condition[1] / cos_delta_theta *
            (d_condition[1] * delta_theta_prime - kappa_r_d_prime))
    return x, y, v, a, theta, kappa 