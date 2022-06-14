import math
import numpy as np
from scipy.interpolate import CubicSpline
from src.data.geometry import wrap_angles
from src.data.frenet_utils import (
    get_arc_length, get_closest_point, compute_curvature, compute_curvature_derivative)

class FrenetPath:
    """ Object used to store (reference) paths in the frenet frame """
    def __init__(self, ref_coords):
        """
        Args:
            ref_coords (np.array): x and y coordinates of the reference path sorted 
                in heading direction [num_points, 2]
        """
        ref_coords_sorted = ref_coords[np.argsort(ref_coords[:, 0])]
        # whether the x coordinates are reversed
        self.sign = 1
        if not np.all(ref_coords == ref_coords_sorted):
            self.sign = -1
        self.interpolator = CubicSpline(ref_coords_sorted[:, 0], ref_coords_sorted[:, 1])
        
        # interpolate spline
        step_size = 0.5
        num_grids = np.abs((ref_coords[0, 0] - ref_coords[-1, 0]) / step_size).astype(int)
        x_grid = np.linspace(ref_coords[0, 0], ref_coords[-1, 0], num_grids)
        y_grid = self.interpolator(x_grid)
        self.cubic_spline = np.stack([x_grid, y_grid]).T
        
        # total arc length of the path
        self.arc_length = get_arc_length(
            self.interpolator.derivative(),
            ref_coords_sorted[0, 0], ref_coords_sorted[-1, 0]
        )
        
        # parameterize the path's x and y coordinates wrt arc length
        s = np.linspace(0, self.arc_length, len(self.cubic_spline))
        self.fx = np.poly1d(np.polyfit(s, self.cubic_spline[:, 0], 5))
        self.dfx = self.fx.deriv()
        self.ddfx = self.dfx.deriv()
        self.dddfx = self.ddfx.deriv()
        
        self.fy = np.poly1d(np.polyfit(s, self.cubic_spline[:, 1], 5))
        self.dfy = self.fy.deriv()
        self.ddfy = self.dfy.deriv()
        self.dddfy = self.ddfy.deriv()
        
        # fit inverse of curve to locate x along arc length easily
        self.fx_inv = np.poly1d(np.polyfit(self.cubic_spline[:, 0], s, 5))
    
    def get_tangent(self, s):
        return math.atan2(self.dfy(s), self.dfx(s))
    
    def get_curvature(self, s):
        return compute_curvature(
            self.dfx(s), self.ddfx(s), self.dfy(s), self.ddfy(s)
        )
    
    def get_d_curvature(self, s):
        return compute_curvature_derivative(
            self.dfx(s), self.ddfx(s), self.dddfx(s), 
            self.dfy(s), self.ddfy(s), self.dddfy(s)
        )
    
    def frenet_to_cartesian(self, s_condition, d_condition):
        """ Convert from frenet to cartesian frame
        
        Args:
            s_condition (np.array): longitudinal component of state [s, ds, dds]
            d_condition (np.array): lateral component of state [d, dd, ddd]
        
        Returns:
            x (float): x coordinate in cartesian frame
            y (float): y coordinate in cartesian frame
            v (float): velocity in cartesian frame
            a (float): acceleration in cartesian frame
            theta (float): heading in cartesian frame
            kappa (float): curvature in cartesian frame
        """
        # get tangent point states
        rs = s_condition[0]
        rx = self.fx(rs)
        ry = self.fy(rs)
        rtheta = self.get_tangent(rs)
        rkappa = self.get_curvature(rs)
        rdkappa = self.get_d_curvature(rs)
        x, y, v, a, theta, kappa = frenet_to_cartesian(
            rs, rx, ry, rtheta, rkappa, rdkappa, s_condition, d_condition
        )
        return x, y, v, a, theta, kappa
    
    def get_tangent_point(self, x, y):
        """ Get the tangent point of (x, y) on the frenet path 
        
        Args: 
            x (float): x coordinate of target point
            y (float): y coordinate of target point
        
        Returns:
            rx (float): x coordinate of tangent point
            ry (float): y coordinate of tangent point
            rs (float): arc length at tagent point
        """
        rx, ry = get_closest_point(self.interpolator, x, y)
        rs = self.fx_inv(rx)        
        return rx, ry, rs
    
    def cartesian_to_frenet(self, x, y, v, a, theta, kappa):
        """ Convert from cartesian to frenet frame
        dds and ddd will be less accurate when deviating too far from the path
        
        Args:
            x (float): x coordinate in cartesian frame
            y (float): y coordinate in cartesian frame
            v (float): velocity in cartesian frame
            a (float): acceleration in cartesian frame
            theta (float): heading in cartesian frame
            kappa (float): curvature in cartesian frame
        
        Returns:
            s_condition (np.array): longitudinal component of state [s, ds, dds]
            d_condition (np.array): lateral component of state [d, dd, ddd]
        """
        rx, ry, rs = self.get_tangent_point(x, y)
        
        rtheta = self.get_tangent(rs)
        rkappa = self.get_curvature(rs)
        rdkappa = self.get_d_curvature(rs)
        s_condition, d_condition = cartesian_to_frenet(
            rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa
        )
        return s_condition, d_condition


class Trajectory:
    """ Object used to store agent trajectory in both cartesian and frenet frame 
    Requires a FrenetPath object to convert the trajectory to the frenet frame
    """
    def __init__(self, x, y, vx, vy, ax, ay, theta):
        """
        Args:
            x (np.array): array of agent x coordinate in cartesian frame
            y (np.array): array of agent y coordinate in cartesian frame
            vx (np.array): array of agent x velocity in cartesian frame
            vy (np.array): array of agent y velocity in cartesian frame
            ax (np.array): array of agent x acceleration in cartesian frame
            ay (np.array): array of agent y acceleration in cartesian frame
            theta (np.array): array of agent heading in cartesian frame
        """
        assert len(x) > 1, f"trajectory length={len(x)} is too short"
        
        # cartesian trajectory
        self.x = x
        self.y = y
        self.v = np.linalg.norm(np.stack([vx, vy]), axis=0)
        self.a = np.linalg.norm(np.stack([ax, ay]), axis=0)
        self.theta = theta
        self.kappa = None
        
        # trajectory properties 
        self.length = len(self.x)
        self.interpolator = None
        self.arc_length = None
        self._interpolate()
        
        # frenet trajectory
        self.s_condition = None
        self.d_condition = None
    
    def _interpolate(self):
        """ Interpolate trajectory to obtain curvature """
        x = self.x[np.argsort(self.x)].copy()
        y = self.y[np.argsort(self.x)].copy()
        
        self.interpolator = CubicSpline(x, y)
        self.arc_length = get_arc_length(
            self.interpolator.derivative(), self.x[0], self.x[-1]
        )
        
        s = np.linspace(0, self.arc_length, len(self.x))
        self.fx = np.poly1d(np.polyfit(s, self.x, 5))
        self.dfx = self.fx.deriv()
        self.ddfx = self.dfx.deriv()
        self.dddfx = self.ddfx.deriv()
        
        self.fy = np.poly1d(np.polyfit(s, self.y, 5))
        self.dfy = self.fy.deriv()
        self.ddfy = self.dfy.deriv()
        self.dddfy = self.ddfy.deriv()
        
        self.kappa = np.array([compute_curvature(
            self.dfx(s[i]), self.ddfx(s[i]), self.dfy(s[i]), self.ddfy(s[i])
        ) for i in range(len(s))])
    
    def get_frenet_trajectory(self, ref_path):
        """ Convert self trajectory from cartesian to frenet frame
        
        Args:
            ref_path (FrenetPath): the reference path object
        """
        s_condition = [np.empty(0)] * self.length
        d_condition = [np.empty(0)] * self.length
        for t in range(self.length):
            s_condition[t], d_condition[t] = ref_path.cartesian_to_frenet(
                self.x[t], self.y[t], self.v[t],
                self.a[t], self.theta[t], self.kappa[t]
            )
        self.s_condition = np.array(s_condition)
        self.d_condition = np.array(d_condition)


def cartesian_to_frenet(rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa):
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
    
    Returns:
        s_condition (np.array): longitudinal component of the target point in frenet frame [s, ds, dds]
        d_condition (np.array): lateral component of the target point in frenet frame [d, dd, ddd]
    """
    s_condition = np.zeros(3)
    d_condition = np.zeros(3)
    
    dx = x - rx
    dy = y - ry
    
    cos_theta_r = math.cos(rtheta)
    sin_theta_r = math.sin(rtheta)
    
    cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
    d_condition[0] = math.copysign(math.sqrt(dx * dx + dy * dy), cross_rd_nd)
    
    delta_theta = theta - rtheta
    tan_delta_theta = math.tan(delta_theta)
    cos_delta_theta = math.cos(delta_theta)
    
    one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
    d_condition[1] = one_minus_kappa_r_d * tan_delta_theta
    
    kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]
    
    d_condition[2] = (-kappa_r_d_prime * tan_delta_theta + 
        one_minus_kappa_r_d / cos_delta_theta / cos_delta_theta *
        (kappa * one_minus_kappa_r_d / cos_delta_theta - rkappa))
    
    s_condition[0] = rs
    s_condition[1] = v * cos_delta_theta / one_minus_kappa_r_d
    
    delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
    s_condition[2] = ((a * cos_delta_theta -
        s_condition[1] * s_condition[1] *
        (d_condition[1] * delta_theta_prime - kappa_r_d_prime)) /
        one_minus_kappa_r_d)
    return s_condition, d_condition

def frenet_to_cartesian(rs, rx, ry, rtheta, rkappa, rdkappa, s_condition, d_condition):
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
    
    Returns:
        x (float): x coordinate of the target point in cartesian frame
        y (float): y coordinate of the target point in cartesian frame
        v (float): velocity norm of the target point in cartesian frame
        a (float): acceleration norm the target point in cartesian frame
        theta (float): heading of the target point in cartesian frame
        kappa (float): curvature of the target point trajectory in cartesian frame
    """
    if math.fabs(rs - s_condition[0])>= 1.0e-6:
        print("The reference point s and s_condition[0] don't match")
    
    cos_theta_r = math.cos(rtheta)
    sin_theta_r = math.sin(rtheta)
    
    x = rx - sin_theta_r * d_condition[0]
    y = ry + cos_theta_r * d_condition[0]
    
    one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
    tan_delta_theta = d_condition[1] / one_minus_kappa_r_d
    delta_theta = math.atan2(d_condition[1], one_minus_kappa_r_d)
    cos_delta_theta = math.cos(delta_theta)
    
    theta = float(wrap_angles(delta_theta + rtheta))
    kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]
    
    kappa = ((((d_condition[2] + kappa_r_d_prime * tan_delta_theta) *
        cos_delta_theta * cos_delta_theta) / (one_minus_kappa_r_d) 
        + rkappa) * cos_delta_theta / (one_minus_kappa_r_d))
    
    d_dot = d_condition[1] * s_condition[1]
    
    v = math.sqrt(one_minus_kappa_r_d**2 * s_condition[1]**2 + d_dot**2)
    
    delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * (kappa) - rkappa     
    a = (s_condition[2] * one_minus_kappa_r_d / cos_delta_theta +
        s_condition[1] * s_condition[1] / cos_delta_theta *
        (d_condition[1] * delta_theta_prime - kappa_r_d_prime))
    return x, y, v, a, theta, kappa 