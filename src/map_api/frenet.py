import math
import numpy as np
from scipy.interpolate import CubicSpline
from src.data.geometry import wrap_angles
from src.map_api.frenet_utils import (
    compute_arc_length, get_closest_point, compute_curvature, 
    compute_curvature_derivative, compute_tangent_and_normal_vectors)
from src.map_api.frenet_utils import cartesian_to_frenet, frenet_to_cartesian

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
        self.arc_length = compute_arc_length(
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
    
    def frenet_to_cartesian(self, s_condition, d_condition, order=3):
        """ Convert from frenet to cartesian frame
        
        Args:
            s_condition (np.array): longitudinal component of state [s, ds, dds]
            d_condition (np.array): lateral component of state [d, dd, ddd]
            order (int, optional): order to derivative to convert. Default=3
        
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
            rs, rx, ry, rtheta, rkappa, rdkappa, s_condition, d_condition, order
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
    
    def cartesian_to_frenet(self, x, y, v, a, theta, kappa, order=3):
        """ Convert from cartesian to frenet frame
        dds and ddd will be less accurate when deviating too far from the path
        
        Args:
            x (float): x coordinate in cartesian frame
            y (float): y coordinate in cartesian frame
            v (float): velocity in cartesian frame
            a (float): acceleration in cartesian frame
            theta (float): heading in cartesian frame
            kappa (float): curvature in cartesian frame
            order (int, optional): order to derivative to convert. Default=3
        
        Returns:
            s_condition (np.array): longitudinal component of state [s, ds, dds]
            d_condition (np.array): lateral component of state [d, dd, ddd]
        """
        rx, ry, rs = self.get_tangent_point(x, y)
        
        rtheta = self.get_tangent(rs)
        rkappa = self.get_curvature(rs)
        rdkappa = self.get_d_curvature(rs)
        s_condition, d_condition = cartesian_to_frenet(
            rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa, order
        )
        return s_condition, d_condition


class Trajectory:
    """ Object used to store agent trajectory in both cartesian and frenet frame 
    Requires a FrenetPath object to convert the trajectory to the frenet frame
    """
    def __init__(self, x, y, vx, vy, ax, ay, theta, dt=0.1):
        """
        Args:
            x (np.array): array of agent x coordinate in cartesian frame
            y (np.array): array of agent y coordinate in cartesian frame
            vx (np.array): array of agent x velocity in cartesian frame
            vy (np.array): array of agent y velocity in cartesian frame
            ax (np.array): array of agent x acceleration in cartesian frame
            ay (np.array): array of agent y acceleration in cartesian frame
            theta (np.array): array of agent heading in cartesian frame
            dt (float, optional): time step. Default=0.1
        """
        assert len(x) > 1, f"trajectory length={len(x)} is too short"
        self.dt = dt

        # compute acceleration sign
        acc_vec = np.arctan2(ay, ax)
        delta_vec = wrap_angles(acc_vec - theta)
        self.sign = np.ones_like(x)
        self.sign[delta_vec > 0.5 * np.pi] = -1
        self.sign[delta_vec < 0.5 * -np.pi] = -1
        
        # cartesian trajectory
        self.x = x
        self.y = y
        self.v = np.linalg.norm(np.stack([vx, vy]), axis=0)
        self.a = self.sign * np.linalg.norm(np.stack([ax, ay]), axis=0)
        self.theta = theta
        self.kappa = None

        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay
        self.tan_vec = None
        self.norm_vec = None
        
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
        # x = self.x[np.argsort(self.x)]
        # y = self.y[np.argsort(self.x)]
        # self.interpolator = CubicSpline(x, y)
        # self.arc_length = np.abs(compute_arc_length(
        #     self.interpolator.derivative(), x[0], x[-1]
        # ))
        # s = np.linspace(0, self.arc_length, self.length)

        dx = np.hstack([np.array([0]), np.diff(self.x)])
        dy = np.hstack([np.array([0]), np.diff(self.y)])
        ds = np.sqrt(dx**2 + dy**2)
        s = np.cumsum(ds)
        self.arc_length = s[-1]
        
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
        ) for i in range(self.length)])
        
        self.tan_vec, self.norm_vec = compute_tangent_and_normal_vectors(
            self.x, self.y, self.dt
        )

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
