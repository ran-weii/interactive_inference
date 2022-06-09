import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString, Polygon
from shapely import ops
from scipy.interpolate import CubicSpline
from src.data.geometry import (
    get_heading, get_cardinal_direction, mid_point, dist_two_points, wrap_angles)

class L2Point:
    def __init__(self, id_, metric_point, geo_point, type_, point_subtype):
        self.id_ = id_
        self.point = metric_point
        self.geo_point = geo_point
        self.type_ = type_
        self.subtype = point_subtype


class L2Linestring:
    def __init__(self, id_, linestring, type_, subtype):
        self.id_ = id_
        self.linestring = linestring
        self.type_ = type_
        self.subtype = subtype
        
        self.lanelet_references = []

        self.interpolator = None
        self.cubic_spline = None
        self.spline_heading = None
        self.spline_cumdist = None
    
    def add_reference(self, lanelet_id):
        self.lanelet_references.append(lanelet_id)

    def _interpolate(self, step=1):
        coords = np.array(list(self.linestring.coords))
        coords_sorted = coords[np.argsort(coords[:, 0])]
        
        self.interpolator = CubicSpline(coords_sorted[:, 0], coords_sorted[:, 1])
        
        # interpolate
        num_grids = np.abs((coords[0, 0] - coords[-1, 0]) / step).astype(int)
        x_grid = np.linspace(coords[0, 0], coords[-1, 0], num_grids)
        y_grid = self.interpolator(x_grid)
        self.cubic_spline = np.stack([x_grid, y_grid]).T
        self.spline_heading = get_heading(x_grid[:-1], y_grid[:-1], x_grid[1:], y_grid[1:])
        self.spline_heading = np.insert(self.spline_heading, -1, self.spline_heading[-1])
        self.spline_cumdist = np.cumsum(dist_two_points(
            self.cubic_spline[:-1, 0], self.cubic_spline[:-1, 1],
            self.cubic_spline[1:, 0], self.cubic_spline[1:, 1]
        ))
        self.spline_cumdist = np.insert(self.spline_cumdist, -1, self.spline_cumdist[-1])

     
class L2Polygon:
    def __init__(self, id_, polygon, type_, subtype):
        self.id_ = id_
        self.polygon = polygon
        self.type_ = type_
        self.subtype = subtype
            

class Cell:
    """ Section of a lane, represented as a Shapely polygon, with a defined heading """
    def __init__(self, polygon, heading, left_bound, right_bound):
        self.polygon = polygon  # Shapely polygon
        self.heading = heading  # radians clockwise from y-axis
        self.left_bound = left_bound
        self.right_bound = right_bound
        
        left_bound_coords = list(left_bound.coords)
        right_bound_coords = list(right_bound.coords)
        self.left_bound_heading = get_heading(
            left_bound_coords[0][0], left_bound_coords[0][1], 
            left_bound_coords[1][0], left_bound_coords[1][1]
        )
        self.right_bound_heading = get_heading(
            right_bound_coords[0][0], right_bound_coords[0][1], 
            right_bound_coords[1][0], right_bound_coords[1][1]
        )

        self._center_line = None
        center_line_coords = list(self.center_line.coords)
        self.center_line_heading = get_heading(
            center_line_coords[0][0], center_line_coords[0][1],
            center_line_coords[1][0], center_line_coords[1][1]
        )
    
    @property
    def center_line(self):
        if self._center_line:
            return self._center_line
        
        left_bound_coords = list(self.left_bound.coords)
        right_bound_coords = list(self.right_bound.coords)

        pt1 = mid_point(
            left_bound_coords[0][0], left_bound_coords[0][1], 
            right_bound_coords[0][0], right_bound_coords[0][1], 
        )
        pt2 = mid_point(
            left_bound_coords[1][0], left_bound_coords[1][1], 
            right_bound_coords[1][0], right_bound_coords[1][1], 
        )
        self._center_line = LineString([pt1, pt2])
        return self._center_line

       
class Lanelet:
    def __init__(self, id_, subtype, region, location, one_way, 
        turn_direction, vehicle_participant, pedestrian_participant, 
        bicycle_participant, left_bound=None, right_bound=None, 
        centerline=None, regulatory_elements=[], cell_len=5., buffer_=0):
        self.id_ = id_
        self.subtype = subtype
        self.region = region
        self.location = location
        self.one_way = one_way
        self.turn_direction = turn_direction
        self.vehicle_participant = vehicle_participant
        self.pedestrian_participant = pedestrian_participant
        self.bicycle_participant = bicycle_participant
        self.regulatory_elements = regulatory_elements
        self.cell_len = cell_len
        self.buffer_ = buffer_

        # L2Linestring
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.centerline = centerline

        # calculated fields for property methods
        self._polygon = None
        self._cells = []
    
    def has_opposing_linestrings(self):
        """ Determines if a lanelet's left and right bounds have opposing headings """
        left_bound_coords = list(self.left_bound.linestring.coords)
        right_bound_coords = list(self.right_bound.linestring.coords)

        left_head = Point(left_bound_coords[-1])  # last point of the left bound 
        right_tail = Point(right_bound_coords[0])  # first point of the right bound
        right_head = Point(right_bound_coords[-1])  # last point of the right bound
        return True if left_head.distance(right_head) > left_head.distance(right_tail) else False
    
    def _align_bounds(self):
        """ Sort left and right bound points in the heading direction """
        left_bound_coords = list(self.left_bound.linestring.coords)
        right_bound_coords = list(self.right_bound.linestring.coords)
        if self.has_opposing_linestrings():
            right_bound_coords.reverse()
            self.right_bound.linestring = LineString(right_bound_coords)
            
        left_tail0 = Point(left_bound_coords[0]) 
        right_tail0 = Point(right_bound_coords[0]) 
        right_tail1 = Point(right_bound_coords[1]) 
        
        # compute right bound vector (0 -> 1) heading and left_tail0 cardinal direction
        right_heading = get_heading(
            right_tail0.x, right_tail0.y, right_tail1.x, right_tail1.y
        )
        card = get_cardinal_direction(
            right_tail0.x, right_tail0.y, right_heading, left_tail0.x, left_tail0.y
        )
        
        if card < 0 and card >= -np.pi: # left_tail0 to the right of right bound vector
            left_bound_coords.reverse()
            right_bound_coords.reverse()
            self.left_bound.linestring = LineString(left_bound_coords)
            self.right_bound.linestring = LineString(right_bound_coords)
    
    def _find_centerline(self):
        if self.centerline is None:
            centerline = ops.linemerge([c.center_line for c in self.cells])
            self.centerline = L2Linestring(self.id_, centerline, None, None)

    @property
    def polygon(self):
        if self._polygon:
            return self._polygon
        
        left_bound_coords = list(self.left_bound.linestring.coords)
        right_bound_coords = list(self.right_bound.linestring.coords)
        
        # reversal will occur if bounds point the same direction
        if not self.has_opposing_linestrings():
            right_bound_coords.reverse()
   
        left_bound_coords.extend(right_bound_coords)
        self._polygon = Polygon(left_bound_coords).buffer(self.buffer_)
        return self._polygon

    @property
    def cells(self):
        """ List of polygons with max distance of self.cell_len """
        if self._cells:
            return self._cells

        # reverse left bound if opposed
        right_bound_linestr = self.right_bound.linestring
        left_bound_linestr = self.left_bound.linestring
        if self.has_opposing_linestrings():
            left_bound_linestr = LineString(self.left_bound.linestring.coords[::-1]) 
        
        # determine which linestring is longer
        right_is_longer = True
        if right_bound_linestr.length > left_bound_linestr.length:
            longer_linestr = right_bound_linestr
            shorter_linestr = left_bound_linestr
        else:
            longer_linestr = left_bound_linestr
            shorter_linestr = right_bound_linestr
            right_is_longer = False
        
        # interpolate shorter linestring by cell distance
        shorter_distances = np.arange(0, shorter_linestr.length, self.cell_len)
        shorter_points = [shorter_linestr.interpolate(d) for d in shorter_distances] + [shorter_linestr.boundary[1]]
        longer_distances = [0] + [longer_linestr.project(p) for p in shorter_points[1:-1]]
        longer_points = [longer_linestr.interpolate(d) for d in longer_distances] + [longer_linestr.boundary[1]]
        for i in range(len(shorter_points) - 1):
            cell_coords = [(p.x, p.y) for p in [shorter_points[i], shorter_points[i+1], longer_points[i+1], longer_points[i]]]
            cell_polygon = Polygon(cell_coords).buffer(self.buffer_)
            
            if right_is_longer:
                cell_left_bound = LineString([shorter_points[i], shorter_points[i+1]])
                cell_right_bound = LineString([longer_points[i], longer_points[i+1]])
            else:
                cell_left_bound = LineString([longer_points[i], longer_points[i+1]])
                cell_right_bound = LineString([shorter_points[i], shorter_points[i+1]])
            
            cell_heading = get_heading(
                shorter_points[i].x, shorter_points[i].y, 
                shorter_points[i+1].x, shorter_points[i+1].y
            )
            
            cell = Cell(cell_polygon, cell_heading, cell_left_bound, cell_right_bound)
            self._cells.append(cell)    
        return self._cells


class Lane:
    def __init__(self, id_, lanelets, cell_len=5.):
        self.id_ = id_
        self.lanelets = [l for l in lanelets]
        self.buffer_ = 0
        self.cell_len = cell_len
        
        self.left_bound = None
        self.right_bound = None
        self.centerline = None
        
        self._polygon = None
        self._cells = []
        self._align_lanelets()
        
        self.left_adjacent_lane_id = []
        self.right_adjacent_lane_id = []
    
    @property
    def polygon(self):
        if self._polygon:
            return self._polygon
        
        lanelet_polygons = [l.polygon for l in self.lanelets if l.subtype != "crosswalk"]
        self._polygon = ops.unary_union(lanelet_polygons)
        return self._polygon
    
    @property
    def cells(self):
        if self._cells:
            return self._cells
        
        left_bound_linestr = self.left_bound.linestring
        right_bound_linestr = self.right_bound.linestring
        
        # determine which linestring is longer
        right_is_longer = True
        if right_bound_linestr.length > left_bound_linestr.length:
            longer_linestr = right_bound_linestr
            shorter_linestr = left_bound_linestr
        else:
            longer_linestr = left_bound_linestr
            shorter_linestr = right_bound_linestr
            right_is_longer = False
        
        # interpolate shorter linestring by cell distance
        shorter_distances = np.arange(0, shorter_linestr.length, self.cell_len)
        shorter_points = [shorter_linestr.interpolate(d) for d in shorter_distances] + [shorter_linestr.boundary[-1]]
        longer_distances = [0] + [longer_linestr.project(p) for p in shorter_points[1:-1]]
        longer_points = [longer_linestr.interpolate(d) for d in longer_distances] + [longer_linestr.boundary[-1]]
        for i in range(len(shorter_points) - 1):
            cell_coords = [(p.x, p.y) for p in [shorter_points[i], shorter_points[i+1], longer_points[i+1], longer_points[i]]]
            cell_polygon = Polygon(cell_coords).buffer(self.buffer_)
            
            if right_is_longer:
                cell_left_bound = LineString([shorter_points[i], shorter_points[i+1]])
                cell_right_bound = LineString([longer_points[i], longer_points[i+1]])
            else:
                cell_left_bound = LineString([longer_points[i], longer_points[i+1]])
                cell_right_bound = LineString([shorter_points[i], shorter_points[i+1]])
            
            cell_heading = get_heading(
                shorter_points[i].x, shorter_points[i].y, 
                shorter_points[i+1].x, shorter_points[i+1].y
            )
            
            cell = Cell(cell_polygon, cell_heading, cell_left_bound, cell_right_bound)
            self._cells.append(cell)    
        return self._cells
    
    def _align_lanelets(self):
        """ Sort lanelets by heading direction and add lane bounds as properties """
        def get_order(node1, node2):
            """ Return topological order between node1 and node2 """
            left_bound_linestr1 = node1.left_bound.linestring
            left_bound_linestr2 = node2.left_bound.linestring
            pt = left_bound_linestr1.intersection(left_bound_linestr2)
            if pt.is_empty:
                return None
            else:
                left_bound_start_pt1 = left_bound_linestr1.boundary[0]
                left_bound_end_pt1 = left_bound_linestr1.boundary[-1]
                left_bound_start_pt2 = left_bound_linestr2.boundary[0]
                left_bound_end_pt2 = left_bound_linestr2.boundary[-1]
                if pt == left_bound_end_pt1 and pt == left_bound_start_pt2: # node1 -> node2
                    return "parent"
                elif pt == left_bound_start_pt1 and pt == left_bound_end_pt2: # node2 -> node1
                    return "child"
                else:
                    return None
        
        G = nx.DiGraph()
        for i in range(len(self.lanelets) - 1):
            for j in range(i+1, len(self.lanelets)):
                node_id1, node_value1 = self.lanelets[i].id_, self.lanelets[i]
                node_id2, node_value2 = self.lanelets[j].id_, self.lanelets[j]
                order =  get_order(node_value1, node_value2)
                if order == "parent":
                    G.add_edge((node_id1, node_value1), (node_id2, node_value2))
                elif order == "child":
                    G.add_edge((node_id2, node_value2), (node_id1, node_value1))
        
        sorted_nodes = nx.topological_sort(G)
        self.lanelets = [n[1] for n in sorted_nodes]
        
        left_bound_linestr = ops.linemerge([l.left_bound.linestring for l in self.lanelets])
        right_bound_linestr = ops.linemerge([l.right_bound.linestring for l in self.lanelets])
        centerline_linestr = ops.linemerge([l.centerline.linestring for l in self.lanelets])
        
        self.left_bound = L2Linestring(self.id_, left_bound_linestr, None, None)
        self.right_bound = L2Linestring(self.id_, right_bound_linestr, None, None)
        self.centerline = L2Linestring(self.id_, centerline_linestr, None, None)
        
        self.left_bound._interpolate(step=1)
        self.right_bound._interpolate(step=1)
        self.centerline._interpolate(step=1)
    
    def get_frenet_coords(self, x, y):
        """ Get the lane centric frenet coordinates of point (x, y)

        Args:
            x (float): x coord of the target point
            y (float): y coord of the target point
        
        Returns:
            x_tan (float): x coord of the tangent point 
            y_tan (float): y coord of the tangent point
            psi_tan (float): heading of the trangent line
            centerline_dist (float): signed distance to the center line. 
                Left to the center line is positive
            left_bound_dist (float): signed distance to the left bound. 
                Left to the left bound is positive
            right_bound_dist (float): signed distance to the right bound. 
                Left to the right bound is positive
        """
        # find the tangent point
        p = Point([x, y])
        x_grid = np.linspace(x - 10, x + 10, 50) 
        y_grid = self.centerline.interpolator(x_grid)
        grid_linestr = LineString(np.stack([x_grid, y_grid]).T)
        p_proj = grid_linestr.interpolate(grid_linestr.project(p))
        x_tan, y_tan = p_proj.x, p_proj.y
        
        # get the heading of the closet point on centerline
        centerline_spline = self.centerline.cubic_spline
        centerline_heading = self.centerline.spline_heading
        d = dist_two_points(x_tan, y_tan, centerline_spline[:, 0], centerline_spline[:, 1])
        psi_adj = centerline_heading[np.argmin(d)]
        
        # get tanget line heading
        slope = self.centerline.interpolator(p_proj.x, 1)
        psi_tan = np.arctan2(slope, 1)
        if np.abs(wrap_angles(psi_tan - psi_adj)) > np.deg2rad(30):
            psi_tan = np.arctan2(slope, -1)

        # get the normal line
        slope_norm = -1 / slope
        normal_linestring = LineString([
            (x_tan - 10, y_tan - 10 * slope_norm), (x_tan + 10, y_tan + 10 * slope_norm)
        ])
        left_bound_pt = normal_linestring.intersection(self.left_bound.linestring)
        right_bound_pt = normal_linestring.intersection(self.right_bound.linestring)
        
        # get signed lane distances
        centerline_dist = dist_two_points(x, y, x_tan, y_tan)
        left_bound_dist = dist_two_points(x, y, left_bound_pt.x, left_bound_pt.y)
        right_bound_dist = dist_two_points(x, y, right_bound_pt.x, right_bound_pt.y)

        centerline_card = get_cardinal_direction(x_tan, y_tan, psi_tan, x, y)
        left_bound_card = get_cardinal_direction(left_bound_pt.x, left_bound_pt.y, psi_tan, x, y)
        right_bound_card = get_cardinal_direction(right_bound_pt.x, right_bound_pt.y, psi_tan, x, y)
        
        centerline_dist *= 2 * np.heaviside(centerline_card, 1) - 1
        left_bound_dist *= 2 * np.heaviside(left_bound_card, 1) - 1
        right_bound_dist *= 2 * np.heaviside(right_bound_card, 1) - 1
        return x_tan, y_tan, psi_tan, centerline_dist, left_bound_dist, right_bound_dist

    def get_waypoints(self, x, y, wp_dist):
        """ Get waypoints at fixed look ahead distances

        Args:
            x (float): x coord of target position
            y (float): y coord of target position
            wy_dist (np.array): array of waypoint look ahead distances

        Returns:
            wp_coords (np.array): coords of waypoints
            wp_headings (np.array): headings of waypoints
        """
        centerline_spline = self.centerline.cubic_spline
        centerline_heading = self.centerline.spline_heading
        centerline_cumdist = self.centerline.spline_cumdist
        d = dist_two_points(x, y, centerline_spline[:, 0], centerline_spline[:, 1])
        centerline_spline = centerline_spline[np.argsort(d)[0]:]
        centerline_heading = centerline_heading[np.argsort(d)[0]:]
        centerline_cumdist = centerline_cumdist[np.argsort(d)[0]:]
        centerline_cumdist -= centerline_cumdist[0]

        # get waypoint ids
        wp_id = [np.where(centerline_cumdist >= d)[0][:1] for d in wp_dist]
        wp_id = np.array([i[0] for i in wp_id if len(i) > 0]).astype(int)
        if len(wp_id) < len(wp_dist):
            wp_id = np.hstack([wp_id, (len(centerline_cumdist) - 1) * np.ones(len(wp_dist) - len(wp_id))]).astype(int)
        
        wp_coords = centerline_spline[wp_id]
        wp_headings = centerline_heading[wp_id]
        return wp_coords, wp_headings