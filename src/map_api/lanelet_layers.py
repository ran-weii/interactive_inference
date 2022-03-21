import numpy as np
from shapely.geometry import Point, LineString, Polygon

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
    
    def add_reference(self, lanelet_id):
        self.lanelet_references.append(lanelet_id)

     
class L2Polygon:
    def __init__(self, id_, polygon, type_, subtype):
        self.id_ = id_
        self.polygon = polygon
        self.type_ = type_
        self.subtype = subtype
            
            
class Lanelet:
    class Cell:
        ''' Section of a lane, represented as a Shapely polygon, with a defined heading '''
        def __init__(self, polygon, heading):
            self.polygon = polygon  # Shapely polygon
            self.heading = heading  # radians clockwise from y-axis
            
    def __init__(self, id_, subtype, region, location, one_way, 
        turn_direction, vehicle_participant, pedestrian_participant, 
        bicycle_participant, left_bound=None, right_bound=None, 
        centerline=None, regulatory_elements=[], cell_distance=5., buffer_=0):
        self.id_ = id_
        self.subtype = subtype
        self.region = region
        self.location = location
        self.one_way = one_way
        self.turn_direction = turn_direction
        self.pedestrian_participant = pedestrian_participant
        self.regulatory_elements = regulatory_elements
        self.cell_distance = cell_distance
        self.buffer_ = buffer_
        self.bicycle_participant = bicycle_participant

        # L2Linestring
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.centerline = centerline

        # both left and right linestrings must point in the same direction
        # used to handle inversion of linestring points order
        self._flip_left = False
        self._flip_right = False

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
        if self._cells:
            return self._cells

        # reverse left bound if opposed
        right_bound_linestr = self.right_bound.linestring
        left_bound_linestr = self.left_bound.linestring
        if self.has_opposing_linestrings():
            left_bound_linestr = LineString(self.left_bound.linestring.coords[::-1]) 
        
        # determine which linestring is longer
        if right_bound_linestr.length > left_bound_linestr.length:
            longer_linestr = right_bound_linestr
            shorter_linestr = left_bound_linestr
        else:
            longer_linestr = left_bound_linestr
            shorter_linestr = right_bound_linestr
        
        # interpolate shorter linestring by cell distance
        shorter_distances = np.arange(0, shorter_linestr.length, self.cell_distance)
        shorter_points = [shorter_linestr.interpolate(d) for d in shorter_distances] + [shorter_linestr.boundary[1]]
        longer_distances = [0] + [longer_linestr.project(p) for p in shorter_points[1:-1]]
        longer_points = [longer_linestr.interpolate(d) for d in longer_distances] + [longer_linestr.boundary[1]]
        for i in range(len(shorter_points) - 1):
            cell_coords = [(p.x, p.y) for p in [shorter_points[i], shorter_points[i+1], longer_points[i+1], longer_points[i]]]
            cell_polygon = Polygon(cell_coords).buffer(self.buffer_)
            
            # compute cell heading
            delta_y = shorter_points[i+1].y - shorter_points[i].y
            delta_x = shorter_points[i+1].x - shorter_points[i].x
            cell_heading = np.arctan2(delta_y, delta_x)
            
            cell = self.Cell(cell_polygon, cell_heading)
            self._cells.append(cell)    
        return self._cells


class Lane:
    def __init__(self, id_, lanelets, left_bound=None, right_bound=None):
        self.id = id_
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.lanelets = {v.id_: v for v in lanelets}