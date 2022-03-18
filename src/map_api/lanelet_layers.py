import math
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
        centerline=None, regulatory_elements=[], buffer_=0):
        self.id_ = id_
        self.subtype = subtype
        self.region = region
        self.location = location
        self.one_way = one_way
        self.turn_direction = turn_direction
        self.pedestrian_participant = pedestrian_participant
        self.regulatory_elements = regulatory_elements
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
        left_bound_linestr = self.left_bound.linestring
        if self.has_opposing_linestrings():
            left_bound_linestr = LineString(self.left_bound.linestring.coords[::-1]) 
        
        # determine linestring with more points		
        num_right_pts = len(self.right_bound.linestring.coords)  
        num_left_pts = len(self.left_bound.linestring.coords) 
        right_has_more = False
        if num_right_pts > num_left_pts:
            right_has_more = True
            more_pts_linestr = self.right_bound.linestring
            less_pts_linestr = left_bound_linestr 
        else:
            more_pts_linestr = left_bound_linestr
            less_pts_linestr = self.right_bound.linestring
        
        less_first_pt = Point(less_pts_linestr.coords[0][0], less_pts_linestr.coords[0][1])  
        less_second_pt = Point(less_pts_linestr.coords[1][0], less_pts_linestr.coords[1][1])  
        less_last_pt = Point(less_pts_linestr.coords[-1][0], less_pts_linestr.coords[-1][1])  
        
        # connect points from linestring (with more points) to other linestring (that has less points)
        more_pts_coords = more_pts_linestr.coords
        for i in range(len(more_pts_coords) - 1):
            curr_pt = Point(more_pts_coords[i][0], more_pts_coords[i][1])  # convert to Shapely point
            next_pt = Point(more_pts_coords[i + 1][0], more_pts_coords[i + 1][1])  # to compute second bound and heading
            
            # compute closest point on other linestring
            # endpoints guarantee other point is a coordinate of linestring
            # middle points project to points that are not necessarily coordiantes of linestring
            if i == 0:
                bound_pt_1 = less_pts_linestr.interpolate(less_pts_linestr.project(next_pt))
                bound_pt_2 = less_first_pt if next_pt.distance(less_first_pt) < next_pt.distance(less_last_pt) else less_last_pt
            elif i == (len(more_pts_coords) - 1):
                bound_pt_1 = less_first_pt if next_pt.distance(less_first_pt) < next_pt.distance(less_last_pt) else less_last_pt
                bound_pt_2 = less_pts_linestr.interpolate(less_pts_linestr.project(curr_pt)) 
            else:
                bound_pt_1 = less_pts_linestr.interpolate(less_pts_linestr.project(next_pt))
                bound_pt_2 = less_pts_linestr.interpolate(less_pts_linestr.project(curr_pt))
            
            cell_coords = [(p.x, p.y) for p in [curr_pt, next_pt, bound_pt_1, bound_pt_2]]
            
            cell_polygon = Polygon(cell_coords).buffer(self.buffer_)

            # (assuming) can define heading based on lanelet's right bound
            delta_x = next_pt.x - curr_pt.x if right_has_more else less_second_pt.x - less_first_pt.x
            delta_y = next_pt.y - curr_pt.y if right_has_more else less_second_pt.y - less_first_pt.y
            cell_heading = math.atan(delta_y / delta_x) + math.pi / 2 if delta_x else 0 # since headings in radians clockwise from y-axis

            cell = self.Cell(cell_polygon, cell_heading)
            self._cells.append(cell)
            
        return self._cells


class Lane:
    def __init__(self, id_, lanelets, left_bound=None, right_bound=None):
        self.id = id_
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.lanelets = {v.id_: v for v in lanelets}