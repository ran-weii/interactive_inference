import xml.etree.ElementTree as ET
import networkx as nx

from shapely.geometry import Point, LineString, Polygon
from shapely import speedups
speedups.disable()

from src.map_api.lanelet_layers import L2Point, L2Linestring, L2Polygon, Lanelet, Lane
from src.map_api.utils import LL2XYProjector
from src.map_api.utils import parse_node, parse_way, parse_relation

class MapData:
    """ lanelet2 parser adapted from https://github.com/findaheng/lanelet2_parser """
    def __init__(self):
        self.points = {}
        self.linestrings = {}
        self.polygons = {}
        self.lanelets = {}
        self.lanes = {} # concatenation of lanelets
        self.crosswalks = {}
        self.areas = {}
        self.regulatory_elements = {}
        
    def parse(self, filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()
        geo_projector = LL2XYProjector(0, 0)
        
        assert root.tag == "osm", f"{filepath} does not appear to be an OSM-XML file"
        
        for node in root.iter("node"):
            (id_, lon, lat, type_, subtype, ele, x, y) = parse_node(node)
            self._extract_point(id_, lon, lat, type_, subtype, ele, x, y, geo_projector)
            
        for way in root.iter("way"):
            (id_, ref_point_ids, area_tag, type_, subtype) = parse_way(way)
            if area_tag:
                self._extract_polygon(id_, ref_point_ids, type_, subtype)
            else:
                self._extract_linestring(id_, ref_point_ids, type_, subtype)
        
        for relation in root.iter("relation"):
            (id_, type_tag, subtype, region, 
            location, turn_direction, one_way, vehicle, 
            pedestrian, bicycle, fallback) = parse_relation(relation)
            if type_tag == "lanelet":
                self._extract_lanelet(id_, subtype, region, location, one_way, 
                    turn_direction, vehicle, pedestrian, bicycle, relation)
        
        self._extract_lanes()
        
    def _extract_lanes(self):    
        def is_connected(lanelet1, lanelet2):
            left_bound_linestring_1 = lanelet1.left_bound.linestring
            left_bound_linestring_2 = lanelet2.left_bound.linestring
            right_bound_linestring_1 = lanelet1.right_bound.linestring
            right_bound_linestring_2 = lanelet2.right_bound.linestring
            out = left_bound_linestring_1.intersects(left_bound_linestring_2)
            out = out and right_bound_linestring_1.intersects(right_bound_linestring_2)
            return out
        
        def add_all_edges(G):
            node_list = list(G.nodes)
            for i, (node1_id, node1_val) in enumerate(node_list):
                for j, (node2_id, node2_val) in enumerate(node_list):
                    if is_connected(node1_val, node2_val):
                        G.add_edge(node_list[i], node_list[j])
                        G.add_edge(node_list[j], node_list[i])
            return G
        
        G = nx.Graph()
        
        # add all lanelets as nodes
        for lanelet_id, lanelet in self.lanelets.items():
            G.add_nodes_from([(lanelet_id, lanelet)])
        
        # add connected lanelets as edges     
        G = add_all_edges(G)
        
        # add reachable lanelets as lanes
        node_list = list(G.nodes)
        counter = 0
        while len(node_list) > 0:
            curr_node = node_list[0]
            connected_nodes = list(nx.descendants(G, curr_node))
            connected_nodes.append(curr_node)
            
            lanelets = [n[1] for n in connected_nodes]
            self.lanes[counter] = Lane(counter, lanelets)
            
            # remove all nodes found on the lane
            for n_id in connected_nodes:
                node_list.remove(n_id)
            counter += 1        
        
        print(f"found {counter} lanes")
        
    def _extract_point(self, id_, lon, lat, type_, subtype, ele, x, y, geo_projector):
        x, y = geo_projector.latlon2xy(lat, lon)
        
        geo_point = Point(lon, lat)
        metric_point = Point(x, y)
        self.points[id_] = L2Point(id_, metric_point, geo_point, type_, subtype)
        
    def _extract_linestring(self, id_, ref_point_ids, type_, subtype):
        ref_points = [self.points[i] for i in ref_point_ids]
        ref_point_coords = [(p.point.x, p.point.y) for p in ref_points]
        linestring = LineString(ref_point_coords)
        self.linestrings[id_] = L2Linestring(id_, linestring, type_, subtype)
    
    def _extract_polygon(self, id_, ref_point_ids, type_, subtype):
        ref_points = [self.points[i] for i in ref_point_ids]
        ref_point_coords = [(p.point.x, p.point.y) for p in ref_points]
        polygon = Polygon(ref_point_coords)
        self.polygons[id_] = L2Polygon(id_, polygon, type_, subtype)
        
    def _extract_lanelet(self, id_, subtype, region, location, one_way, 
            turn_direction, vehicle, pedestrian, bicycle, relation):
        lanelet = Lanelet(id_, subtype, region, location, one_way, 
            turn_direction, vehicle, pedestrian, bicycle)
        
        for member in relation.iter("member"):
            member_role = member.get("role")
            ref_id = int(member.get("ref"))
            
            # regulatory element handle
            if member_role == "regulatory_element":
                continue
            
            linestring = self.linestrings[ref_id]
            linestring.add_reference(id_)
            if member_role == "left":
                lanelet.left_bound = linestring
            elif member_role == "right":
                lanelet.right_bound = linestring
            elif member_role == "centerline":
                lanelet.centerline = linestring
            else:
                raise ValueError(f"Unknown member role {member_role} in lanelet with id={id_}")
        
        assert lanelet.left_bound and lanelet.right_bound, f"Lanelet with id={id_} missing bound(s)"
        if subtype == "crosswalk":
            self.crosswalks[id_] = lanelet
        else:
            self.lanelets[id_] = lanelet
    
    def _align_lanelets(self, align_range=0):
        counter = 0
        lanelets = [v for v in self.lanelets.values()]
        for i in range(len(lanelets) - 1):
            curr = lanelets[i]
            curr_left = list(curr.left_bound.linestring.coords)
            curr_right = list(curr.right_bound.linestring.coords)
            curr_bound_pts = tuple(Point(coords) for coords in (curr_left[0], curr_left[-1], curr_right[0], curr_right[-1]))
            
            # print(curr_bound_pts)
            for j in range(i + 1, len(lanelets)):
                other = lanelets[j]
                other_left = list(other.left_bound.linestring.coords)
                other_right = list(other.right_bound.linestring.coords)
                other_bound_pts = tuple(Point(coords) for coords in (other_left[0], other_left[-1], other_right[0], other_right[-1]))
                
                for u in range(len(curr_bound_pts)):
                    curr_pt = curr_bound_pts[u]
                    
                    for k in range(len(other_bound_pts)):
                        other_pt = other_bound_pts[k]
                        dist = curr_pt.distance(other_pt)
                        
                        if dist <= align_range:
                            counter += 1
                            
                            avg_x = (curr_pt.x + other_pt.x) / 2
                            avg_y = (curr_pt.y + other_pt.y) / 2
                            
                            if u == 0 or u == 1: # replace left bound coordinates
                                new_coords = curr_left
                                new_coords[0 if u ==0 else -1] = (avg_x, avg_y)
                                curr.right_bound.linestring = LineString(new_coords)
                            else: # replace right bound coordinates
                                new_coords = curr_right
                                new_coords[0 if u == 2 else -1] = (avg_x, avg_y)
                                curr.right_bound.linestring = LineString(new_coords)
                            
                            if k == 0 or k == 1: # replace left bound coordinates
                                new_coords = other_left
                                new_coords[0 if k == 0 else -1] = (avg_x, avg_y)
                                other.left_bound.linestring = LineString(new_coords)
                            else: # replace right bound coordinates
                                new_coords = other_right
                                new_coords[0 if k == 2 else -1] = (avg_x, avg_y)
                                other.right_bound.linestring = LineString(new_coords)
        print(f"{counter} points aligned")