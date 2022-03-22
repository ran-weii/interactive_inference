import xml.etree.ElementTree as ET
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union, nearest_points
from shapely import speedups
speedups.disable()

from src.map_api.lanelet_layers import L2Point, L2Linestring, L2Polygon, Lanelet, Lane
from src.map_api.utils import LL2XYProjector
from src.map_api.utils import parse_node, parse_way, parse_relation
from src.map_api.visualization import (set_visible_area, plot_points, 
    plot_ways, plot_lanelets, plot_lanes)

class MapData:
    """ lanelet2 parser adapted from https://github.com/findaheng/lanelet2_parser """
    def __init__(self, cell_distance=5):
        """
        Args:
            cell_distance (float, optional): distance of drivable cells. Defaults to 5.
        """
        self.cell_distance = cell_distance
        self.points = {}
        self.linestrings = {}
        self.polygons = {}
        self.lanelets = {}
        self.lanes = {} # concatenation of lanelets
        self.crosswalks = {}
        self.areas = {} # not implemented
        self.regulatory_elements = {} # note implemented
        
        # properties
        self._drivable_polygon = None
        self._cells = []
    
    @property
    def drivable_polygon(self):
        if self._drivable_polygon:
            return self._drivable_polygon
        
        lanelet_polygons = [l.polygon for l in self.lanelets.values() if l.subtype != "crosswalk"]
        self._drivable_polygon = unary_union(lanelet_polygons)
        return self._drivable_polygon
    
    @property
    def cells(self):
        if self._cells:
            return self._cells
        
        for lanelet in self.lanelets.values():
            for cell in lanelet.cells:
                self._cells.append((cell.polygon, cell.heading))
        return self._cells
    
    def match(self, x, y):
        """ Match point to map

        Args:
            x (float): target point x coordinate
            y (float): target point y coordinate

        Returns:
            lane_id (int): matched lane id, Returns None if not matched
            lanelet_id (int): matched lanelet id. Returns None if not matched
            cell_id (int): matched cell id.Returns None if not matched
            left_bound_dist (float): distance to cell left bound. Returns None if not matched
            right_bound_dist (float): distance to cell right bound. Returns None if not matched
        """
        p = Point(x, y)
        
        matched = False
        lane_id = None
        lanelet_id = None
        cell_id = None
        left_bound_dist = None
        right_bound_dist = None
        for lane_id, lane in self.lanes.items():
            if lane.polygon.contains(p):
                for lanelet in lane.lanelets:
                    lanelet_id = lanelet.id_
                    if lanelet.polygon.contains(p):
                        for cell_id, cell in enumerate(lanelet.cells):
                            if cell.polygon.contains(p):
                                left_bound_dist = p.distance(cell.left_bound)
                                right_bound_dist = p.distance(cell.right_bound)
                                return lane_id, lanelet_id, cell_id, left_bound_dist, right_bound_dist
        if not matched:
            lane_id = None
        return lane_id, lanelet_id, cell_id, left_bound_dist, right_bound_dist
    
    def plot(self, option="ways", figsize=(15, 6)):
        """
        Args:
            option (str, optional): plotting options, 
                one of ["ways", "lanelets", "cells", "lanes"]. Defaults to "ways".
            figsize (tuple, optional): figure size. Defaults to (15, 6).

        Returns:
            _type_: _description_
        """
        point_coords = np.array([(p.point.x, p.point.y) for p in self.points.values()])
        min_coords = point_coords.min(axis=0)
        max_coords = point_coords.max(axis=0)
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        set_visible_area(min_coords[0], min_coords[1], max_coords[0], max_coords[1], ax)
        
        if option == "ways":
            plot_ways(self, ax)
        elif option == "lanelets":
            plot_lanelets(self, ax, plot_cells=False, fill=True, annot=True, alpha=0.4)
        elif option == "cells":
            plot_lanelets(self, ax, plot_cells=True, fill=True, annot=True, alpha=0.4)
        elif option == "lanes":
            plot_lanelets(self, ax, plot_cells=False, fill=False, annot=False, alpha=0.4)
            plot_lanes(self, ax, annot=True, alpha=0.4)
        return fig, ax
    
    def parse(self, filepath, verbose=False):
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
        
        if verbose:
            print("found {} points, {} ways, {} lanelets, {} lanes".format(
                len(self.points), len(self.linestrings), len(self.lanelets), len(self.lanes)
            ))
        
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
            turn_direction, vehicle, pedestrian, bicycle, 
            cell_distance=self.cell_distance)
        
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
        lanelet._align_bounds()
        
        if subtype == "crosswalk":
            self.crosswalks[id_] = lanelet
        else:
            self.lanelets[id_] = lanelet
    
    def _extract_lanes(self):    
        def is_connected(lanelet1, lanelet2):
            left_bound_linestring_1 = lanelet1.left_bound.linestring
            left_bound_linestring_2 = lanelet2.left_bound.linestring
            right_bound_linestring_1 = lanelet1.right_bound.linestring
            right_bound_linestring_2 = lanelet2.right_bound.linestring
            out = left_bound_linestring_1.intersects(left_bound_linestring_2)
            out = out and right_bound_linestring_1.intersects(right_bound_linestring_2)
            return out
        
        # build lanelet graph
        G = nx.Graph()
        lanelets = list(self.lanelets.values())
        for i in range(len(lanelets) - 1):
            for j in range(1, len(lanelets)):
                node1_id, node1_val = lanelets[i].id_, lanelets[i]
                node2_id, node2_val = lanelets[j].id_, lanelets[j]
                if is_connected(node1_val, node2_val):
                        G.add_edge((node1_id, node1_val), (node2_id, node2_val))
                        G.add_edge((node2_id, node2_val), (node1_id, node1_val))
        
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