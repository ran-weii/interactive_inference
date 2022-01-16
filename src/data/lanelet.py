import xml.etree.ElementTree as xml
import pyproj
import math
import networkx as nx

class Point:
    def __init__(self):
        self.x = None
        self.y = None

class LL2XYProjector:
    def __init__(self, lat_origin, lon_origin):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.zone = math.floor((lon_origin + 180.) / 6) + 1 # works for most tiles, and for all in the dataset
        self.p = pyproj.Proj(
            proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84'
        )
        [self.x_origin, self.y_origin] = self.p(lon_origin, lat_origin)

    def latlon2xy(self, lat, lon):
        [x, y] = self.p(lon, lat)
        return [x - self.x_origin, y - self.y_origin]

def get_type(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "type":
            return tag.get("v")
    return None

def get_subtype(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "subtype":
            return tag.get("v")
    return None

def get_x_y_lists(element, point_dict):
    """ Get x and y pos of all nodes in a way

    Args:
        element (xml.Element): lanelet way elements
        point_dict (dict): dict of all nodes

    Returns:
        x_list (list): list of x coor
        y_list (list): list of y coor
    """
    x_list = list()
    y_list = list()
    for nd in element.findall("nd"):
        pt_id = int(nd.get("ref"))
        point = point_dict[pt_id]
        x_list.append(point.x)
        y_list.append(point.y)
    return x_list, y_list

def get_way_id(element):
    way_ids = []
    for member in element.findall("member"):
        if member.get("type") == "way":
            way_ids.append(member.get("ref"))
    return way_ids

def get_way_vis(way_type, way_subtype):
    """ Get visualization dict of a way

    Args:
        way_type (str): 
        way_subtype ([str): 

    Returns:
        type_dict: visualization dict of way including 
            color, linewidth, dash, and draw order
    """
    type_dict = None
    if way_type is None:
        raise RuntimeError("Linestring type must be specified")
    elif way_type == "curbstone":
        type_dict = dict(color="black", linewidth=1, zorder=10)
    elif way_type == "line_thin":
        if way_subtype == "dashed":
            type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[10, 10])
        else:
            type_dict = dict(color="white", linewidth=1, zorder=10)
    elif way_type == "line_thick":
        if way_subtype == "dashed":
            type_dict = dict(color="white", linewidth=2, zorder=10, dashes=[10, 10])
        else:
            type_dict = dict(color="white", linewidth=2, zorder=10)
    elif way_type == "pedestrian_marking":
        type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
    elif way_type == "bike_marking":
        type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
    elif way_type == "stop_line":
        type_dict = dict(color="white", linewidth=3, zorder=10)
    elif way_type == "virtual":
        type_dict = dict(color="blue", linewidth=1, zorder=10, dashes=[2, 5])
    elif way_type == "road_border":
        type_dict = dict(color="black", linewidth=1, zorder=10)
    elif way_type == "guard_rail":
        type_dict = dict(color="black", linewidth=1, zorder=10)
    elif way_type == "traffic_sign":
        pass
    else:
        pass
    return type_dict

def find_all_points(element, lat_origin=0, lon_origin=0):
    """ 
    Args:
        element (xml.Element): lanelet map elements
        lat_origin (int, optional): Defaults to 0.
        lon_origin (int, optional): Defaults to 0.
        
    Returns:
        point_dict (dict): dict of all nodes and their x and y coor
    """
    projector = LL2XYProjector(lat_origin, lon_origin)
    
    # load all points
    point_dict = dict()
    for node in element.findall("node"):
        point = Point()
        point.x, point.y = projector.latlon2xy(
            float(node.get('lat')), float(node.get('lon'))
        )
        point_dict[int(node.get('id'))] = point
    return point_dict

def find_all_ways(element, point_dict):
    """
    Args:
        element (xml.Element): lanelet map elements
        point_dict (dict): dict of all nodes
        
    Returns:
        way_dict (dict): dict of all ways and their coors and visualization dict
    """
    way_dict = {}
    for i, way in enumerate(element.findall("way")):
        way_id = way.get("id")
        way_type = get_type(way)
        way_subtype = get_subtype(way) if get_subtype(way) is not None else None
        vis_dict = get_way_vis(way_type, way_subtype)
        
        x_list, y_list = get_x_y_lists(way, point_dict)
        x_list = np.array(x_list)
        y_list = np.array(y_list)
        
        way_dict[way_id] = {
            "type": way_type, 
            "subtype": way_subtype,
            "vis_dict": vis_dict, 
            "x": x_list, 
            "y": y_list
        }
    return way_dict

def find_all_relations(element, way_dict):
    """
    Args:
        element (xml.Element): lanelet map elements
        way_dict (dict): dict of all ways
        
    Returns:
        relation_dict (dict): dict of all relations, each is a list of way_dict
    """
    relation_dict = {}
    for i, relation in enumerate(element.findall("relation")):
        relation_id = relation.get("id")
        way_ids = get_way_id(relation)
        # relation_dict[relation_id] = [way_dict[i] for i in way_ids]
        relation_dict[relation_id] = {i: way_dict[i] for i in way_ids}
    return relation_dict

def set_visible_area(point_dict, axes):
    min_x = 10e9
    min_y = 10e9
    max_x = -10e9
    max_y = -10e9

    for key, point in point_dict.items():
        min_x = min(point.x, min_x)
        min_y = min(point.y, min_y)
        max_x = max(point.x, max_x)
        max_y = max(point.y, max_y)

    axes.set_aspect('equal', adjustable='box')
    axes.set_xlim([min_x - 10, max_x + 10])
    axes.set_ylim([min_y - 10, max_y + 10])
    
if __name__ == "__main__":
    """ TODO 
    (done) render each line by pause to figure out how this map is produced
    (done) lable each line center with way_id
    (done) understand how relations are formed by plotting each relation in order
        is it only two parallel lines? 
            left and right don't mean left and right, they are up and down here and do not even follow the same order
            the relations don't include all ways, some ways don't have relations
            ways may appear in multiple relations
    (done) can we ignore guard rails and are there any lines underneath the guard rails?
        there is no lines underneath the guard rails. as the map is defined by top layer nodes, every connection between two nodes uniquely appear once
    (almost) how to identify a continuing lane? 
        for each way segment we find all segments that share the same nodes, and take their union
        plot a map with all nodes labeled, find segment not yet connected on lane map and verify their end nodes
    is it possible to identify and mask lane segments in an image? if so lane assignment may be done with the amount of intersection area
    how to identify 2 lanes merge into one and space between guard rails that are not lanes? 
        we can potentially render the relations one by one and label them, the result should be a segmentation map with hand labels
        for agent observation, lane change can be stored as tuple (direction, ahead distance)
    how to define lane curvature or angle? maybe for highway we just consider the road as straight?
    """
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    data_path = "../../interaction-dataset-master/maps"
    file_path = os.path.join(data_path, "DR_CHN_Merging_ZS.osm")
    # file_path = os.path.join(data_path, "DR_DEU_Merging_MT.osm")
    
    e = xml.parse(file_path).getroot()
    
    point_dict = find_all_points(e, lat_origin=0, lon_origin=0)
    way_dict = find_all_ways(e, point_dict)
    relation_dict = find_all_relations(e, way_dict)
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    ax.set_aspect('equal', adjustable='box')
    ax.patch.set_facecolor('lightgrey')
    set_visible_area(point_dict, ax)
    
    # plot all points
    # x_list, y_list = [], []
    # for k, v in point_dict.items():
    #     x_list.append(v.x)
    #     y_list.append(v.y)
    
    # ax.plot(x_list, y_list, "o", markersize=4)
    # plt.show()
    
    
    # plot all ways
    # for i, (way_id, val) in enumerate(way_dict.items()):
    #     way_type = val["type"]
    #     way_subtype = val["subtype"]
    #     vis_dict = val["vis_dict"]
        
    #     if vis_dict is None:
    #         continue
        
    #     x = val["x"]
    #     y = val["y"]
    #     sort_id = np.argsort(x)
        
    #     ax.plot(x, y, **vis_dict)
    #     ax.text(x[sort_id[0]], y[sort_id[0]], way_id, size=8)
    #     # plt.pause(0.5)
        
    #     print(i, way_id, way_type, way_subtype)
    #     print("x", x[sort_id])
    #     print("y", y[sort_id])
    
    # plot all relations
    # for i, (relation_id, relation_val) in enumerate(relation_dict.items()):
    #     way_ids = []
    #     for j, (way_id, way_val) in enumerate(relation_val.items()):
    #         way_type = way_val["type"]
    #         way_subtype = way_val["subtype"]
    #         vis_dict = way_val["vis_dict"]
            
    #         if vis_dict is None:
    #             continue
            
    #         x = way_val["x"]
    #         y = way_val["y"]
    #         sort_id = np.argsort(x)
            
    #         ax.plot(x, y, "-o", markersize=2, **vis_dict)
    #         ax.text(x[sort_id[0]], y[sort_id[0]], relation_id, size=8)
            
    #         way_ids.append(way_id)
        
    #     print(i, relation_id, way_ids)
    #     plt.pause(0.5)
        
    # plt.show()