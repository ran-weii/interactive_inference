import pyproj
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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
        relation_dict[relation_id] = {i: way_dict[i] for i in way_ids}
    return relation_dict

def find_all_lanes(way_dict):
    # find all way types
    way_types = []
    for way_id, way_val in way_dict.items():
        way_types.append(
            "_".join([way_val["type"], str(way_val["subtype"])])
        )
    way_types = np.unique(way_types)
    
    def is_connected(nd1, nd2):
        """ Check if two nodes are connected """
        # nd1 <-> nd2
        start_connected = all((
            nd1["x"][-1] == nd2["x"][0],
            nd1["y"][-1] == nd2["y"][0]
        ))
        # nd2 <-> nd1
        end_connected = all((
            nd1["x"][0] == nd2["x"][-1],
            nd1["x"][0] == nd2["y"][-1]
        ))
        if start_connected:
            return True
        elif end_connected:
            return True
        else:
            return False
        
    def add_all_nodes(G, way_dict, way_type):
        for i, (way_id, way_val) in enumerate(way_dict.items()):
            cur_way_type = "_".join([way_val["type"], str(way_val["subtype"])])
            if cur_way_type == way_type:
                G.add_nodes_from([(way_id, way_val)])
        return G
    
    def add_all_edges(G):
        node_list = list(G.nodes)
        for i, n1_id in enumerate(node_list):
            for j, n2_id in enumerate(node_list):
                if is_connected(G.nodes[n1_id], G.nodes[n2_id]):
                    G.add_edge(n1_id, n2_id)
                    G.add_edge(n2_id, n1_id)
        return G
            
    # find all lanes by searching reachable nodes
    lane_dict = {}
    lane_num = 0
    for way_type in way_types:
        G = nx.Graph()
        G = add_all_nodes(G, way_dict, way_type)
        G = add_all_edges(G)
        
        node_list = list(G.nodes)
        counter = 0
        while len(node_list) > 0:
            cur_nodes = list(nx.descendants(G, node_list[0]))
            cur_nodes.append(node_list[0])
            
            lane_dict[lane_num] = {
                "type": G.nodes[cur_nodes[0]]["type"],
                "subtype": G.nodes[cur_nodes[0]]["subtype"],
                "len": len(cur_nodes),
                "way_dict": {n_id: G.nodes[n_id] for n_id in cur_nodes}
            }
            
            # remove all nodes found on the lane
            for n_id in cur_nodes:
                node_list.remove(n_id)
            
            lane_num += 1
            counter += 1
            if counter > 200:
                break 
    return lane_dict

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

def plot_all_ways(point_dict, way_dict, figsize=(15, 6), show=False, pause=0):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_aspect('equal', adjustable='box')
    ax.patch.set_facecolor('lightgrey')
    set_visible_area(point_dict, ax)
    
    for i, (way_id, val) in enumerate(way_dict.items()):
        way_type = val["type"]
        way_subtype = val["subtype"]
        vis_dict = val["vis_dict"]
        
        if vis_dict is None:
            continue
        
        x = val["x"]
        y = val["y"]
        sort_id = np.argsort(x)
        
        ax.plot(x, y, "-o", markersize=2, **vis_dict)
        ax.text(x[sort_id[0]], y[sort_id[0]], way_id, size=8)
        
        if pause > 0:
            plt.pause(pause)
        
    if show:
        plt.show()
    return fig

def plot_all_relations(
    point_dict, relation_dict, figsize=(15, 6), show=False, pause=0
    ):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_aspect('equal', adjustable='box')
    ax.patch.set_facecolor('lightgrey')
    set_visible_area(point_dict, ax)
    
    for i, (relation_id, relation_val) in enumerate(relation_dict.items()):
        way_ids = []
        for j, (way_id, way_val) in enumerate(relation_val.items()):
            way_type = way_val["type"]
            way_subtype = way_val["subtype"]
            vis_dict = way_val["vis_dict"]
            
            if vis_dict is None:
                continue
            
            x = way_val["x"]
            y = way_val["y"]
            sort_id = np.argsort(x)
            
            ax.plot(x, y, "-o", markersize=2, **vis_dict)
            ax.text(x[sort_id[0]], y[sort_id[0]], relation_id, size=8)
            
            way_ids.append(way_id)
        
        if pause > 0:
            plt.pause(0.5)
            
    if show:
        plt.show()
    return fig

def plot_all_lanes(
    point_dict, lane_dict, figsize=(15, 6), show=False, pause=0, callback=None
    ):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_aspect('equal', adjustable='box')
    ax.patch.set_facecolor('lightgrey')
    set_visible_area(point_dict, ax)
    
    for i, (lane_id, lane_val) in enumerate(lane_dict.items()):
        lane_type = lane_val["type"]
        lane_subtype = lane_val["subtype"]
        vis_dict = get_way_vis(lane_type, lane_subtype)
        
        if vis_dict is None:
            continue
            
        for j, (way_id, way_val) in enumerate(lane_dict[lane_id]["way_dict"].items()):
            x = way_val["x"]
            y = way_val["y"]
            sort_id = np.argsort(x)
            
            ax.plot(x, y, "-o", markersize=2, **vis_dict)
            ax.text(x[sort_id[0]], y[sort_id[0]], way_id, size=8)
        
        if pause > 0:
            plt.pause(pause)
            
        if callback is not None:
            callback()
        
    if show:
        plt.show()
    return fig, callback
