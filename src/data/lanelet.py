import pyproj
import math
import json
import numpy as np
import pandas as pd
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
        x_list, y_list = get_x_y_lists(way, point_dict)
        
        way_dict[way_id] = {
            "type": way_type, 
            "subtype": way_subtype,
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

def load_lanelet_df(lanelet_json_path):
    """
    Args:
        lanelet_json_path (str): labeled lanelet json file

    Returns:
        df_lanelet (pd.dataframe): dataframe with labeled ways
    """
    assert ".json" in lanelet_json_path
    
    with open(lanelet_json_path, "r") as f:
        lane_dict = json.load(f)
    
    # remove unlabeled lanes
    lane_dict = {
        k: v for (k, v) in lane_dict.items() if v["label"] is not None
    }
    
    # dict to df
    df_lanelet = []
    for i, (lane_id, lane_val) in enumerate(lane_dict.items()):
        label = lane_val["label"]
        num_ways = lane_val["len"]
        way_dict = lane_val["way_dict"]
        
        df_way = []
        for j, (way_id, way_val) in enumerate(way_dict.items()):
            way_val["way_id"] = way_id
            df_way.append(way_val)
            
        df_way = pd.DataFrame(df_way)
        df_way["lane_id"] = lane_id
        df_way["lane_label"] = [label for i in range(len(df_way))]
        df_way["num_ways"] = num_ways
        
        df_lanelet.append(df_way)
        
    df_lanelet = pd.concat(df_lanelet, axis=0).reset_index(drop=True)
    return df_lanelet