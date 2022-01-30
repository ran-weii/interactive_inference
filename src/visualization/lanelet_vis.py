import numpy as np
import matplotlib.pyplot as plt

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
    
    for i, (way_id, way_val) in enumerate(way_dict.items()):
        way_type = way_val["type"]
        way_subtype = way_val["subtype"]
        vis_dict = get_way_vis(way_type, way_subtype)
        
        if vis_dict is None:
            continue
        
        x = np.array(way_val["x"])
        y = np.array(way_val["y"])
        sort_id = np.argsort(x)
        
        ax.plot(x, y, "-o", markersize=2, **vis_dict)
        ax.text(x[sort_id[0]], y[sort_id[0]], way_id, size=8)
        
        if pause > 0:
            plt.pause(pause)
        
    if show:
        plt.show()
    return fig, ax

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
            vis_dict = get_way_vis(way_type, way_subtype)
            
            if vis_dict is None:
                continue
            
            x = np.array(way_val["x"])
            y = np.array(way_val["y"])
            sort_id = np.argsort(x)
            
            ax.plot(x, y, "-o", markersize=2, **vis_dict)
            ax.text(x[sort_id[0]], y[sort_id[0]], relation_id, size=8)
            
            way_ids.append(way_id)
        
        if pause > 0:
            plt.pause(0.5)
            
    if show:
        plt.show()
    return fig, ax

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
            x = np.array(way_val["x"])
            y = np.array(way_val["y"])
            sort_id = np.argsort(x)
            
            ax.plot(x, y, "-o", markersize=2, **vis_dict)
            ax.text(x[sort_id[0]], y[sort_id[0]], way_id, size=8)
        
        if pause > 0:
            plt.pause(pause)
            
        if callback is not None:
            callback()
        
    if show:
        plt.show()
    return fig, ax, callback
