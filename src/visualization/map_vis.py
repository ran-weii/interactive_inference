import numpy as np

def get_way_styling(way_type, way_subtype):
    """ Get styling dict of a way

    Args:
        way_type (str): 
        way_subtype ([str): 

    Returns:
        type_dict: styling dict of way including 
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

def set_visible_area(map_data, ax):
    """ Set canvas to map size """
    point_coords = np.array([(p.point.x, p.point.y) for p in map_data.points.values()])
    min_coords = point_coords.min(axis=0)
    max_coords = point_coords.max(axis=0)
    
    ax.patch.set_facecolor('lightgrey')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([min_coords[0] - 10, max_coords[0] + 10])
    ax.set_ylim([min_coords[1] - 10, max_coords[1] + 10])
    
def plot_points(map_data, ax, annot=False):
    set_visible_area(map_data, ax)
    for point in map_data.points.values():
        p = point.point
        ax.plot(p.x, p.y, "ko")
        if annot:
            ax.text(
                p.x, p.y, point.id_, size=8, 
                horizontalalignment="center", 
                verticalalignment="center"
            )
    return ax

def plot_ways(map_data, ax, annot=True):
    set_visible_area(map_data, ax)
    marker = "-o" if annot else "-"
    for (way_id, way) in map_data.linestrings.items():
        coords = list(way.linestring.coords)[0]
        style_dict = get_way_styling(way.type_, way.subtype)
        ax.plot(*way.linestring.xy, marker, markersize=2, **style_dict)
        if annot:
            ax.text(
                coords[0], coords[1], 
                way_id, size=8, 
                horizontalalignment="left", 
                verticalalignment="center"
            )
    return ax

def plot_lanelets(map_data, ax, plot_cells=False, fill=True, annot=False, alpha=0.4):
    set_visible_area(map_data, ax)
    for (lanelet_id, lanelet) in map_data.lanelets.items():
        polygon = lanelet.polygon
        centroid_coords = list(polygon.centroid.coords)[0]
        ax.plot(*polygon.exterior.xy, "k-o", markersize=4)
        
        if fill:
            if not plot_cells:
                ax.fill(*polygon.exterior.xy, alpha=alpha)
            else:
                for cell in lanelet.cells:
                    cell_polygon = cell.polygon
                    ax.fill(*cell_polygon.exterior.xy, alpha=alpha)
            
        if annot:
            ax.text(
                centroid_coords[0], 
                centroid_coords[1], 
                lanelet_id, size=8, 
                horizontalalignment="center", 
                verticalalignment="center"
            )
    return ax

def plot_lanes(map_data, ax, plot_cells=False, annot=True, alpha=0.4):
    set_visible_area(map_data, ax)
    for lane_id, lane in map_data.lanes.items():
        polygon = lane.polygon
        left_bound_linestr = lane.left_bound.linestring
        right_bound_linestr = lane.right_bound.linestring
        centerline_linestr = lane.centerline.linestring
        ax.plot(*left_bound_linestr.xy, "k-")
        ax.plot(*right_bound_linestr.xy, "k-")
        ax.plot(*centerline_linestr.xy, "g--")
        
        if not plot_cells:
            ax.fill(*polygon.exterior.xy, alpha=alpha)
        else:
            for cell in lane.cells:
                cell_polygon = cell.polygon
                ax.fill(*cell_polygon.exterior.xy, alpha=alpha)
        if annot:
            for cell_id, cell in enumerate(lane.cells):
                centroid_coords = list(cell.polygon.centroid.coords)[0]
                ax.text(
                    centroid_coords[0], 
                    centroid_coords[1], 
                    cell_id, size=8, 
                    color="g"
                )
                if cell_id == 0:
                    center_line_coords = cell.center_line.coords[0]
                    ax.text(
                    center_line_coords[0], 
                    center_line_coords[1], 
                    lane_id, size=8, 
                )
    return ax
