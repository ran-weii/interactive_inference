import math
import pyproj

class LL2XYProjector:
    """ From the INTERACTOIN repository """
    def __init__(self, lat_origin, lon_origin):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.zone = math.floor((lon_origin + 180.) / 6) + 1 
        self.p = pyproj.Proj(
            proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84'
        )
        [self.x_origin, self.y_origin] = self.p(lon_origin, lat_origin)

    def latlon2xy(self, lat, lon):
        [x, y] = self.p(lon, lat)
        return [x - self.x_origin, y - self.y_origin]
    
def parse_node(node):
    node_id = int(node.get("id"))
    node_lon = float(node.get("lon"))
    node_lat = float(node.get("lat"))
    
    type_tag = None
    subtype_tag = None
    ele_tag = None
    x_tag = None
    y_tag = None
    for tag in node.iter("tag"):
        key = tag.get("k")
        value = tag.get("v")
        
        if key == "type":
            type_tag = value
        elif key == "subtype":
            subtype_tag = value
        elif key == "ele":
            ele_tag = float(value)
        elif key == "x":
            x_tag = float(value)
        elif key == "y":
            y_tag = float(value)
        else:
            print(f"Unhandled tag at node {node_id} with key={key}")
    return (node_id, node_lon, node_lat, type_tag, 
        subtype_tag, ele_tag, x_tag, y_tag)

def parse_way(way):
    way_id = int(way.get("id"))
    ref_point_ids = [int(point.get('ref')) for point in way.findall('nd')]
    
    area_tag = False
    type_tag = None
    subtype_tag = None
    for tag in way.iter("tag"):
        key = tag.get("k")
        value = tag.get("v")
        
        if key == "area":
            area_tag = True if value == "yes" else False
        elif key == "type":
            type_tag = value
        elif key == "subtype":
            subtype_tag = value
        else:
            print(f"Unhandled tag at way {way_id} with key={key}")
    return (way_id, ref_point_ids, area_tag, type_tag, subtype_tag)

def parse_relation(relation):
    realtion_id = int(relation.get("id"))
    
    type_tag = None
    subtype_tag = None
    region_tag = None
    location_tag = None
    turn_direction_tag = None
    one_way_tag = True
    vehicle_tag = False
    pedestrian_tag = False
    bicycle_tag = False
    fallback_tag = False # for regulatory elements
    for tag in relation.iter("tag"):
        key = tag.get("k")
        value = tag.get("v")
        
        if key == "type":
            type_tag = value
        elif key == "subtype":
            subtype_tag = value
        elif key == "region":
            region_tag = value
        elif key == 'location':
            location_tag = value
        elif key == 'turn_direction':
            turn_direction_tag = value
        elif key == 'one_way':
            one_way_tag = True if value == 'yes' else False
        elif key == 'participant:vehicle':
            vehicle_tag = True if value == 'yes' else False
        elif key == 'participant:pedestrian':
            pedestrian_tag = True if value == 'yes' else False
        elif key == 'participant:bicycle':
            bicycle_tag = True if value == 'yes' else False
        elif key == 'fallback':
            fallback_tag = True if value == 'yes' else False
        else:
            print(f'Unhandled at relation {realtion_id} with key={key}')
    return (realtion_id, type_tag, subtype_tag, region_tag, 
        location_tag, turn_direction_tag, one_way_tag, vehicle_tag, 
        pedestrian_tag, bicycle_tag, fallback_tag)
