import argparse
import os
import json
import xml.etree.ElementTree as xml
import matplotlib.pyplot as plt
from src.data.lanelet import find_all_points, find_all_ways, find_all_lanes    
from src.visualization.lanelet_vis import plot_all_ways, plot_all_lanes

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../interaction-dataset-master"
    )
    parser.add_argument(
        "--save_path", type=str, default="../exp/lanelet"
    )
    parser.add_argument("--scenario", type=str, default="DR_CHN_Merging_ZS")
    parser.add_argument("--label", type=bool_, default=True)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

class LaneLabler():
    """ Lane labeling callback to be used with plot_all_lanes """
    def __init__(self):
        self.t = 0
        self.labels = []
    
    def __call__(self):
        """ Interactive labeling function to be called in callback 
            
            Follow the lane labeling rules in root/doc/lanelet.md
        """
        x = input(
            "Please assign an integer label to the current lane."
            " If it is a merging lane, please enter all of its lane number,"
            " separated by a comma (,): "
        )
        try:
            x = [int(i) for i in x.split(",")]
        except ValueError:
            x = None
            
        self.t += 1
        self.labels.append(x)
        
if __name__ == "__main__":
    arglist = parse_args()
    file_path = os.path.join(
        arglist.data_path, 
        "maps", 
        arglist.scenario + ".osm"
    )
    
    e = xml.parse(file_path).getroot()
    
    point_dict = find_all_points(e, lat_origin=0, lon_origin=0)
    way_dict = find_all_ways(e, point_dict)
    lane_dict = find_all_lanes(way_dict)
    
    if arglist.label:
        labler = LaneLabler()
    else:
        labler = None
    
    with plt.ion():    
        plot_all_ways(point_dict, way_dict, show=False, pause=0)
        plot_all_lanes(
            point_dict, lane_dict, show=False, pause=0.1, callback=labler
        )
    
    # update lane_dict
    if arglist.label:
        for i, (lane_id, lane_val) in enumerate(lane_dict.items()):
            lane_dict[lane_id]["label"] = labler.labels[i]
        
        if arglist.save:
            if not os.path.exists(arglist.save_path):
                os.mkdir(arglist.save_path)
            
            file_name = os.path.join(arglist.save_path, arglist.scenario + ".json")
            with open(file_name, "w") as f:
                json.dump(lane_dict, f)
            
            print("map saved")
    
    plt.show()