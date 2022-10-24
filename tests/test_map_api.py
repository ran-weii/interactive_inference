import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.map_api.lanelet import MapReader

data_path = "../interaction-dataset-master"
scenario1 = "DR_CHN_Merging_ZS"
scenario2 = "DR_DEU_Merging_MT"
scenario = scenario1

def test_map_parser():
    filepath = os.path.join(data_path, "maps", scenario + ".osm")
    map_data = MapReader(cell_len=10)
    map_data.parse(filepath, verbose=True)
    print("frenet path", map_data.lanes[0].centerline.frenet_path.__dict__)
    fig, ax = map_data.plot(option="cells", annot=True)
    
    # x, y = 1095, 943
    # x, y = 1124.225, 957.044
    # x, y = 1091.742, 950.918
    x, y = 1047.779, 960.887
    # x, y = 1048.325, 960.852
    # x, y = 1003, 969
    # x, y = 1144, 972
    # x, y = 1105.103, 940.97
    # x, y = 1017.196, 965.270
    # x, y = 989.7095260004819, 932.8067411936204
    x, y = 1065.303, 958.918	
    ax.plot(x, y, "o")

    # x_tan, y_tan, psi_tan, centerline_dist, lp, rp = map_data.lanes[5].get_frenet_coords(x, y)
    # print(x_tan, y_tan, psi_tan, centerline_dist, lp, rp)
    # ax.plot(x_tan, y_tan, "o")
    # lane_id, psi_tan, centerline_dist, left_bound_dist, right_bound_dist, wp_coords, wp_headings = map_data.match_frenet(x, y, target_lane_id=None)
    # print(psi_tan, centerline_dist, left_bound_dist, right_bound_dist)
    # print(wp_headings)
    # lane_id, psi_tan, centerline_dist, left_bound_dist, right_bound_dist, wp_coords, wp_headings = map_data.match_frenet(1077.188, 960.923)
    # print(psi_tan, centerline_dist, left_bound_dist, right_bound_dist)
    # ax.plot(wp_coords[:, 0], wp_coords[:, 1], "o")
    
    plt.show()

def test_match_lane():
    filepath = os.path.join(data_path, "maps", scenario + ".osm")
    map_data = MapReader(cell_len=10)
    map_data.parse(filepath, verbose=False)
    
    # test lane matching
    x, y = 1095, 943 # lane 2 mid
    # x, y = 1093, 938 # lane 1 right
    # x, y = 1073.16, 961.68 # lane 6 left
    # x, y = 1142, 963 # lane 4 right
    # x, y = 1013.123, 964.365 # lane 6 left
    
    # out = map_data.match(x, y, target_lane_id=2, max_cells=10)
    # x, y = 1124.225, 957.044
    # x, y = 1091.742, 950.918
    x, y = 1047.779, 960.887
    # x, y = 1048.325, 960.852
    # out = map_data.match(x, y, target_lane_id=None, max_cells=10)
    out = map_data.match_frenet(x, y, target_lane_id=None, max_cells=10)
    print(out)
    
    # test lane query
    # for lane in map_data.lanes.values():
    #     print(lane.id_, lane.left_adjacent_lane_id, lane.right_adjacent_lane_id)

if __name__ == "__main__":
    # test_match_lane()
    test_map_parser()