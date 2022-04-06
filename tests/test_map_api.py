import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.map_api.lanelet import MapData

data_path = "../interaction-dataset-master"
scenario1 = "DR_CHN_Merging_ZS"
scenario2 = "DR_DEU_Merging_MT"
scenario = scenario1

def test_map_parser():
    filepath = os.path.join(data_path, "maps", scenario + ".osm")
    map_data = MapData(cell_len=10)
    map_data.parse(filepath, verbose=True)
    fig, ax = map_data.plot(option="cells")
    
    # x, y = 1013.123, 964.365
    # ax.plot(x, y, "o")
    plt.show()

def test_match_lane():
    filepath = os.path.join(data_path, "maps", scenario + ".osm")
    map_data = MapData(cell_len=5)
    map_data.parse(filepath, verbose=False)
    
    # test lane matching
    x, y = 1093, 938 # lane 1 right
    # x, y = 1073.16, 961.68 # lane 6 left
    # x, y = 1142, 963 # lane 4 right
    # x, y = 1013.123, 964.365 # lane 6 left
    
    out = map_data.match(x, y, max_cells=10)
    print(out)
    
    # test lane query
    # for lane in map_data.lanes.values():
    #     print(lane.id_, lane.left_adjacent_lane_id, lane.right_adjacent_lane_id)

if __name__ == "__main__":
    # test_match_lane()
    test_map_parser()