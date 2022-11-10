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
    
    fig, ax = map_data.plot(option="cells", annot=True) 
    x, y = 1065.303, 958.918	
    ax.plot(x, y, "o")

    plt.show()

def test_match_lane():
    filepath = os.path.join(data_path, "maps", scenario + ".osm")
    map_data = MapReader(cell_len=10)
    map_data.parse(filepath, verbose=False)
    
    # test lane matching
    x, y = 1047.779, 960.887
    out = map_data.match_frenet(x, y, target_lane_id=None, max_cells=10)
    print(out)

if __name__ == "__main__":
    test_map_parser()
    # test_match_lane()