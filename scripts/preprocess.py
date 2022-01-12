import argparse
import os
import glob
import numpy as np
import pandas as pd

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../interaction-dataset-master"
    )
    parser.add_argument("--scenario", type=str, default="Merging")
    parser.add_argument("--debug", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def main(arglist):
    map_path = os.path.join(arglist.data_path, "maps")
    scenario_paths = glob.glob(
        os.path.join(arglist.data_path, "recorded_trackfiles", "*/")
    )
    
    scenario_counter = 0
    for i, scenario_path in enumerate(scenario_paths):
        scenario = os.path.basename(os.path.dirname(scenario_path))
        if arglist.scenario in scenario:
            print(f"Scenario {i+1}: {scenario}")
            
            track_paths = glob.glob(
                os.path.join(scenario_path, "*.csv")
            )
            
            for j, track_path in enumerate(track_paths):
                df_track = pd.read_csv(track_path)
                print(df_track)
            
                if arglist.debug:
                    break
            
            scenario_counter += 1
        else:
            pass 
        
        if arglist.debug and scenario_counter > 0:
            break
    
    return 

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)