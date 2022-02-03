import os
os.environ["MODIN_CPUS"] = "8"

import time
import numpy as np
import pandas as pd
import swifter
from tqdm import tqdm
import modin
import modin.pandas as mpd

modin.config.ProgressBar.enable()
print(f"modin cpu count: {modin.config.CpuCount.get()}")
print(f"modin progress bar: {modin.config.ProgressBar.get()}")

import ray
ray.init()

def test_pandas_vs_modin():
    data_path = "../interaction-dataset-master/recorded_trackfiles"
    scenario = "DR_CHN_Merging_ZS"
    filename = "vehicle_tracks_007.csv"
    file_path = os.path.join(data_path, scenario, filename)
    assert os.path.exists(file_path)
    
    # test loading
    start_pandas  = time.time()
    df_pandas = pd.read_csv(file_path)
    pandas_time = time.time() - start_pandas
    
    start_modin = time.time()
    df_modin = mpd.read_csv(file_path)
    modin_time = time.time() - start_modin
    print(f"loading time - pandas: {pandas_time}, modin: {modin_time}")
    
    # test apply
    custom_f = lambda x: x["x"] ** 2

    start_pandas = time.time()
    df_pandas.apply(custom_f, axis=1)
    pandas_time = time.time() - start_pandas
    
    start_swifter = time.time()
    df_pandas.swifter.apply(custom_f, axis=1)
    swifter_time = time.time() - start_swifter
    
    start_modin = time.time()
    df_modin.apply(custom_f, axis=1)
    modin_time = time.time() - start_modin
    modin_time = None
    print(f"apply time - pandas: {pandas_time}, swifter:{swifter_time}, modin: {modin_time}")

if __name__ == "__main__":
    test_pandas_vs_modin()
    ray.shutdown()