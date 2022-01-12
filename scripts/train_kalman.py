import argparse
import os
import glob
import numpy as np
import pandas as pd

""" TODO
write tests:
    get raw pos and vel statistics, are they too extreme
    find the relationship between yaw, pos, and vel
    derive raw acc from vel
    interpolate pos, vel, acc, raw to access accuracy
check how original repo handle tracks to render
train kalman filter
"""
def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../interaction-dataset-master"
    )
    parser.add_argument("--scenario", type=str, default="Merging")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def main(arglist):
    # load data
    
    # subsample tracks
    
    # derive acceleration
    
    # init parameters
    
    # train
    return 

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)