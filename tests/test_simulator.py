import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.train_utils import load_data
from src.map_api.lanelet import MapReader
from src.data.ego_dataset import EgoDataset, RelativeDataset
from src.simulation.simulator import InteractionSimulator
from src.visualization.animation import animate, save_animation

from src.simulation.observers import FEATURE_SET
from src.simulation.observers import Observer
# from src.simulation.controllers import AgentWrapper
from src.map_api.frenet import FrenetPath
from src.map_api.frenet_utils import compute_normal_from_kappa
from src.map_api.frenet_utils import compute_acceleration_vector
from src.data.geometry import angle_to_vector, wrap_angles

import warnings
warnings.filterwarnings("ignore")

lanelet_path = "../exp/lanelet"
data_path = "../interaction-dataset-master"
scenario1 = "DR_CHN_Merging_ZS"
scenario2 = "DR_DEU_Merging_MT"
scenario = scenario1
filename = "vehicle_tracks_007.csv"

# load map
map_data = MapReader(cell_len=10)
map_data.parse(os.path.join(data_path, "maps", scenario + ".osm"))

def test_simulator_from_data():
    """ Use preprocessed frenet actions to control the simulator """
    df_track = load_data(data_path, scenario, filename)
    ego_dataset = EgoDataset(df_track)
    observer = Observer(map_data)
    env = InteractionSimulator(ego_dataset, map_data, observer)
    
    obs = env.reset(0)
    
    eps_id = env._track_data["meta"][1]
    df_eps = df_track.loc[df_track["eps_id"] == eps_id].reset_index(drop=True)
    
    rewards = []
    for t in range(env.T):
        # get agent frenet acceleration
        dds = df_eps["dds"].values[t]
        ddd = df_eps["ddd"].values[t]
        ctl = torch.tensor([dds, ddd]).view(1, -1)
        
        obs, r, done, info = env.step(ctl)
        rewards.append(r)
        
        if done:
            break
    print(rewards)
    print(np.mean(rewards))
    exit()
    ani = animate(map_data, env._sim_states, env._track_data, title="test", annot=True)
    save_animation(ani, "/Users/rw422/Documents/render_ani.mp4")

def test_simulator_with_observer():
    """ Test simulator from user defined accelerations """
    df_track = load_data(data_path, scenario, filename)
    ego_dataset = EgoDataset(df_track)
    
    env = InteractionSimulator(ego_dataset, map_data)
    
    # frenet actions
    act = np.array([0, 0])

    class Agent:
        def __init__(self):
            self._b = None
        
        def reset(self):
            pass 

        def eval(self):
            pass 

        def choose_action(self, *kargs, **kwargs):
            return torch.from_numpy(act)

    agent = Agent()
    observer = Observer(map_data)
    controller = AgentWrapper(observer, agent, ["dds", "ddd"], "ace")
    controller.reset()

    obs_env = env.reset(1)
    for t in range(env.T):
        # get true agent control
        # ctl_env = env.get_action()
        ctl_env = controller.choose_action(obs_env)
        
        obs_env, r, done, info = env.step(ctl_env)
        
        if done:
            break

    ani = animate(map_data, env._sim_states, env._track_data, title="test", annot=False)
    save_animation(ani, "/Users/rw422/Documents/render_ani.mp4")
    print("test_simulator passed")

def test_observer():
    observer = Observer(
        map_data, ego_features=FEATURE_SET["ego"],
        relative_features=FEATURE_SET["relative"]
    )
    
    # create synthetic data
    # ego_state = np.array([1065.303, 958.918, -10, -0.01, np.pi, 0])
    ego_state = np.array([1091.742, 950.918, -10, -0.01, np.pi, 0])
    agent_state = np.array([[1060.303, 960.918, -10, -0.01, np.pi]])
    state = {"ego": ego_state, "agents": agent_state}
    obs = observer.observe(state)
    assert list(obs.shape) == [1, len(observer.feature_set)]
    print("test_pbserver_new passed")

def test_data_wrapper():
    from src.simulation.controllers import DataWrapper
    df_track = load_data(data_path, scenario, filename)
    ego_dataset = EgoDataset(df_track)

    ego_features = ["d", "ds", "dd", "kappa_r", "psi_error_r", ]
    relative_features = ["s_rel", "d_rel", "ds_rel", "dd_rel", "loom_s"]
    feature_set = ego_features + relative_features
    action_set = ["dds", "ddd"]
    # action_set = ["ax_ego", "ay_ego"]
    rel_dataset = RelativeDataset(df_track, feature_set, action_set, max_eps_len=1000)
    
    env = InteractionSimulator(ego_dataset, map_data)
    
    # dummy agent
    class Agent:
        def __init__(self):
            self._b = None
        
        def reset(self):
            pass 

        def eval(self):
            pass 

        def choose_action(self, *kargs, **kwargs):
            return torch.from_numpy(np.array([0, 0]))
    
    eps_id = 2
    observer = Observer(map_data)
    agent = Agent()
    eps_data = rel_dataset[eps_id]
    controller = DataWrapper(eps_data, observer, agent, action_set, "ace")
    controller.reset()

    obs_env = env.reset(eps_id)
    for t in range(env.T-1):
        # get true agent control
        # ctl_env = env.get_action()
        ctl_env = controller.choose_action(obs_env)
        
        obs_env, r, done, info = env.step(ctl_env)
        # exit()
        
        if done:
            break

    ani = animate(map_data, env._sim_states, env._track_data, title="test", annot=False)
    save_animation(ani, "/Users/rw422/Documents/render_ani.mp4")

    print("test_data_wrapper passed")

def test_idm_agent():
    torch.manual_seed(0)
    from src.agents.rule_based import IDM

    df_track = load_data(data_path, scenario, filename, train=True)

    ego_dataset = EgoDataset(df_track, train_labels_col="is_train", max_eps=100, create_svt=False, seed=0)
    
    ego_features = ["d", "ds", "dd", "kappa_r", "psi_error_r", ]
    relative_features = ["s_rel", "d_rel", "ds_rel", "dd_rel", "loom_s"]
    feature_set = ego_features + relative_features
    action_set = ["dds", "ddd"]
    observer = Observer(map_data, ego_features=ego_features, relative_features=relative_features)

    env = InteractionSimulator(ego_dataset, map_data, observer)

    ctl_std = torch.from_numpy(df_track.loc[df_track["is_train"] == 1][action_set].var().values).view(1, 2).to(torch.float32)
    
    agent = IDM(feature_set, std=ctl_std)

    obs = env.reset(1)
    
    # set sim states away
    env._sim_states[0][1] -= 10
    env._sim_states[0][3] -= 3
    
    for t in range(env.T):
        # get true agent control
        # ctl_env = env.get_action()
        # ctl_env = controller.choose_action(obs_env)
        ctl = agent.choose_action(obs, None)
        # print(ctl)
        # break
        
        obs, r, done, info = env.step(ctl)
        
        if done:
            break
    # print(env._sim_acts)
    ani = animate(map_data, env._sim_states, env._track_data, title="test", annot=False)
    save_animation(ani, "/Users/hfml/Documents/test_idm.mp4")
    print("test_idm_agent passed")

def test_sensor_simulator():
    import matplotlib.pyplot as plt
    from src.data.train_utils import load_data
    from src.simulation.utils import create_svt_from_df
    from src.simulation.simulator import InteractionSimulator
    from src.simulation.sensors import EgoSensor, LeadVehicleSensor, LidarSensor
    from src.map_api.lanelet import MapReader
    from src.visualization.animation import animate, save_animation
    from src.data.data_filter import filter_segment_by_length
    
    filepath = os.path.join(data_path, "maps", scenario + ".osm")
    map_data = MapReader(cell_len=10)
    map_data.parse(filepath, verbose=True)
    
    filename = "vehicle_tracks_007.csv"
    df_track = load_data(data_path, scenario, filename)
    df_track = df_track.iloc[:5000].reset_index(drop=True)
    
    svt_object = create_svt_from_df(df_track, eps_id_col="track_id")
    
    num_beams = 20
    ego_sensor = EgoSensor(map_data)
    lv_sensor = LeadVehicleSensor(map_data)
    lidar_sensor = LidarSensor(num_beams)
    sensors = [ego_sensor, lv_sensor, lidar_sensor]
    # sensors = [ego_sensor, lv_sensor]
    
    action_set = ["ax_ego", "ay_ego"]
    env = InteractionSimulator(map_data, sensors, action_set, svt_object)
    
    env.reset(1) # beam too sparse for eps 3
    for t in range(env.T - 1):
        action = env.get_data_action()
        # action = torch.randn(1, 2)
        obs, _, done, _ = env.step(action)
        if done:
            break
    
    ani = animate(
        map_data, env._data, annot=True, axis_on=False
    )
    save_animation(ani, "/Users/hfml/Documents/test_lidar.mp4")

    print("test_sensor_simulator passed")

if __name__ == "__main__":
    # test_simulator_from_data()
    # test_simulator_with_observer()
    # test_observer()
    # test_data_wrapper()
    # test_idm_agent()
    test_sensor_simulator()