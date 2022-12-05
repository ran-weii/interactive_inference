import os
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.agents.vin_agent import VINAgent
from src.agents.hyper_vin_agent import HyperVINAgent
from src.explanation.explainer import VINExplainer
from src.explanation.imagination import simulate_imagination

def load_vin_agent():
    exp_name = "11-13-2022 22-59-15"
    exp_path = os.path.join("../exp", "agents", "vin", exp_name)

    # load config
    with open(os.path.join(exp_path, "args.json"), "rb") as f:
        config = json.load(f)

    # load state dict
    state_dict = torch.load(os.path.join(exp_path, "model.pt"), map_location=torch.device("cpu"))
    state_dict = state_dict if "model_state_dict" not in state_dict.keys() else state_dict["model_state_dict"]
    state_dict = {k.replace("agent.", ""): v.cpu() for (k, v) in state_dict.items() if "agent." in k}

    # load agent
    feature_set = config["feature_set"]
    action_set = config["action_set"]

    obs_dim = len(feature_set)
    ctl_dim = len(action_set)

    agent = VINAgent(
        config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["hmm_rank"],
        config["horizon"], alpha=config["alpha"], beta=config["beta"], obs_model=config["obs_model"],
        obs_cov=config["obs_cov"], ctl_cov=config["ctl_cov"], rwd=config["rwd"]
    )

    agent.load_state_dict(state_dict, strict=False)
    agent.eval()

    for p in agent.parameters():
        p.requires_grad = False
    return agent, feature_set, action_set

def load_hvin_agent():
    exp_name = "11-16-2022 15-38-25"
    exp_path = os.path.join("../exp", "agents", "hvin", exp_name)

    # load config
    with open(os.path.join(exp_path, "args.json"), "rb") as f:
        config = json.load(f)

    # load state dict
    state_dict = torch.load(os.path.join(exp_path, "model.pt"), map_location=torch.device("cpu"))
    state_dict = state_dict if "model_state_dict" not in state_dict.keys() else state_dict["model_state_dict"]
    state_dict = {k.replace("agent.", ""): v.cpu() for (k, v) in state_dict.items() if "agent." in k}

    # load agent
    feature_set = config["feature_set"]
    action_set = config["action_set"]

    obs_dim = len(feature_set)
    ctl_dim = len(action_set)

    agent = HyperVINAgent(
        config["state_dim"], config["act_dim"], obs_dim, ctl_dim, config["hmm_rank"],
        config["horizon"], config["hyper_dim"], config["hidden_dim"], config["num_hidden"],
        config["gru_layers"], config["activation"], alpha=config["alpha"], beta=config["beta"],
        obs_model=config["obs_model"], obs_cov=config["obs_cov"], ctl_cov=config["ctl_cov"], rwd=config["rwd"], 
        hyper_cov=config["hyper_cov"]
    )

    agent.load_state_dict(state_dict, strict=False)
    agent.eval()

    for p in agent.parameters():
        p.requires_grad = False
    return agent, feature_set, action_set

def test_vin_explainer():
    agent, feature_set, action_set = load_vin_agent()

    # init explainer
    by_feature = "lv_inv_tau"
    explainer = VINExplainer(agent, feature_set, action_set)
    explainer.sort_state_action(by_feature)

    # explainer.plot_action_component_pdfs()
    # explainer.plot_transition_matrix()
    # explainer.plot_target_dist(sort_by="lv_inv_tau")
    # explainer.plot_value(sort_by="lv_inv_tau")
    # explainer.plot_policy(sort_by="lv_inv_tau")
    # explainer.plot_controlled_transition_matrix()
    # explainer.plot_controlled_stationary_dist()

    # color_by = "stationary_dist"
    # df = explainer.create_observation_component_data(color_by=color_by, log=True)
    # explainer.plot_observation_scatter_flat(
    #     df[explainer.feature_set], df["state_index"]
    # )
    # explainer.plot_observation_scatter_matrix(
    #     df[explainer.feature_set], df["state_index"], 
    #     plot_component_centers=True, cbar_label=color_by
    # )
    # plt.show()
    
    # fig = explainer.plot_observation_scatter_3d(color_by="stationary_dist")
    # fig.show()

def test_hvin_explainer():
    agent, feature_set, action_set = load_hvin_agent()

    # init explainer
    by_feature = "lv_inv_tau"
    z = torch.tensor([[1., 0., 0., 0.]])
    explainer = VINExplainer(agent, feature_set, action_set, z)
    explainer.sort_state_action(by_feature)

    # explainer.plot_action_component_pdfs()
    # explainer.plot_transition_matrix()
    # explainer.plot_target_dist(sort_by="lv_inv_tau")
    # explainer.plot_policy(sort_by="lv_inv_tau")
    # explainer.plot_controlled_transition_matrix()
    # explainer.plot_controlled_stationary_dist()
    
    # color_by = "initial_belief"
    # df = explainer.create_observation_component_data(color_by=color_by, log=False)
    # explainer.plot_observation_scatter_flat(
    #     df[explainer.feature_set], df["state_index"]
    # )
    # plt.show()

def test_imagination():
    # agent, feature_set, action_set = load_vin_agent()
    agent, feature_set, action_set = load_hvin_agent()
    
    # init explainer
    by_feature = "lv_inv_tau"
    z = torch.tensor([[10., 0., 0., 0.]])
    explainer = VINExplainer(agent, feature_set, action_set, z=z)
    explainer.sort_state_action(by_feature)

    s0 = 0
    max_steps = 1000
    data = simulate_imagination(agent, s0, max_steps, z=z, sample_method="ace")
    
    explainer.plot_belief_simulation_observations(data)
    explainer.plot_belief_simulation_states(data)
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(0)
    # test_vin_explainer()
    # test_hvin_explainer()
    # test_imagination()